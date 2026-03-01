"""
Ambient Clinical Scribe — FastAPI Backend (Azure App Service)
POST /transcribe: audio file → Azure Speech (diarized) → GPT-4o → structured JSON
GET /: serves phone PWA
GET /opnote: serves OpNote Intelligence with optional scribe data injection
GET /encounter/{id}: returns stored encounter JSON (Azure Table Storage)
GET /health: health check
"""

import os, json, tempfile, threading, logging, uuid, base64, time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("scribe")
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.staticfiles import StaticFiles
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
import static_ffmpeg
static_ffmpeg.add_paths()
from pydub import AudioSegment

load_dotenv()

app = FastAPI()

# Serve static files
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ── Azure Table Storage for encounters ───────────────────────────────────────
from azure.data.tables import TableServiceClient

_table_conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
_table_client = None
TABLE_NAME = "encounters"


def get_table():
    """Lazy-init Azure Table client. Falls back to in-memory dict if no connection string."""
    global _table_client
    if _table_client is not None:
        return _table_client
    if not _table_conn:
        log.warning("No AZURE_STORAGE_CONNECTION_STRING — using in-memory store (encounters lost on restart)")
        return None
    svc = TableServiceClient.from_connection_string(_table_conn)
    svc.create_table_if_not_exists(TABLE_NAME)
    _table_client = svc.get_table_client(TABLE_NAME)
    log.info("Connected to Azure Table Storage: %s", TABLE_NAME)
    return _table_client


# In-memory fallback (local dev without Table Storage)
_mem_store: dict[str, dict] = {}


def store_encounter(eid: str, encounter: dict):
    tbl = get_table()
    if tbl:
        entity = {
            "PartitionKey": "encounter",
            "RowKey": eid,
            "data": json.dumps(encounter),
        }
        tbl.upsert_entity(entity)
    else:
        _mem_store[eid] = encounter


def load_encounter(eid: str) -> dict | None:
    tbl = get_table()
    if tbl:
        try:
            entity = tbl.get_entity("encounter", eid)
            return json.loads(entity["data"])
        except Exception:
            return None
    return _mem_store.get(eid)


# ── Prep table (pre-clinic briefings) ─────────────────────────────────────────
_prep_table_client = None
PREP_TABLE = "prep"
_prep_mem: dict[str, dict] = {}


def get_prep_table():
    global _prep_table_client
    if _prep_table_client is not None:
        return _prep_table_client
    if not _table_conn:
        return None
    svc = TableServiceClient.from_connection_string(_table_conn)
    svc.create_table_if_not_exists(PREP_TABLE)
    _prep_table_client = svc.get_table_client(PREP_TABLE)
    return _prep_table_client


def store_prep(date_str: str, patient_id: str, data: dict):
    tbl = get_prep_table()
    entity = {"PartitionKey": date_str, "RowKey": patient_id, "data": json.dumps(data)}
    if tbl:
        tbl.upsert_entity(entity)
    else:
        _prep_mem[f"{date_str}_{patient_id}"] = data


def load_prep_by_date(date_str: str) -> list[dict]:
    tbl = get_prep_table()
    if tbl:
        entities = tbl.query_entities(f"PartitionKey eq '{date_str}'")
        return [json.loads(e["data"]) for e in entities]
    return [v for k, v in _prep_mem.items() if k.startswith(date_str + "_")]


def load_prep_by_id(date_str: str, patient_id: str) -> dict | None:
    tbl = get_prep_table()
    if tbl:
        try:
            e = tbl.get_entity(date_str, patient_id)
            return json.loads(e["data"])
        except Exception:
            return None
    return _prep_mem.get(f"{date_str}_{patient_id}")


def update_prep(date_str: str, patient_id: str, updates: dict):
    existing = load_prep_by_id(date_str, patient_id)
    if not existing:
        return False
    existing.update(updates)
    store_prep(date_str, patient_id, existing)
    return True


# ── Cases table (cross-device OpNote sync) ───────────────────────────────────
_cases_table_client = None
CASES_TABLE = "opnotecases"
_cases_mem: dict[str, dict] = {}


def get_cases_table():
    global _cases_table_client
    if _cases_table_client is not None:
        return _cases_table_client
    if not _table_conn:
        return None
    svc = TableServiceClient.from_connection_string(_table_conn)
    svc.create_table_if_not_exists(CASES_TABLE)
    _cases_table_client = svc.get_table_client(CASES_TABLE)
    return _cases_table_client


def _case_entity_to_dict(entity) -> dict:
    return json.loads(entity["data"])


def _upsert_case(case_id: str, data: dict, updated_at: float, deleted: bool = False):
    tbl = get_cases_table()
    entity = {
        "PartitionKey": "dan",
        "RowKey": case_id,
        "data": json.dumps(data),
        "updatedAt": float(updated_at),
        "deleted": deleted,
    }
    if tbl:
        tbl.upsert_entity(entity)
    else:
        _cases_mem[case_id] = entity


def _load_all_cases() -> list[dict]:
    """Return all non-deleted case entities from storage."""
    tbl = get_cases_table()
    if tbl:
        entities = list(tbl.query_entities("PartitionKey eq 'dan'"))
    else:
        entities = list(_cases_mem.values())
    return entities


# ── Config table (PIN hash storage) ──────────────────────────────────────────
_config_table_client = None
CONFIG_TABLE = "opnoteconfig"
_config_mem: dict[str, str] = {}


def get_config_table():
    global _config_table_client
    if _config_table_client is not None:
        return _config_table_client
    if not _table_conn:
        return None
    svc = TableServiceClient.from_connection_string(_table_conn)
    svc.create_table_if_not_exists(CONFIG_TABLE)
    _config_table_client = svc.get_table_client(CONFIG_TABLE)
    return _config_table_client


def get_pin_hash() -> str | None:
    tbl = get_config_table()
    if tbl:
        try:
            entity = tbl.get_entity("config", "pin_hash")
            return entity.get("data")
        except Exception:
            return None
    return _config_mem.get("pin_hash")


def set_pin_hash(pin_hash: str):
    tbl = get_config_table()
    entity = {"PartitionKey": "config", "RowKey": "pin_hash", "data": pin_hash}
    if tbl:
        tbl.upsert_entity(entity)
    else:
        _config_mem["pin_hash"] = pin_hash


# ── Rate limiting (in-memory) ───────────────────────────────────────────────
_rate_failures: dict[str, list[float]] = {}
RATE_WINDOW = 300  # 5 minutes
RATE_MAX_FAILURES = 50


def _check_rate_limit(client_ip: str) -> bool:
    """Return True if rate-limited (too many failures)."""
    now = time.time()
    failures = _rate_failures.get(client_ip, [])
    failures = [t for t in failures if now - t < RATE_WINDOW]
    _rate_failures[client_ip] = failures
    return len(failures) >= RATE_MAX_FAILURES


def _record_failure(client_ip: str):
    now = time.time()
    failures = _rate_failures.get(client_ip, [])
    failures = [t for t in failures if now - t < RATE_WINDOW]
    failures.append(now)
    _rate_failures[client_ip] = failures


# ── Auth middleware ──────────────────────────────────────────────────────────
PUBLIC_PATHS = {"/health", "/", "/opnote", "/auth/status", "/auth/register"}


class PinAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        # Public paths — no auth needed
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        # Check if a PIN is registered; if not, allow all (first-run)
        stored_hash = get_pin_hash()
        if not stored_hash:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"

        # Rate limit check
        if _check_rate_limit(client_ip):
            return JSONResponse(
                {"error": "rate_limited", "retry_after": RATE_WINDOW},
                status_code=429,
            )

        # Check Authorization header first, then cookie fallback (sendBeacon)
        token = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
        if not token:
            # Cookie fallback for sendBeacon
            cookies = request.cookies
            token = cookies.get("__opnote_token")

        if token == stored_hash:
            return await call_next(request)

        # Auth failed
        _record_failure(client_ip)
        return JSONResponse({"error": "unauthorized"}, status_code=401)


app.add_middleware(PinAuthMiddleware)


# ── Azure OpenAI client ─────────────────────────────────────────────────────
oai = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
)

EXTRACTION_PROMPT_CLINIC = """\
You are a clinical data extractor for a neurosurgery practice.
Extract ONLY factual clinical content from this doctor-patient transcript.
Output valid JSON only. Do not write a note. Do not add information not discussed.
Use short phrases not sentences.
For long encounters, be thorough — capture every clinical fact discussed even if the conversation is 30+ minutes.

Use precise neurosurgical terminology (cervical/thoracic/lumbar disc herniation, stenosis, \
myelopathy, radiculopathy, spondylolisthesis, laminectomy, discectomy, fusion, craniotomy, \
tumor resection, aneurysm, AVM, subdural/epidural hematoma, hydrocephalus, VP shunt, \
Chiari malformation, syringomyelia, etc.)

The encounter type is: {encounter_type}

Return ONLY valid JSON with this exact schema:
{{
  "facts": {{
    "chief_complaint": "string or null",
    "symptoms": ["short phrase"],
    "exam_discussed": ["short phrase"],
    "imaging": ["short phrase"],
    "assessment": "string or null",
    "plan": ["short phrase"],
    "medications": ["short phrase"],
    "laterality": "left|right|bilateral|null"
  }},
  "patient_info": {{
    "name_mentioned": "string or null",
    "age_mentioned": "string or null"
  }}
}}"""

EXTRACTION_PROMPT_OPERATIVE = """\
You are a clinical data extractor for a neurosurgery practice.
Extract ONLY factual operative content from this surgeon's dictation or recording.
Output valid JSON only. Do not write a note. Do not add information not discussed.
Use short phrases not sentences.

Use precise neurosurgical terminology (ACDF, TLIF, laminectomy, discectomy, fusion, craniotomy, \
tumor resection, pedicle screws, interbody cage, rod, plate, allograft, autograft, etc.)

The encounter type is: {encounter_type}

Return ONLY valid JSON with this exact schema:
{{
  "facts": {{
    "procedure_performed": "string or null",
    "approach": "string or null",
    "findings": ["short phrase"],
    "implants": ["short phrase"],
    "ebl": "string or null",
    "complications": "string or null",
    "specimens": ["short phrase"]
  }},
  "patient_info": {{
    "name_mentioned": "string or null",
    "age_mentioned": "string or null"
  }}
}}"""


# Formats Azure Speech SDK can read natively (no conversion needed)
NATIVE_SPEECH_EXTS = {".wav", ".ogg", ".mp3", ".flac"}


def transcribe_and_extract_direct(audio_path: str, encounter_type: str) -> tuple[str, dict]:
    """Single GPT-4o call: audio in → structured JSON out. ~15-30s total."""
    # Ensure WAV for reliable audio input
    wav_path = convert_to_wav(audio_path)
    target = wav_path or audio_path
    try:
        with open(target, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

    prompt = (EXTRACTION_PROMPT_OPERATIVE if encounter_type == "OpNote" else EXTRACTION_PROMPT_CLINIC).format(encounter_type=encounter_type)
    prompt += '\n\nAlso include a top-level "transcript" field with a clean verbatim transcript of the audio.'

    response = oai.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "text", "text": "Transcribe and extract structured clinical data from this recording."},
            ]},
        ],
        temperature=0.2,
        max_completion_tokens=4000,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    transcript = result.pop("transcript", "")
    return transcript, result


def convert_to_wav(input_path: str) -> str | None:
    """Convert to WAV only if Azure Speech SDK can't read the format natively."""
    ext = Path(input_path).suffix.lower()
    if ext in NATIVE_SPEECH_EXTS:
        log.info(f"Native format ({ext}), skipping conversion")
        return None
    wav_path = input_path + ".wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
    audio.export(wav_path, format="wav")
    return wav_path


def transcribe_audio(audio_path: str) -> str:
    """Run Azure Speech Service conversation transcription with diarization on an audio file."""
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
    )
    speech_config.speech_recognition_language = "en-US"
    speech_config.set_property(
        speechsdk.PropertyId.SpeechServiceResponse_DiarizeIntermediateResults, "true"
    )

    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    transcriber = speechsdk.transcription.ConversationTranscriber(
        speech_config=speech_config, audio_config=audio_config
    )

    lines: list[str] = []
    done = threading.Event()
    error_msg = None

    def on_transcribed(evt):
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            speaker = evt.result.speaker_id or "Unknown"
            lines.append(f"[{speaker}]: {evt.result.text}")

    def on_canceled(evt):
        nonlocal error_msg
        if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
            error_msg = evt.cancellation_details.error_details
        done.set()

    transcriber.transcribed.connect(on_transcribed)
    transcriber.canceled.connect(on_canceled)
    transcriber.session_stopped.connect(lambda evt: done.set())

    transcriber.start_transcribing_async().get()
    done.wait(timeout=900)  # 15 min max (supports 30-min encounters)
    transcriber.stop_transcribing_async().get()

    if error_msg:
        raise RuntimeError(f"Speech transcription error: {error_msg}")

    return "\n".join(lines)


def extract_structured(transcript: str, encounter_type: str) -> dict:
    """Send diarized transcript to GPT-4o, get structured JSON back."""
    response = oai.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        messages=[
            {"role": "system", "content": (EXTRACTION_PROMPT_OPERATIVE if encounter_type == "OpNote" else EXTRACTION_PROMPT_CLINIC).format(encounter_type=encounter_type)},
            {"role": "user", "content": f"Diarized transcript:\n\n{transcript}"},
        ],
        temperature=0.2,
        max_completion_tokens=2000,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ── Encounter type mapping: scribe → OpNote note types ──
ENCOUNTER_TO_NT = {
    "Clinic - New Patient": "clinic",
    "Clinic - Follow-up": "clinic",
    "Clinic - Post-Op": "clinic",
    "Clinic - Established": "clinic",
    "Consult": "consult",
    "Progress Note": "progress",
    "OpNote": "operative",
    # Legacy types (backward compat)
    "New Patient": "clinic",
    "Follow-up": "clinic",
    "Post-Op": "clinic",
    "Pre-Op": "clinic",
}
ENCOUNTER_TO_VISIT = {
    "Clinic - New Patient": "new",
    "Clinic - Follow-up": "followup",
    "Clinic - Post-Op": "postop",
    "Clinic - Established": "followup",
    "Consult": "new",
    "Progress Note": "followup",
    "OpNote": "new",
    "New Patient": "new",
    "Follow-up": "followup",
    "Post-Op": "postop",
    "Pre-Op": "new",
}
ENCOUNTER_TO_PT = {
    "Clinic - New Patient": "new",
    "Clinic - Follow-up": "established",
    "Clinic - Post-Op": "established",
    "Clinic - Established": "established",
    "Consult": "new",
    "Progress Note": "established",
    "OpNote": "new",
    "New Patient": "new",
    "Follow-up": "established",
    "Post-Op": "established",
    "Pre-Op": "established",
}


# ── Patient data redaction ───────────────────────────────────────────────────
def _redact_case(case: dict, index: int) -> dict:
    """Strip patient-identifying data from a case, replacing with Patient #N."""
    label = f"Patient #{index}"
    if "patient_info" in case:
        pi = case["patient_info"]
        if isinstance(pi, dict):
            if pi.get("name_mentioned"):
                pi["name_mentioned"] = label
    if case.get("patient_name"):
        case["patient_name"] = label
    return case


def _redact_prep(patient: dict, index: int) -> dict:
    """Strip patient name from a prep record, replacing with Patient #N."""
    if patient.get("patient_name"):
        patient["patient_name"] = f"Patient #{index}"
    return patient


@app.get("/speech-token")
async def speech_token():
    """Return a short-lived Azure Speech token so the frontend can use the Speech SDK
    without exposing the subscription key."""
    import urllib.request
    region = os.getenv("AZURE_SPEECH_REGION", "eastus")
    key = os.getenv("AZURE_SPEECH_KEY", "")
    if not key:
        return JSONResponse({"error": "Speech not configured"}, status_code=500)
    url = f"https://{region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
    req = urllib.request.Request(url, data=b"", headers={
        "Ocp-Apim-Subscription-Key": key,
        "Content-Length": "0",
    })
    try:
        with urllib.request.urlopen(req) as resp:
            token = resp.read().decode()
        return {"token": token, "region": region}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/extract")
async def extract(request: Request):
    """Text-only fact extraction — no audio. For use with frontend real-time transcription."""
    try:
        body = await request.json()
        transcript = body.get("transcript", "")
        encounter_type = body.get("encounter_type", "Clinic - Established")
        if not transcript.strip():
            return JSONResponse({"error": "No transcript provided"}, status_code=400)

        extracted = extract_structured(transcript, encounter_type)

        eid = uuid.uuid4().hex[:12]
        encounter = {
            "source": "realtime_scribe",
            "encounter_type": encounter_type,
            "raw_transcript": transcript,
            "facts": extracted.get("facts", {}),
            "patient_info": extracted.get("patient_info", {}),
            "opnote_nt": ENCOUNTER_TO_NT.get(encounter_type, "clinic"),
            "opnote_visit": ENCOUNTER_TO_VISIT.get(encounter_type, "new"),
            "opnote_pt": ENCOUNTER_TO_PT.get(encounter_type, "new"),
        }
        store_encounter(eid, encounter)
        log.info(f"Stored encounter {eid} (realtime)")

        redacted = _redact_case(encounter, 1)
        return {
            "encounter_id": eid,
            "opnote_url": f"/opnote?scribe={eid}",
            "facts": redacted["facts"],
            "patient_info": redacted.get("patient_info", {}),
            "encounter_type": encounter_type,
        }
    except Exception as e:
        log.exception("Extract endpoint error")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health")
async def health():
    return {"status": "ok", "storage": "table" if _table_conn else "memory"}


@app.get("/auth/status")
async def auth_status():
    return {"registered": get_pin_hash() is not None}


@app.post("/auth/register")
async def auth_register(request: Request):
    body = await request.json()
    pin_hash = body.get("pin_hash", "")
    if not pin_hash or len(pin_hash) != 64:
        return JSONResponse({"error": "Invalid pin_hash"}, status_code=400)
    if get_pin_hash() is not None:
        return JSONResponse({"error": "PIN already registered"}, status_code=409)
    set_pin_hash(pin_hash)
    return {"ok": True}


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "opnote.html"), media_type="text/html")


@app.get("/opnote")
async def opnote():
    return FileResponse(str(STATIC_DIR / "opnote.html"), media_type="text/html")


@app.get("/encounter/{encounter_id}")
async def get_encounter(encounter_id: str):
    enc = load_encounter(encounter_id)
    if enc is None:
        return JSONResponse({"error": "Encounter not found"}, status_code=404)
    return _redact_case(enc, 1)


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    encounter_type: str = Form("Follow-up"),
):
    # Save uploaded audio to temp file
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    wav_path = None
    try:
        log.info(f"Received audio: {audio.filename} ({audio.content_type})")
        transcript = None
        extracted = None

        # Fast path: single GPT-4o call with direct audio input
        try:
            log.info("Trying direct GPT-4o audio input (fast path)...")
            transcript, extracted = transcribe_and_extract_direct(tmp_path, encounter_type)
            log.info(f"Direct audio done: transcript={len(transcript)} chars")
        except Exception as fast_err:
            log.info(f"Direct audio not supported ({fast_err}), falling back to Azure Speech...")

        # Slow fallback: Azure Speech transcription + GPT-4o extraction
        if extracted is None:
            wav_path = convert_to_wav(tmp_path)
            audio_for_sdk = wav_path or tmp_path
            if wav_path:
                log.info(f"Converted to WAV: {wav_path}")
            log.info("Starting Azure Speech transcription...")
            transcript = transcribe_audio(audio_for_sdk)
            log.info(f"Transcription done: {len(transcript)} chars")
            if not transcript.strip():
                return JSONResponse(
                    {"error": "No speech detected in audio"},
                    status_code=400,
                )
            log.info("Sending to GPT-4o for extraction...")
            extracted = extract_structured(transcript, encounter_type)
            log.info("Extraction complete")

        # Step 3: Build encounter record and persist
        eid = uuid.uuid4().hex[:12]
        encounter = {
            "source": "ambient_scribe",
            "encounter_type": encounter_type,
            "raw_transcript": transcript or "",
            "facts": extracted.get("facts", {}),
            "patient_info": extracted.get("patient_info", {}),
            "opnote_nt": ENCOUNTER_TO_NT.get(encounter_type, "clinic"),
            "opnote_visit": ENCOUNTER_TO_VISIT.get(encounter_type, "new"),
            "opnote_pt": ENCOUNTER_TO_PT.get(encounter_type, "new"),
        }
        store_encounter(eid, encounter)
        log.info(f"Stored encounter {eid}")

        redacted = _redact_case(encounter, 1)
        return {
            "encounter_id": eid,
            "opnote_url": f"/opnote?scribe={eid}",
            "facts": redacted["facts"],
            "patient_info": redacted.get("patient_info", {}),
            "encounter_type": encounter_type,
        }
    except Exception as e:
        log.exception("Transcribe endpoint error")
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        os.unlink(tmp_path)
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/prep")
async def get_prep(date: str = ""):
    if not date:
        from datetime import date as dt
        date = dt.today().isoformat()
    patients = load_prep_by_date(date)
    return {"date": date, "patients": patients}


@app.post("/prep")
async def post_prep(request: Request):
    body = await request.json()
    date_str = body.get("date")
    # Bulk save: {date, patients: [...]}
    patients = body.get("patients")
    if patients:
        for p in patients:
            pid = p.get("id") or uuid.uuid4().hex[:8]
            p["id"] = pid
            store_prep(date_str, pid, p)
        return {"ok": True, "count": len(patients)}
    # Single save: {date, patient: {...}}
    patient = body.get("patient", {})
    pid = patient.get("id") or uuid.uuid4().hex[:8]
    patient["id"] = pid
    store_prep(date_str, pid, patient)
    return {"ok": True, "id": pid}


@app.patch("/prep/{patient_id}")
async def patch_prep(patient_id: str, request: Request):
    body = await request.json()
    date_str = body.get("date")
    if not date_str:
        from datetime import date as dt
        date_str = dt.today().isoformat()
    updates = body.get("updates", {})
    ok = update_prep(date_str, patient_id, updates)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {"ok": True}


@app.post("/prep/parse")
async def parse_schedule(request: Request):
    body = await request.json()
    text = body.get("text", "")
    if not text.strip():
        return JSONResponse({"error": "No text"}, status_code=400)
    resp = oai.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        messages=[
            {"role": "system", "content": "Parse this clinic schedule into JSON array: [{\"patient_name\":\"Last, First\",\"appointment_time\":\"HH:MM\",\"is_surgery\":bool,\"visit_type\":\"OpNote|Follow-up|New Patient|Post-Op|Consult\"}]. 07xx = surgery. Return ONLY valid JSON array."},
            {"role": "user", "content": text}
        ],
        max_completion_tokens=4096,
        temperature=0.1,
    )
    raw = resp.choices[0].message.content or "[]"
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
    try:
        patients = json.loads(raw.strip())
    except json.JSONDecodeError:
        return JSONResponse({"error": "Failed to parse AI response"}, status_code=500)
    patients = [_redact_prep(p, i + 1) for i, p in enumerate(patients)]
    return {"patients": patients}


@app.get("/cases")
async def get_cases():
    """Return all non-deleted cases from Azure Table Storage."""
    entities = _load_all_cases()
    cases = []
    for e in entities:
        if e.get("deleted"):
            continue
        try:
            c = json.loads(e["data"])
            c["_updatedAt"] = e.get("updatedAt", 0)
            cases.append(c)
        except Exception:
            pass
    cases = [_redact_case(c, i + 1) for i, c in enumerate(cases)]
    return {"cases": cases}


@app.post("/cases/sync")
async def sync_cases(request: Request):
    """Merge client cases with server. Newer _updatedAt wins per case ID."""
    body = await request.json()
    client_cases = body.get("cases", [])

    # Load current server state into a dict keyed by case id
    entities = _load_all_cases()
    server_map: dict[str, dict] = {}  # id -> entity-like dict
    for e in entities:
        rid = e.get("RowKey") or e.get("id", "")
        server_map[rid] = e

    # Merge: client wins if newer
    for cc in client_cases:
        cid = cc.get("id")
        if not cid:
            continue
        c_ts = cc.get("_updatedAt", 0)
        existing = server_map.get(cid)
        if existing:
            s_ts = existing.get("updatedAt", 0)
            if existing.get("deleted"):
                # Server says deleted — skip client update
                continue
            if c_ts > s_ts:
                _upsert_case(cid, cc, c_ts)
                server_map[cid] = {"RowKey": cid, "data": json.dumps(cc), "updatedAt": c_ts, "deleted": False}
        else:
            _upsert_case(cid, cc, c_ts)
            server_map[cid] = {"RowKey": cid, "data": json.dumps(cc), "updatedAt": c_ts, "deleted": False}

    # Build merged result (non-deleted)
    merged = []
    for rid, e in server_map.items():
        if e.get("deleted"):
            continue
        try:
            c = json.loads(e["data"])
            c["_updatedAt"] = e.get("updatedAt", 0)
            merged.append(c)
        except Exception:
            pass

    merged = [_redact_case(c, i + 1) for i, c in enumerate(merged)]
    return {"cases": merged}


@app.post("/cases/delete")
async def delete_case(request: Request):
    """Soft-delete a case on the server so other devices remove it."""
    body = await request.json()
    cid = body.get("id")
    if not cid:
        return JSONResponse({"error": "Missing id"}, status_code=400)
    _upsert_case(cid, {}, 0, deleted=True)
    return {"ok": True}


@app.post("/generate")
async def generate(request: Request):
    """Server-side LLM proxy — calls Azure OpenAI and returns content."""
    try:
        body = await request.json()
        messages = body.get("messages")
        system_msg = body.get("system")
        user_msg = body.get("user")
        max_completion_tokens = body.get("max_completion_tokens", body.get("max_tokens", 8192))
        temperature = body.get("temperature", 0.3)

        if messages:
            # Full messages mode (supports image_url content)
            chat_msgs = messages
        elif system_msg or user_msg:
            # Simple mode: build messages from system + user strings
            chat_msgs = []
            if system_msg:
                chat_msgs.append({"role": "system", "content": system_msg})
            if user_msg:
                chat_msgs.append({"role": "user", "content": user_msg})
        else:
            return JSONResponse({"error": "Provide 'messages' array or 'system'+'user' strings"}, status_code=400)

        response = oai.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=chat_msgs,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
        stop_reason = "max_tokens" if choice.finish_reason == "length" else "end_turn"
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        } if response.usage else {}

        return {"content": content, "stop_reason": stop_reason, "usage": usage}
    except Exception as e:
        log.exception("Generate endpoint error")
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
