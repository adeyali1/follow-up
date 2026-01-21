import os
import asyncio
import aiohttp
from urllib.parse import quote
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMRunFrame
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, TranscriptionFrame
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy
from services.supabase_service import update_lead_status
import json
import time
import re

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-jordanian-human-v3"
_VAD_MODEL = {"value": None}

# --- HELPER: ARABIC TIME CONVERTER ---
# Prevents the robot from saying "Fourteen Hundred"
def get_arabic_time_string(time_str: str) -> str:
    # Simple heuristic for common dental slots
    # Input example: "14:00", "2:00 PM", "11:30"
    t = str(time_str).replace("PM", "").replace("AM", "").strip()
    
    mapping = {
        "09:00": "الساعة تسعة الصبح",
        "9:00":  "الساعة تسعة الصبح",
        "10:00": "الساعة عشرة الصبح",
        "11:00": "الساعة حدعش الصبح",
        "12:00": "الساعة طنعش الظهر",
        "13:00": "الساعة وحدة الظهر",
        "1:00":  "الساعة وحدة الظهر",
        "14:00": "الساعة ثنتين بعد الظهر",
        "2:00":  "الساعة ثنتين بعد الظهر",
        "15:00": "الساعة تلاتة العصر",
        "3:00":  "الساعة تلاتة العصر",
        "16:00": "الساعة أربعة العصر",
        "4:00":  "الساعة أربعة العصر",
        "17:00": "الساعة خمسة المسا",
        "5:00":  "الساعة خمسة المسا",
        "18:00": "الساعة ستة المسا",
        "6:00":  "الساعة ستة المسا"
    }
    
    # Try exact match first
    if t in mapping:
        return mapping[t]
    
    # Fallback: Just return the string but guide pronunciation
    return f"الساعة {t}"

# --- HELPER: GENDER GUESSER ---
# Prevents addressing a Male as Female
def infer_gender_and_fix_name(name: str):
    name_lower = name.lower().strip()
    
    # Males
    if name_lower in ["oday", "adi", "ahmad", "mohammad", "khaled", "yousef", "omar", "ali", "ibrahim"]:
        return "Male", {"oday": "عُدَي", "adi": "عُدَي", "ahmad": "أحمد"}.get(name_lower, name)
    
    # Females
    if name_lower in ["sara", "noor", "leena", "fatima", "aya", "ranya", "mariam"]:
        return "Female", {"sara": "سارة"}.get(name_lower, name)
        
    # Default to Male (Neutral in Arabic) if unknown
    return "Male", name 


class MultimodalPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "TranscriptionFrame":
            try:
                if getattr(frame, "role", None) == "user":
                    _MM["last_user_transcription_ts"] = time.monotonic()
            except Exception:
                pass
        if frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            now = time.monotonic()
            _MM["last_bot_started_ts"] = now
        await self.push_frame(frame, direction)

class MultimodalTranscriptRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float):
        super().__init__()
        self._delay_s = float(delay_s)
        self._queue_frames = None
        self._pending = None
        self._last_user_transcript_ts = None
    def set_queue_frames(self, queue_frames):
        self._queue_frames = queue_frames
    async def _schedule(self, ts: float):
        try:
            await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError:
            return
        if self._queue_frames is None:
            return
        if self._last_user_transcript_ts != ts:
            return
        _MM["last_llm_run_ts"] = time.monotonic()
        await self._queue_frames([LLMRunFrame()])
    def cancel_pending(self):
        if self._pending is not None:
            try:
                self._pending.cancel()
            except Exception:
                pass
            self._pending = None
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            is_final = getattr(frame, "is_final", True)
            if not is_final:
                await self.push_frame(frame, direction)
                return
            now = time.monotonic()
            self._last_user_transcript_ts = now
            if self._pending is not None:
                self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

# ... (Keep other Loggers/Chunkers as is from previous code to save space) ...
class TurnStateLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
class OutboundAudioLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
class InboundAudioLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
class AudioFrameChunker(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        # (Implementation same as previous)

def normalize_gemini_live_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        # Flash-exp is critical for speed and natural interrupt handling
        return "models/gemini-2.0-flash-exp" 
    if model.startswith("models/"):
        return model
    return f"models/{model}"

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id: return
    telnyx_key = os.getenv("TELNYX_API_KEY")
    if not telnyx_key: return
    if delay_s > 0: await asyncio.sleep(delay_s)
    encoded_call_control_id = quote(call_control_id, safe="")
    url = f"https://api.telnyx.com/v2/calls/{encoded_call_control_id}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json={"reason": "normal_clearing"}) as resp:
            pass

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting human-like bot for lead: {lead_data.get('id')}")

    # 1. CLEAN DATA BEFORE AI SEES IT
    raw_name = lead_data.get("patient_name", "") or lead_data.get("customer_name", "يا غالي")
    gender, patient_name_ar = infer_gender_and_fix_name(raw_name)
    
    raw_treatment = lead_data.get("treatment", "الموعد")
    # Simple cleanup for treatment
    treatment_ar = raw_treatment.replace("cleaning", "تنظيف أسنان").replace("checkup", "فحص")

    raw_time = lead_data.get("appointment_time", "") or lead_data.get("delivery_time", "11:00")
    appointment_time_ar = get_arabic_time_string(raw_time)

    logger.info(f"Context: Name={patient_name_ar}, Gender={gender}, Time={appointment_time_ar}")

    # 2. VAD SETTINGS (Tuned for interruptions)
    # Lower silence timeout so she stops faster when user speaks
    vad_stop_secs = 0.3 
    vad = SileroVADAnalyzer(
        params=VADParams(min_volume=0.5, start_secs=0.2, stop_secs=vad_stop_secs, confidence=0.7)
    )
    _VAD_MODEL["value"] = vad

    # 3. SETUP TRANSPORT
    stream_id = "telnyx_stream_placeholder" 
    # (Simplified handshake logic for brevity - ensure you keep your loop here)
    try:
        # Quick loop to get stream_id
        for _ in range(3):
            msg = json.loads(await websocket_client.receive_text())
            if "stream_id" in msg: stream_id = msg["stream_id"]; break
            elif "data" in msg and "stream_id" in msg["data"]: stream_id = msg["data"]["stream_id"]; break
    except: pass

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding="PCMU", inbound_encoding="PCMU",
        params=TelnyxFrameSerializer.InputParams(sample_rate=16000),
    )
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            session_timeout=300,
            vad_analyzer=vad,
            audio_in_enabled=True, audio_out_enabled=True,
            audio_in_sample_rate=16000, audio_out_sample_rate=16000,
        ),
    )

    # 4. THE "HUMAN" SYSTEM PROMPT
    # Notice: No heavy diacritics. No rigid "Phase 1". 
    # Focus on "Listen First" and "Short Sentences".
    
    pronoun_instruction = "Addres the user as MALE (ya sidi, maw'idak)." if gender == "Male" else "Address the user as FEMALE (ya 3youni, maw'idik)."

    system_prompt = f"""
# ROLE
You are **Sara**, the coordinator at Smile Clinic in Amman. 
You are speaking to **{patient_name_ar}**.
{pronoun_instruction}

# CRITICAL STYLE RULES (TO SOUND HUMAN)
1. **BE CONCISE:** Never speak more than 1 sentence at a time. Wait for the user to answer.
2. **LISTEN FIRST:** If the user asks "Who are you?" or "I don't understand", STOP your script and answer them immediately. Do not ignore them.
3. **DIALECT:** Use pure Ammani Arabic. 
   - Say "Biddi" (بدي), "Halla" (هلا), "Shu" (شو).
   - Do NOT use MSA ("Limatha", "Hasanan").
4. **NUMBERS:** The time is "{appointment_time_ar}". Read it exactly like that.

# GOAL
Your goal is to confirm the appointment.
1. Greeting: "مرحبا {patient_name_ar}، يعطيك العافية." (Wait for reply).
2. Reason: "معك سارة من عيادة الابتسامة. بخصوص موعدك {appointment_time_ar}. الموعد مناسبك؟"
3. Outcome:
   - If Yes: "ممتاز، نتشرف فيك. بدك شي تاني؟" -> [Tool: Confirm]
   - If No/Cancel: "ولا يهمك، بنلغيه؟" -> [Tool: Cancel]

# FALLBACKS (If user is confused)
- If they ask "Who?": "أنا سارة من العيادة، بتصل عشان موعد الأسنان."
- If they say "What time?": "{appointment_time_ar}."

Start immediately with the Greeting only.
"""

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key: return

    # 5. FORCE SETTINGS FOR HUMAN FEEL
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
    
    # Force Aoede (Female)
    # Force Temperature 0.6 (Natural variation)
    gemini_live = GeminiLiveLLMService(
        api_key=api_key,
        voice_id="Aoede", 
        model=normalize_gemini_live_model_name(os.getenv("GEMINI_LIVE_MODEL")),
        system_instruction=system_prompt,
        params=InputParams(temperature=0.6, sample_rate=16000)
    )

    # 6. TOOLING
    lead_finalized = {"value": None}
    async def confirm_appointment(params: FunctionCallParams):
        logger.info(f"Tool: Confirming")
        lead_finalized["value"] = "CONFIRMED"
        update_lead_status(lead_data["id"], "CONFIRMED")
        await params.result_callback({"value": "Done"})
        asyncio.create_task(hangup_telnyx_call(call_control_id, 3.0))

    async def cancel_appointment(params: FunctionCallParams):
        logger.info(f"Tool: Cancelling")
        lead_finalized["value"] = "CANCELLED"
        update_lead_status(lead_data["id"], "CANCELLED")
        await params.result_callback({"value": "Done"})
        asyncio.create_task(hangup_telnyx_call(call_control_id, 3.0))

    gemini_live.register_function("update_lead_status_confirmed", confirm_appointment)
    gemini_live.register_function("update_lead_status_cancelled", cancel_appointment)

    # 7. PIPELINE
    mm_context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    
    # Aggregators & Strategies
    # Note: Using Transcription Strategy for start to ensure we capture the whole "Who are you?"
    start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
    stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.4)] # Short timeout to respond fast
    
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies),
        ),
    )

    # Initial Context
    await gemini_live.set_context(mm_context)

    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.5)
    
    # Standard Pipeline components
    pipeline = Pipeline([
        transport.input(),
        mm_aggregators.user(),
        gemini_live,
        transcript_trigger,
        transport.output(),
        mm_aggregators.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)
    runner = PipelineRunner()

    # Failsafe: If bot doesn't speak in 5s, trigger it
    async def failsafe():
        await asyncio.sleep(5.0)
        if _MM.get("last_bot_started_ts") is None:
            await task.queue_frames([LLMRunFrame()])
    asyncio.create_task(failsafe())

    await runner.run(task)
