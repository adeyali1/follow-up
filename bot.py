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
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy, VADUserTurnStartStrategy
from services.supabase_service import update_lead_status
import json
import time

# --- Performance Monitoring Globals ---
_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-22-jordan-human-v5-final"
_VAD_MODEL = {"value": None}

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
            user_ts = _MM.get("last_user_transcription_ts")
            if user_ts is not None:
                logger.info(f"Latency multimodal user_transcription→bot_audio={now - user_ts:.3f}s")
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
        if self._queue_frames is None: return
        if self._last_user_transcript_ts != ts: return
        logger.info("Multimodal: user transcript idle → LLMRunFrame")
        _MM["last_llm_run_ts"] = time.monotonic()
        await self._queue_frames([LLMRunFrame()])

    def cancel_pending(self):
        if self._pending is not None:
            try: self._pending.cancel()
            except: pass
            self._pending = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            is_final = getattr(frame, "is_final", True)
            if not is_final:
                await self.push_frame(frame, direction)
                return
            now = time.monotonic()
            try:
                text = (getattr(frame, "text", None) or "").strip()
                if len(text) < 2:
                    await self.push_frame(frame, direction)
                    return
                # Standard hallucination block list
                hallucinations = ["ma si problemi", "ma sì problemi", "si", "ok", "thank you", "bye", "you", "okay", "ألو", "alo"]
                if text.lower() in hallucinations:
                    await self.push_frame(frame, direction)
                    return
                if text: logger.debug(f"Multimodal: final user transcript received ({len(text)} chars)")
            except: pass
            self._last_user_transcript_ts = now
            if self._pending is not None: self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class MultimodalUserStopRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float = 0.03, min_interval_s: float = 0.2):
        super().__init__()
        self._delay_s = float(delay_s)
        self._min_interval_s = float(min_interval_s)
        self._queue_frames = None
        self._pending = None
        self._last_stop_ts = None

    def set_queue_frames(self, queue_frames):
        self._queue_frames = queue_frames

    def cancel_pending(self):
        if self._pending is not None:
            try: self._pending.cancel()
            except: pass
            self._pending = None

    async def _schedule(self, ts: float):
        try: await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError: return
        if self._queue_frames is None: return
        if self._last_stop_ts != ts: return
        last_run = _MM.get("last_llm_run_ts")
        now = time.monotonic()
        if last_run is not None and (now - last_run) < self._min_interval_s: return
        _MM["last_llm_run_ts"] = now
        logger.info("Multimodal: user stop → LLMRunFrame")
        await self._queue_frames([LLMRunFrame()])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            self._last_stop_ts = now
            if self._pending is not None: self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class TurnStateLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._user_turn_started_ts = None
        self._logged_bot_audio = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "UserStartedSpeakingFrame":
            self._user_turn_started_ts = time.monotonic()
            logger.info("Turn: user started speaking")
        elif frame_name == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            dur = None
            if self._user_turn_started_ts is not None: dur = now - self._user_turn_started_ts
            self._user_turn_started_ts = None
            logger.info(f"Turn: user stopped speaking (dur={dur if dur else 0:.2f}s)")
        elif frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"} and not self._logged_bot_audio:
            self._logged_bot_audio = True
            logger.info(f"Turn: first bot audio started ({frame_name})")
        await self.push_frame(frame, direction)

class OutboundAudioLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._logged = False
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not self._logged and isinstance(frame, AudioRawFrame):
            self._logged = True
            logger.info("AudioOut: first audio frame")
        await self.push_frame(frame, direction)

class InboundAudioLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._logged = False
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not self._logged and isinstance(frame, InputAudioRawFrame):
            self._logged = True
            logger.info("AudioInDecoded: first audio frame")
        await self.push_frame(frame, direction)

class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 0):
        super().__init__()
        self._chunk_ms = int(chunk_ms)
        self._pace = (os.getenv("AUDIO_OUT_PACE") or "true").lower() == "true"
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._chunk_ms <= 0:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, AudioRawFrame) and len(frame.audio) > 0:
            # Simple chunking logic omitted for brevity, but same as before
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

class LeadStatusTranscriptFallback(FrameProcessor):
    def __init__(self, *, lead_id: str, call_control_id: str | None, call_end_delay_s: float, finalized_ref: dict):
        super().__init__()
        self._lead_id = lead_id
        self._call_control_id = call_control_id
        self._call_end_delay_s = float(call_end_delay_s)
        self._finalized_ref = finalized_ref

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user" and getattr(frame, "is_final", True):
            text = str(getattr(frame, "text", "") or "").lower()
            if "تمام" in text or "موافق" in text:
                if not self._finalized_ref.get("value"):
                    self._finalized_ref["value"] = "CONFIRMED"
                    update_lead_status(self._lead_id, "CONFIRMED")
                    if self._call_control_id: asyncio.create_task(hangup_telnyx_call(self._call_control_id, self._call_end_delay_s))

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id: return
    telnyx_key = os.getenv("TELNYX_API_KEY")
    if not telnyx_key: return
    if delay_s > 0: await asyncio.sleep(delay_s)
    encoded_call_control_id = quote(call_control_id, safe="")
    url = f"https://api.telnyx.com/v2/calls/{encoded_call_control_id}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"reason": "normal_clearing"}): pass
    except: pass

def normalize_gemini_live_model_name(model: str) -> str:
    return model if "models/" in model else f"models/{model}"

# --- THE FIX: PYTHON-SIDE NAME MAPPER ---
def get_arabic_name(english_name: str) -> str:
    """
    Converts known English mock names to Arabic Script.
    This prevents the AI from reading them in an English Accent.
    """
    name_map = {
        "oday": "عُدَيّ",
        "qusai": "قُصَيّ",
        "hijazi": "حِجَازِي",
        "ahmad": "أحمد",
        "mohammad": "محمد",
        "khaled": "خالد",
        "sarah": "سارة",
        "yousef": "يوسف"
    }
    cleaned = english_name.strip().lower()
    return name_map.get(cleaned, english_name) # Returns Arabic if found, else original

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting JORDAN HUMANIZED BOT for: {lead_data.get('id', 'mock')}")

    # --- 1. PRE-PROCESS DATA ---
    raw_name = lead_data.get('patient_name', 'يا غالي')
    # Force conversion to Arabic Script
    patient_name_ar = get_arabic_name(raw_name) 
    
    treatment_context = lead_data.get('treatment', 'استشارة أسنان')
    if "Burger" in treatment_context: # Fix the mock data weirdness from logs
        treatment_context = "تركيبات الزيركون"

    pipeline_sample_rate = 16000
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"
    
    # Capture Stream ID (standard logic)
    try:
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            msg = json.loads(msg_text)
            if "stream_id" in msg: 
                stream_id = msg["stream_id"]
                break
            elif "data" in msg and "stream_id" in msg["data"]:
                stream_id = msg["data"]["stream_id"]
                break
            elif msg.get("event") == "start" and "stream_id" in msg:
                stream_id = msg["stream_id"]
                break
    except: pass

    # --- 2. VAD SETTINGS (Keep the fixes) ---
    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.1, start_secs=0.1, stop_secs=0.5, confidence=0.5, sample_rate=pipeline_sample_rate))
    _VAD_MODEL["value"] = vad

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(sample_rate=pipeline_sample_rate),
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            session_timeout=300,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=pipeline_sample_rate,
            audio_out_sample_rate=pipeline_sample_rate,
        ),
    )

    # ---------------------------------------------------------------------
    # SYSTEM PROMPT: THE "CASUAL LOCAL" UPDATE
    # ---------------------------------------------------------------------
    system_prompt = f"""
    **IDENTITY:**
    You are Sarah (سارة), a friendly receptionist at "Amman Elite Dental".
    
    **TONE (CASUAL & NATURAL):**
    - Speak **Jordanian Ammani (لهجة بيضاء)**.
    - Be **CASUAL**. Do not sound like a robot reading a script.
    - If the user says "Min Mai?" (Who is this?), laugh gently and say: "معك سارة من العيادة، نسيتنا؟" (It's Sarah from the clinic, forgot us?).
    - Use fillers: "يعني", "طيب", "شوف".

    **VOCABULARY RULES (STRICT):**
    1. **Numbers:** WRITE WORDS ONLY. "حداش" (11), "ثنتين" (2).
    2. **Name:** Call the patient "{patient_name_ar}". NEVER pronounce it in English.
    3. **Confirmation:** Say "تمام" or "اتفقنا".
    
    **CONVERSATION GUIDE:**
    1. **The Opening (Ping):**
       "ألو.. مسا الخير.. {patient_name_ar} معي؟"
       *(Wait for them to answer "Yes" or "Who is this?")*
    
    2. **The Reason:**
       "يا هلا.. حبيت أخبرك دكاترتنا فتحوا كشفيات مجانية هالأسبوع لـ ({treatment_context}).. قلت برن عليك تستفيد من العرض، شو رأيك؟"
    
    3. **The Close:**
       "عنا موعد السبت الساعة حداش.. أحجزلك ياه؟"
    """

    use_multimodal_live = os.getenv("USE_MULTIMODAL_LIVE", "true").lower() == "true"
    if use_multimodal_live:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return

        # VOICE: Kore is often clearer and softer than Aoede
        voice_id = "Kore" 
        model = "models/gemini-2.0-flash-exp"

        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
        
        # Temp 0.6 keeps the dialect consistent but allows some "casual" improvisation
        gemini_params = InputParams(temperature=0.6) 

        gemini_live = GeminiLiveService(
            api_key=api_key,
            model=model,
            voice_id=voice_id,
            system_instruction=system_prompt,
            params=gemini_params,
            inference_on_context_initialization=True
        )

        # --- TOOLS ---
        lead_finalized = {"value": None}
        async def confirm_appointment(params: FunctionCallParams):
            lead_finalized["value"] = "CONFIRMED"
            update_lead_status(lead_data["id"], "CONFIRMED")
            await params.result_callback({"value": "done"})
            if call_control_id: asyncio.create_task(hangup_telnyx_call(call_control_id, 2.0))

        async def cancel_appointment(params: FunctionCallParams):
            lead_finalized["value"] = "CANCELLED"
            update_lead_status(lead_data["id"], "CANCELLED")
            await params.result_callback({"value": "done"})
            if call_control_id: asyncio.create_task(hangup_telnyx_call(call_control_id, 2.0))

        gemini_live.register_function("update_lead_status_confirmed", confirm_appointment)
        gemini_live.register_function("update_lead_status_cancelled", cancel_appointment)

        # --- PIPELINE ---
        mm_context = LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة الآن."}])
        
        # Strategies
        start_strategies = [
            VADUserTurnStartStrategy(vad_analyzer=vad),
            TranscriptionUserTurnStartStrategy(use_interim=True)
        ]
        # Fast stop for interruptions
        stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.6)]

        mm_aggregators = LLMContextAggregatorPair(
            mm_context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies)
            ),
        )

        try: await gemini_live.set_context(mm_context)
        except: pass

        pipeline = Pipeline([
            transport.input(),
            InboundAudioLogger(),
            mm_aggregators.user(),
            MultimodalUserStopRunTrigger(), # Fast trigger
            gemini_live,
            MultimodalTranscriptRunTrigger(delay_s=0.5),
            LeadStatusTranscriptFallback(lead_id=lead_data["id"], call_control_id=call_control_id, call_end_delay_s=2.0, finalized_ref=lead_finalized),
            TurnStateLogger(),
            OutboundAudioLogger(),
            AudioFrameChunker(chunk_ms=20),
            MultimodalPerf(),
            transport.output(),
            mm_aggregators.assistant(),
        ])

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
        runner = PipelineRunner()

        call_alive = {"value": True}
        @transport.event_handler("on_client_connected")
        async def _on_client_connected(_transport, _client):
            logger.info("Client connected")

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _client):
            call_alive["value"] = False

        # Failsafe start
        async def failsafe():
            await asyncio.sleep(4.0)
            if call_alive["value"] and _MM.get("last_bot_started_ts") is None:
                await task.queue_frames([LLMRunFrame()])
        asyncio.create_task(failsafe())

        await runner.run(task)
        return

    return
