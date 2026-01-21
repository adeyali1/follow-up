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

# --- GLOBAL TRACKING ---
_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-jordanian-native-ultimate"
_VAD_MODEL = {"value": None}

# --- PROCESSORS ---

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
        if self._queue_frames is None:
            return
        if self._last_user_transcript_ts != ts:
            return
        logger.info("Multimodal: user transcript idle → LLMRunFrame")
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
            is_final = True
            try:
                is_final = bool(getattr(frame, "is_final", True))
            except Exception:
                is_final = True
            if not is_final:
                await self.push_frame(frame, direction)
                return
            now = time.monotonic()
            try:
                text = (getattr(frame, "text", None) or "").strip()
                if text:
                    logger.debug(f"Multimodal: final user transcript received ({len(text)} chars)")
            except Exception:
                pass
            self._last_user_transcript_ts = now
            if self._pending is not None:
                self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class MultimodalUserStopRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float = 0.05, min_interval_s: float = 0.25):
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
            try:
                self._pending.cancel()
            except Exception:
                pass
            self._pending = None
    async def _schedule(self, ts: float):
        try:
            await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError:
            return
        if self._queue_frames is None:
            return
        if self._last_stop_ts != ts:
            return
        last_run = _MM.get("last_llm_run_ts")
        now = time.monotonic()
        if last_run is not None and (now - last_run) < self._min_interval_s:
            return
        _MM["last_llm_run_ts"] = now
        logger.info("Multimodal: user stop → LLMRunFrame")
        await self._queue_frames([LLMRunFrame()])
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            self._last_stop_ts = now
            if self._pending is not None:
                self._pending.cancel()
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
            if self._user_turn_started_ts is not None:
                dur = now - self._user_turn_started_ts
            self._user_turn_started_ts = None
            if dur is None:
                logger.info("Turn: user stopped speaking")
            else:
                logger.info(f"Turn: user stopped speaking (dur={dur:.2f}s)")
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
            try:
                logger.info(f"AudioOut: first audio frame sr={frame.sample_rate} bytes={len(frame.audio)}")
            except Exception:
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
            try:
                logger.info(f"AudioInDecoded: first audio frame sr={frame.sample_rate} bytes={len(frame.audio)}")
            except Exception:
                logger.info("AudioInDecoded: first audio frame")
        await self.push_frame(frame, direction)

class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 40):
        super().__init__()
        self._chunk_ms = int(chunk_ms)
        self._pace = (os.getenv("AUDIO_OUT_PACE") or "true").lower() == "true"
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._chunk_ms <= 0:
            await self.push_frame(frame, direction)
            return
        if isinstance(frame, AudioRawFrame):
            audio = frame.audio
            if isinstance(audio, (bytes, bytearray)) and len(audio) > 0:
                sample_rate = int(getattr(frame, "sample_rate", 0) or 0)
                channels = int(getattr(frame, "num_channels", 1) or 1)
                if sample_rate > 0 and channels > 0 and self._chunk_ms > 0:
                    bytes_per_sample = 2
                    bytes_per_frame = channels * bytes_per_sample
                    chunk_bytes = int(sample_rate * self._chunk_ms / 1000) * bytes_per_frame
                    chunk_bytes = max(chunk_bytes - (chunk_bytes % bytes_per_frame), bytes_per_frame)
                    if len(audio) > chunk_bytes:
                        frame_type = type(frame)
                        first = True
                        for i in range(0, len(audio), chunk_bytes):
                            chunk = audio[i : i + chunk_bytes]
                            if not chunk:
                                continue
                            try:
                                await self.push_frame(
                                    frame_type(audio=chunk, sample_rate=frame.sample_rate, num_channels=frame.num_channels),
                                    direction,
                                )
                                if self._pace and not first:
                                    await asyncio.sleep(self._chunk_ms / 1000)
                            except Exception:
                                await self.push_frame(frame, direction)
                                return
                            first = False
                        return
        await self.push_frame(frame, direction)

class LeadStatusTranscriptFallback(FrameProcessor):
    def __init__(self, *, lead_id: str, call_control_id: str | None, call_end_delay_s: float, finalized_ref: dict):
        super().__init__()
        self._lead_id = lead_id
        self._call_control_id = call_control_id
        self._call_end_delay_s = float(call_end_delay_s)
        self._finalized_ref = finalized_ref
    @staticmethod
    def _normalize(text: str) -> str:
        text = (text or "").strip().lower()
        for ch in ["\n", "\r", "\t", ".", ",", "!", "?", "؟", "،", "؛", "\"", "'"]:
            text = text.replace(ch, " ")
        while "  " in text:
            text = text.replace("  ", " ")
        return text
    @staticmethod
    def _is_confirm(text: str) -> bool:
        t = LeadStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        # Avoid false positives on negations
        if any(x in t for x in ["الغ", "كنسل", "cancel", "مش بد", "مش بدي", "لا بدي", "إلغاء", "الغاء", "لا موافق"]):
            return False
        if "مش" in t and any(x in t for x in ["تمام", "ماشي", "موافق"]):
            return False
        
        # Jordanian specific confirmations
        return any(
            x in t
            for x in [
                "تمام",
                "ماشي",
                "أكيد",
                "اكيد",
                "موافق",
                "اعتمد",
                "توكلنا على الله",
                "ميه ميه",
                "ان شاء الله",
                "يس",
                "okay",
                "ok",
                "yes",
                "بنعم",
                "اه",
                "أه",
            ]
        )
    @staticmethod
    def _is_cancel(text: str) -> bool:
        t = LeadStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        return any(x in t for x in ["الغ", "إلغاء", "الغاء", "كنسل", "cancel", "مش بد", "مش بدي", "ما بدي", "لا تحجز"])
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            try:
                if self._finalized_ref.get("value"):
                    await self.push_frame(frame, direction)
                    return
            except Exception:
                pass
            is_final = True
            try:
                is_final = bool(getattr(frame, "is_final", True))
            except Exception:
                is_final = True
            if is_final:
                text = ""
                try:
                    text = str(getattr(frame, "text", "") or "")
                except Exception:
                    text = ""
                if self._is_confirm(text):
                    self._finalized_ref["value"] = "CONFIRMED"
                    logger.info("Fallback: detected confirmation from transcript; updating lead status CONFIRMED")
                    update_lead_status(self._lead_id, "CONFIRMED")
                    if self._call_control_id:
                        asyncio.create_task(hangup_telnyx_call(self._call_control_id, self._call_end_delay_s))
                elif self._is_cancel(text):
                    self._finalized_ref["value"] = "CANCELLED"
                    logger.info("Fallback: detected cancellation from transcript; updating lead status CANCELLED")
                    update_lead_status(self._lead_id, "CANCELLED")
                    if self._call_control_id:
                        asyncio.create_task(hangup_telnyx_call(self._call_control_id, self._call_end_delay_s))
        await self.push_frame(frame, direction)

# --- HELPER FUNCTIONS ---

def normalize_gemini_live_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        # 2.0 Flash is faster and better at accents
        return "models/gemini-2.5-flash-native-audio-preview-12-2025"
    if model.startswith("models/"):
        return model
    if "/" in model:
        return model
    return f"models/{model}"

def normalize_customer_name_for_ar(name: str) -> str:
    """
    Crucial for Arabic TTS: Convert English names to Arabic script 
    so the AI pronounces them with an Arabic accent, not an English one.
    """
    raw = (name or "").strip()
    if not raw:
        return "المريض"
    
    lowered = raw.lower()
    
    # Common name mapping (Expand this list for production)
    mapping = {
        "oday": "عدي",
        "ahmad": "أحمد", "ahmed": "أحمد",
        "mohammad": "محمد", "mohammed": "محمد", "muhammad": "محمد",
        "sara": "سارة", "sarah": "سارة",
        "khaled": "خالد", "khalid": "خالد",
        "abdallah": "عبدالله", "abdullah": "عبدالله",
        "omar": "عمر",
        "yousef": "يوسف", "yusuf": "يوسف",
        "ibrahim": "ابراهيم",
        "layla": "ليلى", "laila": "ليلى",
        "noor": "نور", "nour": "نور",
        "hassan": "حسن",
        "ali": "علي",
        "fatima": "فاطمة",
        "zain": "زين",
        "rama": "راما",
        "salma": "سلمى",
        "mahmoud": "محمود",
        "sameer": "سمير", "samir": "سمير",
        "jordan": "الأردن"
    }
    
    if lowered in mapping:
        return mapping[lowered]
    
    # If the text is already Arabic, return it
    if re.search(r'[\u0600-\u06FF]', raw):
        return raw

    return raw  # Fallback: return English, the system prompt will try its best

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    if not call_control_id:
        return
    telnyx_key = os.getenv("TELNYX_API_KEY")
    if not telnyx_key:
        return
    if delay_s > 0:
        await asyncio.sleep(delay_s)
    encoded_call_control_id = quote(call_control_id, safe="")
    url = f"https://api.telnyx.com/v2/calls/{encoded_call_control_id}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json={"reason": "normal_clearing"}) as resp:
                if resp.status >= 300:
                    await resp.text()
    except Exception:
        return

# --- MAIN BOT LOGIC ---

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting Ultimate Jordanian Bot for lead: {lead_data.get('id', 'mock-lead-id')}")
    
    # Pipeline Setup
    _MM["last_user_transcription_ts"] = None
    _MM["last_bot_started_ts"] = None
    _MM["last_llm_run_ts"] = None
    
    pipeline_sample_rate = 16000 # Standard for Gemini Live
    
    # Lead Data Prep
    if 'patient_name' not in lead_data:
        lead_data['patient_name'] = lead_data.get('customer_name', 'يا غالي')
    if 'treatment' not in lead_data:
        lead_data['treatment'] = lead_data.get('order_items', 'تنظيف أسنان')
    if 'appointment_time' not in lead_data:
        lead_data['appointment_time'] = lead_data.get('delivery_time', 'الساعة 11:00')
    if 'id' not in lead_data:
        lead_data['id'] = 'mock-lead-id'

    # 1. Telnyx Handshake & Stream Identification
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU" # Default
    stream_sample_rate = 8000
    
    try:
        logger.info("Waiting for Telnyx 'start' event...")
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            msg = json.loads(msg_text)
            
            # Check for format
            if "media_format" in msg:
                fmt = msg["media_format"]
            elif "start" in msg and "media_format" in msg["start"]:
                fmt = msg["start"]["media_format"]
            else:
                fmt = {}
                
            if fmt:
                encoding = fmt.get("encoding", "").upper()
                stream_sample_rate = int(fmt.get("sample_rate", stream_sample_rate) or stream_sample_rate)
                if encoding == "PCMA":
                    inbound_encoding = "PCMA"
                elif encoding == "PCMU":
                    inbound_encoding = "PCMU"
                elif encoding == "L16":
                    inbound_encoding = "L16"
            
            # Check for stream_id
            if "stream_id" in msg:
                stream_id = msg["stream_id"]
                break
            elif "data" in msg and "stream_id" in msg["data"]:
                stream_id = msg["data"]["stream_id"]
                break
            elif msg.get("event") == "start":
                if "stream_id" in msg:
                    stream_id = msg["stream_id"]
                    break
                    
    except Exception as e:
        logger.error(f"Handshake error: {e}")

    # 2. Arabic Specific Logic
    patient_name_ar = normalize_customer_name_for_ar(lead_data.get("patient_name", "المريض"))
    treatment = lead_data.get("treatment", "موعد الأسنان")
    appointment_time = lead_data.get("appointment_time", "الموعد المحدد")
    
    # 3. VAD Configuration (Tuned for Arabic Flow)
    # Arabic speakers overlap; use shorter stop_secs but higher confidence
    vad_stop_secs = 0.4 
    vad = SileroVADAnalyzer(
        params=VADParams(
            min_volume=0.5, 
            start_secs=0.2, 
            stop_secs=vad_stop_secs, 
            confidence=0.7
        )
    )

    # 4. Transport Setup
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

    # 5. THE ULTIMATE JORDANIAN PROMPT
    # Mixed Arabic/English instructions for maximum adherence.
    system_prompt = f"""
# ROLE: SARA (Native Ammani Jordanian Receptionist)
You are Sara, the warm, professional, and purely Jordanian coordinator at "Smile Dental Clinic" (عيادة ابتسامة) in Amman.
Your dialect is 100% Native Ammani (Madani). You sound like a neighbor, not a robot.

# CUSTOMER INFO
- Name: {patient_name_ar} (Pronounce this exactly as written in Arabic).
- Treatment: {treatment}
- Time: {appointment_time}

# DIALECT RULES (STRICT)
1. **NO FUSHA (Modern Standard Arabic):** 
   - NEVER say: "ماذا", "لماذا", "حسناً", "نعم", "سوف".
   - INSTEAD say: "شو", "ليش", "تمام" / "ماشي", "أه", "رح".
2. **Jordanian Fillers (Use sparingly but naturally):**
   - "يعني" (Ya'ni)
   - "هسا" (Hassa - meaning 'now')
   - "شوف" / "شوفي" (Shoof/Shoofi - meaning 'look/listen')
   - "يا زلمة" (for intense emphasis only, rare for receptionist) -> Use "يا غالي" or "عزيزي/عزيزتي" instead.
   - "ان شاء الله" (Inshallah)
3. **Tone:** Warm, hospitable, confident. If interrupted, stop immediately.

# CONVERSATION STRUCTURE
1. **The Greeting:** "السلام عليكم، يسعد صباحك/مساك. معك سارة من عيادة ابتسامة. بحكي مع {patient_name_ar}؟"
2. **Confirmation:** "أهلاً يا {patient_name_ar}. برن عليك عشان نأكد موعدك لـ {treatment} يوم {appointment_time}. الموعد لسا مناسبك، صح؟"
3. **Handling:**
   - **Confirmed:** "ممتاز! خلص اعتمدنا. الدكتور بستناك، وان شاء الله بنشوفك على خير." -> (Function: Confirm)
   - **Cancelled:** "يا خسارة، بسيطة ولا يهمك. بتحب نأجله لوقت تاني؟" -> (If no, Function: Cancel)
   - **Questions:** Answer briefly. "التنظيف ما بوجع، الدكتور ايده خفيفة."

# CRITICAL INSTRUCTIONS
- If the user confirms (says "اه", "تمام", "اكيد"), say a nice closing and call `update_lead_status_confirmed`.
- If the user cancels (says "لا", "بدي الغي"), be polite and call `update_lead_status_cancelled`.
- Keep responses SHORT (1-2 sentences max). People in Amman talk fast.

GOAL: Make the patient feel they are talking to a real human in Abdoun or Khalda.
"""

    # 6. Gemini Live Setup
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("Missing GOOGLE_API_KEY")
        return

    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams, GeminiModalities
    
    # Model Selection - Use Flash 2.0 for speed/accents
    model_name = normalize_gemini_live_model_name(os.getenv("GEMINI_LIVE_MODEL"))
    
    # Voice Selection - 'Aoede' is usually warm and professional for female roles
    voice_id = os.getenv("GEMINI_LIVE_VOICE", "Aoede") 

    gemini_params = InputParams(
        temperature=0.4, # Slightly creative for natural flow
        language=Language.AR,
        sample_rate=pipeline_sample_rate,
        modalities=GeminiModalities.AUDIO
    )

    gemini_live = GeminiLiveLLMService(
        api_key=api_key,
        model=model_name,
        voice_id=voice_id,
        system_instruction=system_prompt,
        params=gemini_params,
        inference_on_context_initialization=True 
    )

    # 7. Function Calling
    call_end_delay_s = 2.0 # Give time to say "Goodbye"
    lead_finalized = {"value": None}

    async def confirm_appointment(params: FunctionCallParams):
        lead_finalized["value"] = "CONFIRMED"
        # update_lead_status(lead_data["id"], "CONFIRMED") # Uncomment for real DB
        await params.result_callback({"value": "Confirmed. Say polite goodbye in Ammani."})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

    async def cancel_appointment(params: FunctionCallParams):
        lead_finalized["value"] = "CANCELLED"
        # update_lead_status(lead_data["id"], "CANCELLED") # Uncomment for real DB
        await params.result_callback({"value": "Cancelled. Say polite goodbye."})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

    gemini_live.register_function("update_lead_status_confirmed", confirm_appointment)
    gemini_live.register_function("update_lead_status_cancelled", cancel_appointment)

    # 8. Pipeline Assembly
    mm_context = LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة"}])
    
    # Strategies
    start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
    stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=0.6)] # Faster turn taking for Arabic

    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies)
        ),
    )

    # Helper processors
    mm_perf = MultimodalPerf()
    transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=0.6)
    transcript_fallback = LeadStatusTranscriptFallback(
        lead_id=lead_data["id"],
        call_control_id=call_control_id,
        call_end_delay_s=call_end_delay_s,
        finalized_ref=lead_finalized,
    )
    
    # Audio logging & chunking
    audio_chunker = AudioFrameChunker(chunk_ms=20) # Smooth out audio delivery

    pipeline = Pipeline(
        [
            transport.input(),
            InboundAudioLogger(),
            mm_aggregators.user(),
            gemini_live,
            transcript_trigger,
            transcript_fallback,
            TurnStateLogger(),
            OutboundAudioLogger(),
            audio_chunker,
            mm_perf,
            transport.output(),
            mm_aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    transcript_trigger.set_queue_frames(task.queue_frames)

    runner = PipelineRunner()
    
    # Event Handlers
    call_alive = {"value": True}
    
    @transport.event_handler("on_client_disconnected")
    async def _on_client_disconnected(_transport, _client):
        call_alive["value"] = False

    # Failsafes
    async def silence_breaker():
        """If user doesn't speak or connection is silent, nudge."""
        await asyncio.sleep(6.0)
        if _MM["last_bot_started_ts"] is None and call_alive["value"]:
            logger.info("Silence detected at start, triggering hello...")
            await task.queue_frames([LLMRunFrame()])

    asyncio.create_task(silence_breaker())

    await runner.run(task)

