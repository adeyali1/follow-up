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
from pipecat.frames.frames import LLMRunFrame, LLMMessagesAppendFrame
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

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-ammani-dental-fixed"
_VAD_MODEL = {"value": None}

# ────────────────────────────────────────────────
# ALL FRAME PROCESSOR CLASSES – keep these as they are
# ────────────────────────────────────────────────

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
        if any(x in t for x in ["الغ", "كنسل", "cancel", "مش بد", "مش بدي", "لا بدي", "إلغاء", "الغاء"]):
            return False
        if "مش" in t and any(x in t for x in ["تمام", "ماشي", "موافق"]):
            return False
        return any(
            x in t
            for x in [
                "تمام",
                "ماشي",
                "أكيد",
                "اكيد",
                "موافق",
                "اوكي",
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
        return any(x in t for x in ["الغ", "إلغاء", "الغاء", "كنسل", "cancel", "مش بد", "مش بدي"])

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

# ────────────────────────────────────────────────
# MISSING FUNCTION – THIS WAS THE CAUSE OF NameError
# ────────────────────────────────────────────────
def normalize_customer_name_for_ar(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw
    lowered = raw.lower()
    if lowered == "oday":
        return "عدي"
    # Add more name mappings here if needed
    return raw

def normalize_gemini_live_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return "models/gemini-2.5-flash-native-audio-preview-12-2025"
    if model.startswith("models/"):
        return model
    if "/" in model:
        return model
    return f"models/{model}"

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data.get('id', 'mock-lead-id')}")
    logger.info(f"Bot build: {BOT_BUILD_ID}")
    _MM["last_user_transcription_ts"] = None
    _MM["last_bot_started_ts"] = None
    _MM["last_llm_run_ts"] = None

    use_multimodal_live = os.getenv("USE_MULTIMODAL_LIVE", "true").lower() == "true"
    pipeline_sample_rate = 16000
    try:
        pipeline_sample_rate = int(os.getenv("PIPELINE_SAMPLE_RATE") or os.getenv("GEMINI_LIVE_SAMPLE_RATE") or 16000)
    except Exception:
        pipeline_sample_rate = 16000

    # Mock/fallback lead data
    if 'patient_name' not in lead_data:
        lead_data['patient_name'] = lead_data.get('customer_name', 'المريض')
    if 'treatment' not in lead_data:
        lead_data['treatment'] = lead_data.get('order_items', 'تنظيف أسنان')
    if 'appointment_time' not in lead_data:
        lead_data['appointment_time'] = lead_data.get('delivery_time', 'الساعة 11:00')
    if 'id' not in lead_data:
        lead_data['id'] = 'mock-lead-id'

    logger.info(f"Using lead_data: {lead_data}")

    # Telnyx handshake
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"
    stream_sample_rate = 8000
    try:
        logger.info("Waiting for Telnyx 'start' event with stream_id...")
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            logger.info(f"Received Telnyx message: {msg_text}")
            msg = json.loads(msg_text)
            if "media_format" in msg:
                encoding = msg["media_format"].get("encoding", "").upper()
                stream_sample_rate = int(msg["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                logger.info(f"Telnyx Media Format (direct): {msg['media_format']}")
                if encoding == "G729":
                    logger.error("CRITICAL: Telnyx G.729 detected – disable in portal")
                elif encoding == "PCMA":
                    inbound_encoding = "PCMA"
                elif encoding == "PCMU":
                    inbound_encoding = "PCMU"
                elif encoding == "L16":
                    inbound_encoding = "L16"
            elif "start" in msg and "media_format" in msg["start"]:
                encoding = msg["start"]["media_format"].get("encoding", "").upper()
                stream_sample_rate = int(msg["start"]["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                logger.info(f"Telnyx Media Format (nested): {msg['start']['media_format']}")
                if encoding == "G729":
                    logger.error("CRITICAL: Telnyx G.729 detected")
                elif encoding == "PCMA":
                    inbound_encoding = "PCMA"
                elif encoding == "PCMU":
                    inbound_encoding = "PCMU"
                elif encoding == "L16":
                    inbound_encoding = "L16"
            if "stream_id" in msg:
                stream_id = msg["stream_id"]
                logger.info(f"Captured stream_id: {stream_id}")
                break
            elif "data" in msg and "stream_id" in msg["data"]:
                stream_id = msg["data"]["stream_id"]
                logger.info(f"Captured stream_id (data): {stream_id}")
                break
            elif msg.get("event") == "start":
                if "stream_id" in msg:
                    stream_id = msg["stream_id"]
                    logger.info(f"Captured stream_id (start event): {stream_id}")
                    break
        if stream_id == "telnyx_stream_placeholder":
            logger.warning("No stream_id found, using placeholder")
    except Exception as e:
        logger.error(f"Failed to capture stream_id: {e}")

    patient_name = normalize_customer_name_for_ar(lead_data.get("patient_name", "المريض"))
    greeting_text = f"السلام عليكم، معك سارة من عيادة أسنان الابتسامة. معي يا {patient_name}؟"

    vad_stop_secs = 0.2
    try:
        vad_stop_secs = float(os.getenv("VAD_STOP_SECS") or 0.2)
    except Exception:
        vad_stop_secs = 0.2

    vad_min_volume = 0.55
    try:
        vad_min_volume = float(os.getenv("VAD_MIN_VOLUME") or 0.55)
    except Exception:
        pass

    vad_confidence = 0.65
    try:
        vad_confidence = float(os.getenv("VAD_CONFIDENCE") or 0.65)
    except Exception:
        pass

    vad = None
    cached_vad = _VAD_MODEL.get("value")
    if cached_vad is None:
        vad = SileroVADAnalyzer(
            params=VADParams(min_volume=vad_min_volume, start_secs=0.2, stop_secs=vad_stop_secs, confidence=vad_confidence)
        )
        _VAD_MODEL["value"] = vad
    else:
        vad = cached_vad

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

    treatment = lead_data.get("treatment", "تنظيف أسنان")
    appointment_time = lead_data.get("appointment_time", "الساعة 11:00")

    system_prompt = f"""
# ROLE: سارة – أفضل منسقة علاج أسنان أردنية أصيلة من عمان
أنتِ سارة، بنت عمانية أصيلة من قلب المدينة، شغالة في عيادة أسنان الابتسامة. صوتك وكلامك 100% عماني مدني أصيل — دافئ، سريع، عفوي، زي ما بنحكي بالتلفون مع الجيران في عبدون أو جبل عمان. ما في ولا ذرة روبوت: غيّري نبرتك، ضحكي شوي 'ههه'، تنهدي براحة، استخدمي كلماتنا اليومية 'يعني'، 'شوف'، 'أه'، 'ماشي'، 'يا غالي/غاليتي'، 'يا ريت'.

# قواعد النطق – مهم جدًا
- الأسماء: نطقها عماني 100%. أمثلة:
  - 'Oday' → 'عُدَيْ' (عُ-دَي، خفيف على العين، شدة على الدال)
  - 'Muhammad' → 'مْحَمَّد' (مْحَمَّد، هاء خفيفة من الحلق)
  - 'Aisha' → 'عايشة' (عاي-شة، مدّي الآي)
  - 'Ahmed' → 'أحْمَد' (أحْ-مَد، هاء قوية)
- عام: الراء خفيفة، القاف 'گ' (گلب)، وقفات طبيعية.

# اللهجة – عمانية صافية – ممنوع الفصحى
- دائمًا: 'شو'، 'ليش'، 'بدي'، 'هسا'، 'تمام/ماشي'، 'أكيد'، 'يا ريت'
- ممنوع: 'ماذا'، 'هل'، 'سوف'، 'نعم' — قولي 'أه' أو 'إيه'
- جمل قصيرة وسريعة (6-10 كلمات غالبًا)

# التحية الأولى – حرفيًا
"{greeting_text}"

# سير العمل
1. بعد التحية: "أهلاً يا {patient_name}، شكراً إنك رديت. بدي أتأكد موعدك لـ {treatment} الساعة {appointment_time} لسا مناسبك؟"
2. لو أكد: (confirm بصمت) → "حلو كتير يا {patient_name}! خلص اعتمدناه. نشوفك على خير إن شاء الله."
3. لو ألغى: (cancel بصمت) → "ماشي يا غالي، ولا يهمك. لو بدك نرجع خبرني."
4. أسئلة: "العلاج سهل وما بيوجع، الدكتور شاطر."

# ممنوع تذكري أي دالة أو كود — كلام طبيعي فقط.
"""

    if use_multimodal_live:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("Missing API key")
            return

        gemini_in_sample_rate = pipeline_sample_rate
        model_env = os.getenv("GEMINI_LIVE_MODEL", "").strip()
        model = normalize_gemini_live_model_name(model_env)
        voice_id = os.getenv("GEMINI_LIVE_VOICE", "Charon").strip()

        logger.info(f"PRE-CONNECT CONFIG → model: {model} | voice: {voice_id} | sr: {gemini_in_sample_rate} | LANGUAGE: NOT SET")

        from pipecat.services.google.gemini_live.llm import (
            GeminiLiveLLMService,
            InputParams as GeminiLiveInputParams,
        )

        gemini_params = GeminiLiveInputParams(temperature=0.3)

        # NO LANGUAGE SET – THIS IS THE FIX

        try:
            gemini_params.sample_rate = gemini_in_sample_rate
        except:
            pass

        try:
            from pipecat.services.google.gemini_live.llm import GeminiModalities
            gemini_params.modalities = GeminiModalities.AUDIO
        except:
            pass

        logger.info(f"GeminiLive mode enabled (model={model}, voice={voice_id}, modalities=AUDIO, in_sr={gemini_in_sample_rate}, out_sr={stream_sample_rate})")

        try:
            gemini_kwargs = {
                "api_key": api_key,
                "voice_id": voice_id,
                "system_instruction": system_prompt,
                "params": gemini_params,
                "inference_on_context_initialization": True,
            }
            if model:
                gemini_kwargs["model"] = model
            gemini_live = GeminiLiveService(**gemini_kwargs)
        except Exception as e:
            logger.error(f"GeminiLive init failed: {e}")
            return

        # ────────────────────────────────────────────────
        # The rest of your original code goes here (call_end_delay_s, functions, pipeline, task, runner, etc.)
        # Copy from your previous file starting from:
        # call_end_delay_s = 2.5
        # to the end (await runner.run(task))
        # ────────────────────────────────────────────────

        # ... (paste the remaining part of your run_bot function here)

        return

    logger.error("USE_MULTIMODAL_LIVE must be true.")
    return
