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

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-saudi-ahmed-mawsool-v2-optimized"
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
    def __init__(self, *, chunk_ms: int = 0):
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
                "صح",
                "عاد",
            ]
        )

    @staticmethod
    def _is_cancel(text: str) -> bool:
        t = LeadStatusTranscriptFallback._normalize(text)
        if not t:
            return False
        return any(x in t for x in ["الغ", "إلغاء", "الغاء", "كنسل", "cancel", "مش بد", "مش بدي", "لا شكر"])

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

def normalize_gemini_live_model_name(model: str) -> str:
    model = (model or "").strip()
    if not model:
        return "models/gemini-2.5-flash-preview-native-audio-dialog"
    if model.startswith("models/"):
        return model
    if "/" in model:
        return model
    return f"models/{model}"

def normalize_customer_name_for_ar(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return ""
    return raw

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

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data.get('id', 'mock-lead-id')}")

    _MM["last_user_transcription_ts"] = None
    _MM["last_bot_started_ts"] = None
    _MM["last_llm_run_ts"] = None

    use_multimodal_live = os.getenv("USE_MULTIMODAL_LIVE", "true").lower() == "true"
    pipeline_sample_rate = 16000

    try:
        pipeline_sample_rate = int(os.getenv("PIPELINE_SAMPLE_RATE") or os.getenv("GEMINI_LIVE_SAMPLE_RATE") or 16000)
    except Exception:
        pipeline_sample_rate = 16000

    if 'patient_name' not in lead_data:
        lead_data['patient_name'] = lead_data.get('customer_name', 'يا غالي')
    if 'treatment' not in lead_data:
        lead_data['treatment'] = lead_data.get('order_items', 'ديمو منصة موصول')
    if 'appointment_time' not in lead_data:
        lead_data['appointment_time'] = lead_data.get('delivery_time', 'خلال هالأسبوع')
    if 'id' not in lead_data:
        lead_data['id'] = 'mock-lead-id'

    logger.info(f"Using lead_data: {lead_data}")

    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"
    stream_sample_rate = 8000

    try:
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            logger.info(f"Received Telnyx message: {msg_text}")
            msg = json.loads(msg_text)

            if "media_format" in msg:
                encoding = msg["media_format"].get("encoding", "").upper()
                stream_sample_rate = int(msg["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                if encoding == "PCMA": inbound_encoding = "PCMA"
                elif encoding == "PCMU": inbound_encoding = "PCMU"
            elif "start" in msg and "media_format" in msg["start"]:
                encoding = msg["start"]["media_format"].get("encoding", "").upper()
                stream_sample_rate = int(msg["start"]["media_format"].get("sample_rate", stream_sample_rate) or stream_sample_rate)
                if encoding == "PCMA": inbound_encoding = "PCMA"
                elif encoding == "PCMU": inbound_encoding = "PCMU"

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
        logger.error(f"Failed to capture stream_id: {e}")

    patient_name = normalize_customer_name_for_ar(lead_data.get("patient_name", ""))

    greeting_text = "هلا والله! معك خالد من شركة موصول، عساك طيب؟"

    # Optimized VAD parameters for low latency and accurate detection
    vad_stop_secs = 0.15  # Reduced from 0.2 for faster response
    vad_start_secs = 0.15  # Faster start detection
    vad_min_volume = 0.5   # Optimized from 0.6 for better sensitivity
    vad_confidence = 0.6   # Optimized from 0.7 for better balance

    try:
        vad_stop_secs = float(os.getenv("VAD_STOP_SECS") or 0.15)
        vad_start_secs = float(os.getenv("VAD_START_SECS") or 0.15)
        vad_min_volume = float(os.getenv("VAD_MIN_VOLUME") or 0.5)
        vad_confidence = float(os.getenv("VAD_CONFIDENCE") or 0.6)
    except Exception:
        pass

    vad = None
    cached_vad = _VAD_MODEL.get("value")
    if cached_vad is None:
        vad = SileroVADAnalyzer(
            params=VADParams(
                min_volume=vad_min_volume,
                start_secs=vad_start_secs,
                stop_secs=vad_stop_secs,
                confidence=vad_confidence
            )
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

    # OPTIMIZED SAUDI SYSTEM PROMPT
    system_prompt = """أنت "خالد"، مندوب مبيعات متخصص من "موصول" (Mawsool - mawsool.tech). لهجتك سعودية نجدية طبيعية ومباشرة.

**القواعد الذهبية:**
1. **اختصر واوضح**: جملة أو جملتين كحد أقصى في كل رد
2. **استمع أكثر**: إذا قاطعك العميل، اسكت فوراً واسمعه
3. **كن طبيعي**: استخدم كلمات مثل "هلا"، "تمام"، "أبشر"، "عاد"، "سم"
4. **لا تكرر**: إذا سألت سؤال، ما تعيده ثاني
5.كأنك تكلم عميل على الجوال.

**الهدف الوحيد:**
حجز موعد ديمو لمنصة موصول - قاعدة بيانات ضخمة لبيانات الشركات (B2B Leads).

**طريقة الحوار:**
1. **الافتتاح** (مرة واحدة فقط): "هلا والله! معك خالد من موصول، عساك طيب؟"
2. **إذا قال مشغول**: "ما أطول عليك، دقيقة وحدة. تسمح لي؟"
3. **الشرح المختصر**: "طال عمرك، موصول توفر لك أرقام وإيميلات مدراء الشركات في السعودية والخليج، دقة عالية وتحديث يومي. تغنيك عن 10 أدوات وتوفر عليك آلاف الريالات."
4. **إغلاق الموعد**: "وش رايك نحجز لك ديمو سريع تشوف فيه النظام؟ متى يناسبك؟"

**الردود** (examples):
- تمام عاد، موصول تعطيك قاعدة بيانات كاملة للشركات، ايميلات وأرقام وكل شي. متى يناسبك نشوف ديمو؟
- صح، النظام يوفر عليك وقت ومجهود. نحجز ديمو؟
- أبشر، هذا برسل لك رابط الديمو

**إذا سأل "مين معي؟":**
- "أنا خالد من موصول! ممكن دقيقة؟"

**ممنوعات:**
- لا تتكلم فصحى أبداً
- لا تسرد نقاط طويلة
- لا تكرر نفسك
- لا تقاطع العميل
- لا تطول في الحديث

**استخدام الأدوات:**
- بعد موافقة العميل على الديمو: استدعي `update_lead_status_confirmed`
- إذا رفض العميل: استدعي `update_lead_status_cancelled`

**تذكر:** خليك مختصر، طبيعي، وواضح.
"""

    if use_multimodal_live:
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GOOGLE_API_KEY/GEMINI_API_KEY is missing.")
            return

        gemini_in_sample_rate = pipeline_sample_rate
        model_env = (os.getenv("GEMINI_LIVE_MODEL") or "").strip()
        model = normalize_gemini_live_model_name(model_env)
        voice_id = (os.getenv("GEMINI_LIVE_VOICE") or "Aoede").strip()

        from pipecat.services.google.gemini_live.llm import (
            GeminiLiveLLMService as GeminiLiveService,
            InputParams as GeminiLiveInputParams,
        )


        http_api_version = (os.getenv("GEMINI_LIVE_HTTP_API_VERSION") or "v1beta").strip()
        http_options = None

        try:
            from google.genai.types import HttpOptions
            http_options = HttpOptions(api_version=http_api_version)
        except Exception:
            pass

        try:
            from pipecat.services.google.gemini_live.llm import GeminiModalities
        except Exception:
            GeminiModalities = None

        # Optimized temperature for more consistent, focused responses
        GeminiLiveInputParams(
    temperature=0.7,      # يعطي حياة
    top_p=0.9             # يخفف الجمود
)


        try:
            gemini_params.sample_rate = gemini_in_sample_rate
            if GeminiModalities is not None:
                gemini_params.modalities = GeminiModalities.AUDIO
        except Exception:
            pass

        logger.info(f"GeminiLive Optimized: model={model}, voice={voice_id}, temp=0.5")

        try:
            gemini_kwargs = {
                "api_key": api_key,
                "voice_id": voice_id,
                "system_instruction": system_prompt,
                "params": gemini_params,
                "inference_on_context_initialization": True,
            }
            if http_options is not None:
                gemini_kwargs["http_options"] = http_options
            if model:
                gemini_kwargs["model"] = model

            gemini_live = GeminiLiveService(**gemini_kwargs)
        except Exception as e:
            logger.error(f"GeminiLive init failed: {e}")
            return

        call_end_delay_s = 2.0  # Slightly reduced for faster call termination
        lead_finalized = {"value": None}

        async def confirm_appointment(params: FunctionCallParams):
            logger.info(f"Tool: Confirming Demo/Appointment for {lead_data['id']}")
            lead_finalized["value"] = "CONFIRMED"
            update_lead_status(lead_data["id"], "CONFIRMED")
            await params.result_callback({"value": "تم تأكيد الموعد بنجاح"})
            if call_control_id:
                asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

        async def cancel_appointment(params: FunctionCallParams):
            logger.info(f"Tool: Cancelling for {lead_data['id']}")
            lead_finalized["value"] = "CANCELLED"
            update_lead_status(lead_data["id"], "CANCELLED")
            await params.result_callback({"value": "تم إلغاء الموعد"})
            if call_control_id:
                asyncio.create_task(hangup_telnyx_call(call_control_id, call_end_delay_s))

        gemini_live.register_function("update_lead_status_confirmed", confirm_appointment)
        gemini_live.register_function("update_lead_status_cancelled", cancel_appointment)

        mm_perf = MultimodalPerf()
        mm_context = LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة وقدم نفسك بشكل مختصر ومباشر."}])

        user_mute_strategies = []
        mute_first_bot = (os.getenv("MULTIMODAL_MUTE_UNTIL_FIRST_BOT") or "true").lower() == "true"
        if mute_first_bot:
            try:
                from pipecat.turns.mute import MuteUntilFirstBotCompleteUserMuteStrategy
                user_mute_strategies.append(MuteUntilFirstBotCompleteUserMuteStrategy())
            except Exception:
                pass

        # Optimized for faster response times
        stop_timeout_s = 0.6  # Reduced from 0.8 for quicker turn detection
        start_strategies = [TranscriptionUserTurnStartStrategy(use_interim=True)]
        stop_strategies = [TranscriptionUserTurnStopStrategy(timeout=stop_timeout_s)]

        mm_aggregators = LLMContextAggregatorPair(
            mm_context,
            user_params=LLMUserAggregatorParams(
                user_turn_strategies=UserTurnStrategies(start=start_strategies, stop=stop_strategies),
                user_mute_strategies=user_mute_strategies,
            ),
        )

        try:
            maybe_coro = gemini_live.set_context(mm_context)
            if asyncio.iscoroutine(maybe_coro):
                await maybe_coro
        except Exception:
            pass

        # Optimized trigger delays for faster responses
        transcript_run_delay_s = 0.5  # Reduced from 0.7
        transcript_trigger = MultimodalTranscriptRunTrigger(delay_s=transcript_run_delay_s)

        transcript_fallback = LeadStatusTranscriptFallback(
            lead_id=lead_data["id"],
            call_control_id=call_control_id,
            call_end_delay_s=call_end_delay_s,
            finalized_ref=lead_finalized,
        )

        user_stop_trigger = MultimodalUserStopRunTrigger(
            delay_s=float(os.getenv("MULTIMODAL_RUN_ON_USER_STOP_DELAY_S") or 0.03),
            min_interval_s=float(os.getenv("MULTIMODAL_RUN_MIN_INTERVAL_S") or 0.2),
        )

        pipeline = Pipeline(
            [
                transport.input(),
                InboundAudioLogger(),
                mm_aggregators.user(),
                user_stop_trigger,
                gemini_live,
                transcript_trigger,
                transcript_fallback,
                TurnStateLogger(),
                OutboundAudioLogger(),
                AudioFrameChunker(chunk_ms=int(os.getenv("AUDIO_OUT_CHUNK_MS") or 20)),
                mm_perf,
                transport.output(),
                mm_aggregators.assistant(),
            ]
        )

        task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

        transcript_trigger.set_queue_frames(task.queue_frames)
        user_stop_trigger.set_queue_frames(task.queue_frames)

        runner = PipelineRunner()

        did_trigger_initial_run = {"value": False}
        call_alive = {"value": True}

        @transport.event_handler("on_client_connected")
        async def _on_client_connected(_transport, _client):
            if not call_alive["value"] or did_trigger_initial_run["value"]:
                return
            did_trigger_initial_run["value"] = True
            logger.info("Multimodal: client connected - starting Saudi Sales Bot")

        @transport.event_handler("on_client_disconnected")
        async def _on_client_disconnected(_transport, _client):
            call_alive["value"] = False
            transcript_trigger.cancel_pending()
            user_stop_trigger.cancel_pending()

        async def multimodal_first_turn_failsafe():
            await asyncio.sleep(6.0)  # Reduced from 8.0 for faster initial response
            if call_alive["value"] and _MM.get("last_bot_started_ts") is None:
                logger.warning("Failsafe: triggering initial LLM run")
                await task.queue_frames([LLMRunFrame()])

        asyncio.create_task(multimodal_first_turn_failsafe())

        await runner.run(task)
        return

    logger.error("USE_MULTIMODAL_LIVE must be true")
    return






