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
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, TranscriptionFrame
from pipecat.turns.user_turn_strategies import UserTurnStrategies, TranscriptionUserTurnStartStrategy, TranscriptionUserTurnStopStrategy
from services.supabase_service import update_lead_status
import json
import time

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-21-ahmed-saudi-v4-ultra-fix"
_VAD_MODEL = {"value": None}

class MultimodalPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "TranscriptionFrame" and getattr(frame, "role", None) == "user":
            _MM["last_user_transcription_ts"] = time.monotonic()
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
        if self._queue_frames and self._last_user_transcript_ts == ts:
            _MM["last_llm_run_ts"] = time.monotonic()
            await self._queue_frames([LLMRunFrame()])
    def cancel_pending(self):
        if self._pending:
            self._pending.cancel()
            self._pending = None
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            if not getattr(frame, "is_final", True): return
            now = time.monotonic()
            self._last_user_transcript_ts = now
            self.cancel_pending()
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
    async def _schedule(self, ts: float):
        try: await asyncio.sleep(self._delay_s)
        except asyncio.CancelledError: return
        if self._queue_frames and self._last_stop_ts == ts:
            now = time.monotonic()
            if _MM["last_llm_run_ts"] and (now - _MM["last_llm_run_ts"]) < self._min_interval_s: return
            _MM["last_llm_run_ts"] = now
            await self._queue_frames([LLMRunFrame()])
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if type(frame).__name__ == "UserStoppedSpeakingFrame":
            now = time.monotonic()
            self._last_stop_ts = now
            if self._pending: self._pending.cancel()
            self._pending = asyncio.create_task(self._schedule(now))
        await self.push_frame(frame, direction)

class TurnStateLogger(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        frame_name = type(frame).__name__
        if frame_name == "UserStartedSpeakingFrame":
            logger.info("Turn: User speaking")
        elif frame_name in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            logger.info("Turn: Bot speaking")
        await self.push_frame(frame, direction)

class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms: int = 20):
        super().__init__()
        self._chunk_ms = chunk_ms
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if self._chunk_ms > 0 and isinstance(frame, AudioRawFrame):
            audio = frame.audio
            sample_rate = getattr(frame, "sample_rate", 16000)
            channels = getattr(frame, "num_channels", 1)
            chunk_bytes = int(sample_rate * self._chunk_ms / 1000) * channels * 2
            
            if len(audio) > chunk_bytes:
                for i in range(0, len(audio), chunk_bytes):
                    chunk = audio[i : i + chunk_bytes]
                    if chunk:
                        # Copy all metadata (transport_destination, etc) to new frame
                        new_frame = AudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=channels)
                        for attr, value in vars(frame).items():
                            if not hasattr(new_frame, attr): setattr(new_frame, attr, value)
                        await self.push_frame(new_frame, direction)
                return
        await self.push_frame(frame, direction)

async def hangup_telnyx_call(call_control_id: str, delay_s: float) -> None:
    telnyx_key = os.getenv("TELNYX_API_KEY")
    await asyncio.sleep(delay_s)
    url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id, safe='')}/actions/hangup"
    headers = {"Authorization": f"Bearer {telnyx_key}", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession() as session:
            await session.post(url, headers=headers, json={"reason": "normal_clearing"})
    except: pass

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info("Ahmed (Saudi Bot) Starting...")
    PIPELINE_SAMPLE_RATE = 16000

    # Ahmed Persona - Short & Sharp
    system_prompt = """أنت "أحمد"، خبير مبيعات من شركة "موصول". لهجتك سعودية نجدية بيضاء، طبيعية جداً.
    
**قواعدك:**
- ردودك قصيرة جداً (ما تزيد عن جملة وحدة).
- لا تسولف كثير، هدفك تحجز "ديمو" للمنصة وبس.
- استخدم كلمات: "سم"، "أبشر"، "يا طويل العمر"، "هلا والله".
- إذا العميل سألك "مين معي؟" قل: "معك أحمد من موصول، عساك طيب؟"
- إذا سألك عن السعر قل: "الديمو مجاني طال عمرك، تبي أحجز لك موعد تجربه؟"
- إذا قاطعك العميل، اسكت فوراً."""

    # Handshake
    stream_id = "telnyx_stream"
    try:
        msg = json.loads(await websocket_client.receive_text())
        stream_id = msg.get("stream_id") or msg.get("data", {}).get("stream_id") or "telnyx_stream"
    except: pass

    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.5, start_secs=0.1, stop_secs=0.4, confidence=0.7))
    
    serializer = TelnyxFrameSerializer(
        stream_id=stream_id, call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding="PCMU", inbound_encoding="PCMU",
        params=TelnyxFrameSerializer.InputParams(sample_rate=PIPELINE_SAMPLE_RATE)
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer, add_wav_header=False, vad_analyzer=vad,
            audio_in_enabled=True, audio_out_enabled=True,
            audio_in_sample_rate=PIPELINE_SAMPLE_RATE, audio_out_sample_rate=PIPELINE_SAMPLE_RATE,
        )
    )

    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams, GeminiModalities
    gemini_live = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        voice_id="Aoede", model="models/gemini-2.0-flash-exp",
        system_instruction=system_prompt,
        params=InputParams(temperature=0.7, modalities=GeminiModalities.AUDIO),
        inference_on_context_initialization=True,
    )

    # Functions
    async def confirm_demo(params: FunctionCallParams):
        update_lead_status(lead_data.get("id"), "CONFIRMED")
        await params.result_callback({"status": "confirmed"})
        if call_control_id: asyncio.create_task(hangup_telnyx_call(call_control_id, 2.0))

    gemini_live.register_function("update_lead_status_confirmed", confirm_demo)

    mm_context = LLMContext(messages=[{"role": "user", "content": "ابدأ المكالمة فوراً وسلم بلهجة سعودية."}])
    mm_aggregators = LLMContextAggregatorPair(
        mm_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy(use_interim=True)],
                stop=[TranscriptionUserTurnStopStrategy(timeout=0.6)]
            )
        )
    )

    await gemini_live.set_context(mm_context)

    pipeline = Pipeline([
        transport.input(),
        mm_aggregators.user(),
        MultimodalUserStopRunTrigger(),
        gemini_live,
        MultimodalTranscriptRunTrigger(delay_s=0.7),
        TurnStateLogger(),
        AudioFrameChunker(chunk_ms=20),
        MultimodalPerf(),
        transport.output(),
        mm_aggregators.assistant(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()

    @transport.event_handler("on_client_connected")
    async def _on_connect(_t, _c): logger.info("Ahmed Online.")

    async def initial_failsafe():
        await asyncio.sleep(1.5)
        if not _MM["last_bot_started_ts"]: await task.queue_frames([LLMRunFrame()])

    asyncio.create_task(initial_failsafe())
    await runner.run(task)
