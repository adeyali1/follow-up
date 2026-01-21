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
from pipecat.turns.user_turn_strategies import (
    UserTurnStrategies,
    TranscriptionUserTurnStartStrategy,
    TranscriptionUserTurnStopStrategy,
)
from services.supabase_service import update_lead_status
import json
import time

_MM = {"last_user_transcription_ts": None, "last_bot_started_ts": None, "last_llm_run_ts": None}
BOT_BUILD_ID = "2026-01-20-dental-coordinator"
_VAD_MODEL = {"value": None}


# -------------------- PERF & TURN HELPERS --------------------

class MultimodalPerf(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and getattr(frame, "role", None) == "user":
            _MM["last_user_transcription_ts"] = time.monotonic()
        if type(frame).__name__ in {"BotStartedSpeakingFrame", "TTSStartedFrame"}:
            _MM["last_bot_started_ts"] = time.monotonic()
        await self.push_frame(frame, direction)


class MultimodalTranscriptRunTrigger(FrameProcessor):
    def __init__(self, *, delay_s: float):
        super().__init__()
        self.delay_s = delay_s
        self._queue_frames = None
        self._pending = None
        self._last_ts = None

    def set_queue_frames(self, q):
        self._queue_frames = q

    async def _run(self, ts):
        await asyncio.sleep(self.delay_s)
        if self._last_ts == ts:
            await self._queue_frames([LLMRunFrame()])

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.role == "user" and frame.is_final:
            ts = time.monotonic()
            self._last_ts = ts
            if self._pending:
                self._pending.cancel()
            self._pending = asyncio.create_task(self._run(ts))
        await self.push_frame(frame, direction)


# -------------------- AUDIO LOGGERS --------------------

class AudioFrameChunker(FrameProcessor):
    def __init__(self, *, chunk_ms=40):
        super().__init__()
        self.chunk_ms = chunk_ms

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


# -------------------- TELNYX HANGUP --------------------

async def hangup_telnyx_call(call_control_id: str, delay_s: float):
    await asyncio.sleep(delay_s)
    if not call_control_id:
        return
    key = os.getenv("TELNYX_API_KEY")
    if not key:
        return
    url = f"https://api.telnyx.com/v2/calls/{quote(call_control_id)}/actions/hangup"
    headers = {"Authorization": f"Bearer {key}"}
    async with aiohttp.ClientSession() as s:
        await s.post(url, headers=headers, json={"reason": "normal_clearing"})


# -------------------- MAIN BOT --------------------

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting Dental Coordinator Bot | Lead {lead_data['id']}")

    stream_id = "telnyx_stream"
    inbound_encoding = "PCMU"
    pipeline_sample_rate = 16000

    # ---------- Telnyx handshake ----------
    try:
        msg = json.loads(await websocket_client.receive_text())
        stream_id = msg.get("stream_id", stream_id)
    except Exception:
        pass

    customer_name = lead_data.get("customer_name") or "حضرتك"

    greeting_text = f"السلام عليكم، معك خالد من مركز الأسنان. معي يا {customer_name}؟"

    vad = _VAD_MODEL["value"]
    if not vad:
        vad = SileroVADAnalyzer(
            params=VADParams(min_volume=0.6, start_secs=0.2, stop_secs=0.25, confidence=0.7)
        )
        _VAD_MODEL["value"] = vad

    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        inbound_encoding=inbound_encoding,
        outbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(sample_rate=pipeline_sample_rate),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=pipeline_sample_rate,
            audio_out_sample_rate=pipeline_sample_rate,
        ),
    )

    # -------------------- SYSTEM PROMPT --------------------

    system_prompt = f"""
# ROLE
You are Khalid, a professional Dental Treatment Coordinator in Amman, Jordan.
This is a real medical follow-up call. Be calm, confident, and respectful.

# DIALECT (JORDANIAN – AMMANI)
- Speak clear Ammani Jordanian Arabic.
- No classical Arabic.
- No English unless patient uses it first.
- Use polite medical tone.

# COMMUNICATION RULES
- Short sentences.
- One question at a time.
- If interrupted: stop immediately.
- If silence: "معي يا غالي؟" مرة وحدة فقط.

# GREETING (MANDATORY)
Say EXACTLY once, then wait:
"{greeting_text}"

# WORKFLOW
1) Confirm appointment:
"{customer_name}، موعد علاج الأسنان يوم {lead_data['appointment_date']} الساعة {lead_data['appointment_time']} مع الدكتور. مناسب؟"

2) If confirmed:
- Call update_lead_status_confirmed
- Say:
"تمام، ثبتنا الموعد. بنستناك بالسلامة."

3) If wants reschedule:
"ولا يهمك، أي يوم أو ساعة بناسبك؟"

4) If cancelled:
- Call update_lead_status_cancelled
- Say:
"تمام، تم إلغاء الموعد. بنخدمك بأي وقت."

# IMPORTANT
- Never sound rushed
- Never pressure the patient
"""

    # -------------------- GEMINI LIVE --------------------

    from pipecat.services.google.gemini_live.llm import (
        GeminiLiveLLMService,
        InputParams,
    )

    gemini = GeminiLiveLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="models/gemini-2.0-flash-live-001",
        voice_id=os.getenv("GEMINI_LIVE_VOICE", "Charon"),
        system_instruction=system_prompt,
        params=InputParams(
            temperature=0.25,
            language=Language.AR,
            sample_rate=pipeline_sample_rate,
        ),
        inference_on_context_initialization=True,
    )

    # ---------- Demo functions ----------

    async def confirm(params: FunctionCallParams):
        update_lead_status(lead_data["id"], "CONFIRMED")
        await params.result_callback({"ok": True})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2.5))

    async def cancel(params: FunctionCallParams):
        update_lead_status(lead_data["id"], "CANCELLED")
        await params.result_callback({"ok": True})
        if call_control_id:
            asyncio.create_task(hangup_telnyx_call(call_control_id, 2.5))

    gemini.register_function("update_lead_status_confirmed", confirm)
    gemini.register_function("update_lead_status_cancelled", cancel)

    context = LLMContext(messages=[{"role": "user", "content": "ابدأ"}])

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy(use_interim=True)],
                stop=[TranscriptionUserTurnStopStrategy(timeout=0.8)],
            )
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            aggregators.user(),
            gemini,
            MultimodalTranscriptRunTrigger(delay_s=0.6),
            AudioFrameChunker(),
            MultimodalPerf(),
            transport.output(),
            aggregators.assistant(),
        ]
    )

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    await runner.run(task)
