import os
import sys
import ast
import asyncio
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMAssistantAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMContextFrame, EndFrame, LLMMessagesFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm_vertex import GoogleVertexLLMService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from services.supabase_service import update_lead_status

import json
from google.oauth2 import service_account



# -------------------------------------------------
# Google credentials helper
# -------------------------------------------------
def get_google_credentials():
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        logger.info(f"Using Google credentials file: {path}")
        return None

    json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not json_str:
        return None

    json_str = json_str.strip().strip("'").strip('"').replace('\\"', '"')
    json_str = json_str.replace("True", "true").replace("False", "false")

    try:
        info = json.loads(json_str)
    except json.JSONDecodeError:
        info = ast.literal_eval(json_str)

    if "private_key" in info:
        info["private_key"] = info["private_key"].replace("\\n", "\n")

    return info


# -------------------------------------------------
# Main bot
# -------------------------------------------------
async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data['id']}")

    # -------------------------------------------------
    # Telnyx handshake
    # -------------------------------------------------
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU"

    try:
        for _ in range(3):
            msg_text = await websocket_client.receive_text()
            msg = json.loads(msg_text)
            logger.info(f"Telnyx message: {msg_text}")

            media = msg.get("media_format") or msg.get("start", {}).get("media_format")
            if media:
                encoding = media.get("encoding", "").upper()
                if encoding == "PCMA":
                    inbound_encoding = "PCMA"
                elif encoding == "PCMU":
                    inbound_encoding = "PCMU"
                elif encoding == "G729":
                    logger.error("G729 NOT SUPPORTED – disable in Telnyx")

            stream_id = (
                msg.get("stream_id")
                or msg.get("data", {}).get("stream_id")
                or stream_id
            )
            if stream_id != "telnyx_stream_placeholder":
                break

    except Exception as e:
        logger.error(f"Handshake error: {e}")

    # -------------------------------------------------
    # Credentials
    # -------------------------------------------------
    creds_dict = get_google_credentials()
    creds_str = json.dumps(creds_dict) if creds_dict else None
    creds_obj = (
        service_account.Credentials.from_service_account_info(creds_dict)
        if creds_dict
        else None
    )

    # -------------------------------------------------
    # STT (Deepgram)
    # -------------------------------------------------
    dg_encoding = "mulaw" if inbound_encoding == "PCMU" else "alaw"
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="ar",
        sample_rate=8000,
        encoding=dg_encoding,
    )

    # -------------------------------------------------
    # LLM
    # -------------------------------------------------
    tools = [
        {
            "function_declarations": [
                {
                    "name": "update_lead_status_confirmed",
                    "description": "Order confirmed",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                    },
                },
                {
                    "name": "update_lead_status_cancelled",
                    "description": "Order cancelled",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                    },
                },
            ]
        }
    ]

    llm = GoogleVertexLLMService(
        project_id=os.getenv("GOOGLE_PROJECT_ID"),
        location="us-central1",
        model="gemini-2.5-flash-lite",
        tools=tools,
        credentials=creds_str,
        credentials_path=None if creds_str else os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    # -------------------------------------------------
    # TTS
    # -------------------------------------------------
    tts = GoogleTTSService(
        voice_id="ar-JO-Standard-A",
        sample_rate=8000,
        encoding="linear16",
        credentials=creds_obj,
        credentials_path=None if creds_obj else os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
    )

    # -------------------------------------------------
    # Tool handlers
    # -------------------------------------------------
    async def confirm_order(*args, **kwargs):
        update_lead_status(lead_data["id"], "CONFIRMED")
        return "تم تأكيد الطلب"

    async def cancel_order(*args, **kwargs):
        update_lead_status(lead_data["id"], "CANCELLED")
        return "تم إلغاء الطلب"

    llm.register_function("update_lead_status_confirmed", confirm_order)
    llm.register_function("update_lead_status_cancelled", cancel_order)

    # -------------------------------------------------
    # Transport (‼️ VAD COMPLETELY DISABLED)
    # -------------------------------------------------
    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        inbound_encoding=inbound_encoding,
        outbound_encoding="PCMA",
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            vad_enabled=False,          # 🔴 MUST BE FALSE
            vad_analyzer=None,          # 🔴 MUST BE NONE
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            session_timeout=300,
            add_wav_header=False,
        ),
    )

    # -------------------------------------------------
    # Prompt
    # -------------------------------------------------
    system_prompt = f"""
You are Kawkab AI, a delivery assistant speaking Jordanian Arabic.
Greet {lead_data['customer_name']} and confirm order:
{lead_data['order_items']} at {lead_data['delivery_time']}.
"""

    context = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "ابدأ المكالمة"},
        ]
    )

    # -------------------------------------------------
    # Aggregators (‼️ NO TURN STRATEGIES)
    # -------------------------------------------------
    user_agg = LLMUserAggregator(
        context,
        user_turn_start_strategy=None,
        user_turn_stop_strategy=None,
    )
    assistant_agg = LLMAssistantAggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,
            assistant_agg,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=False,
            cancel_on_interruption=False,
        ),
    )

    runner = PipelineRunner()

    logger.info("Starting bot (with safe delay)…")
    await asyncio.sleep(0.5)
    await task.queue_frames([LLMContextFrame(context)])
    await runner.run(task)
