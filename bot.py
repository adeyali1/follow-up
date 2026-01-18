import os
import sys
import json
from loguru import logger
from google.oauth2 import service_account

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator, LLMAssistantAggregator
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.frames.frames import LLMContextFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm_vertex import GoogleVertexLLMService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.telnyx import TelnyxFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from services.supabase_service import update_lead_status

# Helper function to get Google Credentials safely
def get_google_credentials():
    json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if not json_str:
        logger.error("GOOGLE_APPLICATION_CREDENTIALS_JSON not found in environment")
        return None

    try:
        # 1. Basic cleaning: remove accidental backticks or wrapping quotes
        json_str = json_str.strip().strip('`').strip('"').strip("'")
        
        # 2. Fix double escaping of backslashes (common in Docker/Coolify)
        # This turns \\n into \n before json loading
        json_str = json_str.replace("\\\\n", "\\n")
        
        # 3. Load the JSON
        info = json.loads(json_str)
        
        # 4. Final fix for the private key specifically
        if "private_key" in info:
            # Ensure the private key has actual newline characters
            info["private_key"] = info["private_key"].replace("\\n", "\n")
            
        return info
    except Exception as e:
        logger.error(f"Failed to parse Google Credentials JSON: {e}")
        return None

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot initialization for lead: {lead_data['id']}")
    
    # --- 1. TELNYX HANDSHAKE (CRITICAL) ---
    # We must wait for the first message to get the real stream_id
    stream_id = None
    try:
        # Telnyx sends a 'connected' or 'start' message immediately
        first_msg = await websocket_client.receive_json()
        logger.debug(f"Received initial Telnyx message: {first_msg}")
        
        if first_msg.get("event") in ["connected", "start"]:
            # Extract stream_id from the nested 'start' object or top level
            stream_id = first_msg.get("stream_id") or first_msg.get("start", {}).get("stream_id")
            
        if not stream_id:
            logger.warning("Could not find stream_id in first message, using placeholder")
            stream_id = "telnyx_stream_placeholder"
        else:
            logger.info(f"Using Telnyx stream_id: {stream_id}")
    except Exception as e:
        logger.error(f"Error during Telnyx handshake: {e}")
        stream_id = "telnyx_stream_placeholder"

    # --- 2. CREDENTIALS SETUP ---
    google_creds_dict = get_google_credentials()
    google_creds_obj = None

    if google_creds_dict:
        try:
            google_creds_obj = service_account.Credentials.from_service_account_info(google_creds_dict)
            logger.info("Google Service Account credentials initialized successfully")
        except Exception as e:
            logger.error(f"Error creating Service Account object: {e}")

    # --- 3. SERVICES SETUP ---
    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.0, start_secs=0.2, stop_secs=0.4, confidence=0.5))
    
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2",
        language="ar",
        sample_rate=8000
    )

    # Tool definitions
    tools = [
        {
            "function_declarations": [
                {
                    "name": "update_lead_status_confirmed",
                    "description": "Call this when the customer confirms the order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                    },
                },
                {
                    "name": "update_lead_status_cancelled",
                    "description": "Call this when the customer cancels the order.",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                    },
                },
            ]
        }
    ]

    # LLM Service (Vertex AI)
    llm = GoogleVertexLLMService(
        project_id=os.getenv("GOOGLE_PROJECT_ID") or google_creds_dict.get("project_id"),
        location="us-central1",
        model="gemini-1.5-flash-001",
        tools=tools,
        credentials=google_creds_dict  # Pipecat takes the dict here
    )
    
    # TTS Service
    tts = GoogleTTSService(
        voice_id="ar-JO-Standard-A",
        credentials=google_creds_obj  # Pipecat takes the credentials object here
    )

    # Register functions
    async def confirm_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Tool Call: Confirming order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CONFIRMED')
        await result_callback("تم تأكيد الطلب بنجاح")

    async def cancel_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Tool Call: Cancelling order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CANCELLED')
        await result_callback("تم إلغاء الطلب")

    llm.register_function("update_lead_status_confirmed", confirm_order)
    llm.register_function("update_lead_status_cancelled", cancel_order)

    # --- 4. TRANSPORT & PIPELINE ---
    serializer = TelnyxFrameSerializer(
        stream_id=stream_id,
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding="PCMU",
        inbound_encoding="PCMU",
        params=TelnyxFrameSerializer.InputParams(sample_rate=8000)
    )
    
    transport = FastAPIWebsocketTransport(
        websocket=websocket_client,
        params=FastAPIWebsocketParams(
            serializer=serializer,
            add_wav_header=False,
            session_timeout=300,
            vad_enabled=True,
            vad_analyzer=vad,
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000
        )
    )

    system_prompt = f"""You are Kawkab AI, a delivery assistant. Speak in Jordanian Arabic.
Greet {lead_data['customer_name']}. Confirm they ordered {lead_data['order_items']} for delivery at {lead_data['delivery_time']}.
If confirmed, call tool update_lead_status_confirmed().
If cancelled, call tool update_lead_status_cancelled().
"""

    context = LLMContext(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "ابدأ المحادثة الآن ورحب بالعميل"}
        ]
    )

    user_agg = LLMUserAggregator(context)
    assistant_agg = LLMAssistantAggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        assistant_agg,
        tts,
        transport.output()
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner()
    
    await task.queue_frames([LLMContextFrame(context)])
    
    logger.info("Bot Pipeline running...")
    await runner.run(task)
