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
from pipecat.frames.frames import LLMContextFrame, EndFrame, LLMRunFrame, LLMMessagesFrame, TTSSpeakFrame
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

# Helper function to get Google Credentials
def get_google_credentials():
    # 1. Try File Path First (Preferred)
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if path and os.path.exists(path):
        logger.info(f"Found Google Credentials file at: {path}")
        return None # Return None so run_bot uses the path directly

    # 2. Try JSON String (Fallback)
    json_str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if json_str:
        logger.info("Loading Google Credentials from JSON string env var")
        
        # Basic cleanup of wrapping quotes
        json_str = json_str.strip().strip("'").strip('"')
        
        # Handle escaped quotes commonly found in env vars (e.g. \"type\" -> "type")
        json_str = json_str.replace('\\"', '"')
        
        # Handle boolean values if they are Python-style
        json_str = json_str.replace("True", "true").replace("False", "false")

        try:
            # Try parsing as standard JSON
            info = json.loads(json_str)
            logger.info("Successfully parsed credentials as JSON")
        except json.JSONDecodeError:
            # Fallback: Try parsing as a Python dictionary string
            try:
                logger.info("JSON decode failed, trying ast.literal_eval")
                info = ast.literal_eval(json_str)
            except (ValueError, SyntaxError) as e:
                logger.error(f"Failed to parse Google Credentials: {e}")
                logger.error(f"First 100 chars: {json_str[:100]}")
                return None
        
        # Post-processing: Handle private key newlines if needed
        if "private_key" in info:
            # Ensure we have actual newlines, not escaped ones
            info["private_key"] = info["private_key"].replace("\\n", "\n")
            
        return info
    
    return None

async def run_bot(websocket_client, lead_data, call_control_id=None):
    logger.info(f"Starting bot for lead: {lead_data['id']}")

    # 0. Handle Telnyx Handshake to get stream_id
    stream_id = "telnyx_stream_placeholder"
    inbound_encoding = "PCMU" # Default to PCMU (G.711u)
    try:
        # Telnyx typically sends a JSON payload first with event="connected"
        # Then it sends event="start" which contains the stream_id
        # We need to loop until we find the stream_id or a reasonable timeout/limit
        
        logger.info("Waiting for Telnyx 'start' event with stream_id...")
        
        for _ in range(3): # Try up to 3 messages
            msg_text = await websocket_client.receive_text()
            logger.info(f"Received Telnyx message: {msg_text}")
            msg = json.loads(msg_text)
            
            # Check if this message has media_format information (usually in 'start' event)
            if "media_format" in msg:
                 encoding = msg["media_format"].get("encoding", "").upper()
                 logger.info(f"Telnyx Media Format (direct): {msg['media_format']}")
                 if encoding == "G729":
                     logger.error("CRITICAL: Telnyx is sending G.729 audio. Pipecat requires PCMU (G.711u) or PCMA (G.711a).")
                     logger.error("Please disable G.729 in your Telnyx Portal SIP Connection settings.")
                 elif encoding == "PCMA":
                     logger.info("Detected PCMA encoding, updating serializer.")
                     inbound_encoding = "PCMA"
                 elif encoding == "PCMU":
                     inbound_encoding = "PCMU"

            elif "start" in msg and "media_format" in msg["start"]:
                 encoding = msg["start"]["media_format"].get("encoding", "").upper()
                 logger.info(f"Telnyx Media Format (nested in start): {msg['start']['media_format']}")
                 if encoding == "G729":
                     logger.error("CRITICAL: Telnyx is sending G.729 audio. Pipecat requires PCMU (G.711u) or PCMA (G.711a).")
                     logger.error("Please disable G.729 in your Telnyx Portal SIP Connection settings.")
                 elif encoding == "PCMA":
                     logger.info("Detected PCMA encoding, updating serializer.")
                     inbound_encoding = "PCMA"
                 elif encoding == "PCMU":
                     inbound_encoding = "PCMU"

            # Check standard locations for stream_id
            if "stream_id" in msg:
                stream_id = msg["stream_id"]
                logger.info(f"Captured stream_id (direct): {stream_id}")
                break
            elif "data" in msg and "stream_id" in msg["data"]:
                stream_id = msg["data"]["stream_id"]
                logger.info(f"Captured stream_id (in data): {stream_id}")
                break
            elif msg.get("event") == "start":
                 # capture stream_id if present
                 if "stream_id" in msg:
                      stream_id = msg["stream_id"]
                      logger.info(f"Captured stream_id (from start event): {stream_id}")
                      break
            
            # If we didn't find it, we loop again to get the next message
            
        if stream_id == "telnyx_stream_placeholder":
             logger.warning("Could not find stream_id in initial messages, using placeholder.")
             
    except Exception as e:
        logger.error(f"Failed to capture stream_id from initial message: {e}")

    # Get Credentials (dictionary)
    google_creds_dict = get_google_credentials()
    
    # Prepare Credentials for different services
    google_creds_str = None
    google_creds_obj = None

    if google_creds_dict:
        # LLM needs JSON string
        google_creds_str = json.dumps(google_creds_dict)
        # TTS likely needs Credentials object
        google_creds_obj = service_account.Credentials.from_service_account_info(google_creds_dict)
    
    # 1. Services
    vad = SileroVADAnalyzer(params=VADParams(min_volume=0.0, start_secs=0.2, stop_secs=0.4, confidence=0.5))
    
    deepgram_key = os.getenv("DEEPGRAM_API_KEY")
    if not deepgram_key:
        logger.error("CRITICAL: DEEPGRAM_API_KEY is missing in environment variables!")
    else:
        logger.info(f"Deepgram API Key found (len={len(deepgram_key)}).")

    dg_encoding = "mulaw" if inbound_encoding == "PCMU" else "alaw" if inbound_encoding == "PCMA" else "linear16"
    logger.info(f"Initializing Deepgram with encoding: {dg_encoding} (inbound: {inbound_encoding})")

    stt = DeepgramSTTService(
        api_key=deepgram_key,
        model="nova-2",
        language="ar",
        sample_rate=8000,
        encoding=dg_encoding
    )
    
    # 2. Tools
    tools = [
        {
            "function_declarations": [
                {
                    "name": "update_lead_status_confirmed",
                    "description": "Call this when the customer confirms the order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Reason for confirmation"}
                        },
                    },
                },
                {
                    "name": "update_lead_status_cancelled",
                    "description": "Call this when the customer cancels the order.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Reason for cancellation"}
                        },
                    },
                },
            ]
        }
    ]

    # Initialize LLM with either path or credentials object
    llm_kwargs = {
        "project_id": os.getenv("GOOGLE_PROJECT_ID"),
        "location": "us-central1",
        "model": "gemini-2.5-flash-lite",
        "tools": tools
    }
    
    if google_creds_str:
        llm_kwargs["credentials"] = google_creds_str
    else:
        llm_kwargs["credentials_path"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    llm = GoogleVertexLLMService(**llm_kwargs)
    
    tts_encoding = "mulaw" if inbound_encoding == "PCMU" else "alaw" if inbound_encoding == "PCMA" else "linear16"
    tts_kwargs = {
        "voice_id": "ar-JO-Standard-A",
        "sample_rate": 8000,
        "encoding": tts_encoding
    }
    if google_creds_obj:
        tts_kwargs["credentials"] = google_creds_obj
    else:
        tts_kwargs["credentials_path"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    tts = GoogleTTSService(**tts_kwargs)

    # 3. Handlers
    async def confirm_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Confirming order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CONFIRMED')
        await result_callback("Order confirmed successfully.")

    async def cancel_order(function_name, tool_call_id, args, llm, context, result_callback):
        logger.info(f"Cancelling order for lead {lead_data['id']}")
        update_lead_status(lead_data['id'], 'CANCELLED')
        await result_callback("Order cancelled.")

    llm.register_function(
        "update_lead_status_confirmed",
        confirm_order
    )
    llm.register_function(
        "update_lead_status_cancelled",
        cancel_order
    )

    # 3. Transport (Telnyx)
    # Note: TelnyxFrameSerializer requires stream_id and call_control_id for full functionality (like hanging up)
    # In a basic setup, we might not have them immediately from the websocket handshake in the same way.
    # We will instantiate it with placeholders or extract if possible.
    # For now, we'll use a dummy stream_id if we don't have one, or rely on what Pipecat needs.
    
    # We need to know the call_control_id to hang up. 
    # If we can't get it from the websocket, we might need to pass it from the webhook handler -> main.py -> run_bot
    # But for now, let's assume we just want audio streaming working.
    
    serializer = TelnyxFrameSerializer(
        stream_id=stream_id, # Captured from handshake
        call_control_id=call_control_id,
        api_key=os.getenv("TELNYX_API_KEY"),
        outbound_encoding=inbound_encoding,
        inbound_encoding=inbound_encoding,
        params=TelnyxFrameSerializer.InputParams(
            sample_rate=8000
        )
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

    # 4. Prompt
    system_prompt = f"""You are Kawkab AI, a delivery assistant. Speak in Jordanian Arabic.
Greet {lead_data['customer_name']}. Confirm they ordered {lead_data['order_items']} for delivery at {lead_data['delivery_time']}.
If confirmed, call tool update_lead_status_confirmed().
If cancelled, call tool update_lead_status_cancelled().
If no answer or voicemail, just hang up (I will handle this via timeout or silence).
"""

    # 4. Prompt & Context
    # Single context with system prompt AND initial user trigger
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Start the conversation by greeting the customer."}
    ]
    context = LLMContext(messages=messages)
    context_frame = LLMContextFrame(context)

    # 5. Pipeline Phase 1: Greeting (Direct TTS - NO LLM)
    # This guarantees the greeting is spoken immediately without LLM latency or triggering issues.
    
    pipeline_greeting = Pipeline([
        tts,
        transport.output()
    ])

    task_greeting = PipelineTask(pipeline_greeting)
    
    # 6. Pipeline Phase 2: Listening (STT + LLM Enabled)
    # We enable STT only after the greeting is likely finished.
    # Disable user turn strategies to prevent false interruptions from Telnyx comfort noise
    user_agg = LLMUserAggregator(
        context,
        user_turn_start_strategy=None,
        user_turn_stop_strategy=None
    )
    assistant_agg_listening = LLMAssistantAggregator(context)

    pipeline_listening = Pipeline([
        transport.input(),
        stt,
        user_agg,
        llm,
        assistant_agg_listening,
        tts,
        transport.output()
    ])

    # Allow interruptions must be False to prevent Telnyx comfort noise from cancelling TTS
    task_listening = PipelineTask(pipeline_listening, params=PipelineParams(allow_interruptions=False))
    
    runner = PipelineRunner()
    
    # Construct the greeting text from the lead data
    greeting_text = f"Marhaba {lead_data.get('customer_name', 'Customer')}. Ana Kawkab AI from delivery service. Just confirming your order of {lead_data.get('order_items', 'items')}."
    
    logger.info("Starting Phase 1: Greeting (Direct TTS)...")
    await task_greeting.queue_frames([TTSSpeakFrame(greeting_text)])
    
    # Run Phase 1 in background and wait for it to process
    t1 = asyncio.create_task(runner.run(task_greeting))
    
    # Wait for greeting to be spoken (approx 4 seconds for a short greeting)
    # Since we don't have a callback for "TTS finished", we use a safe delay.
    await asyncio.sleep(4.0)
    
    # Cancel Phase 1 to switch to Phase 2
    logger.info("Phase 1 complete. Switching to Phase 2: Listening...")
    await task_greeting.cancel()
    try:
        await t1
    except asyncio.CancelledError:
        pass

    # Run Phase 2
    # We need to queue the context for Phase 2 so the LLM knows what happened
    logger.info("Starting Phase 2: Listening (STT Enabled)...")
    # Add the greeting to the context as if the assistant said it
    messages.append({"role": "assistant", "content": greeting_text})
    updated_context = LLMContext(messages=messages)
    await task_listening.queue_frames([LLMContextFrame(updated_context)])
    
    await runner.run(task_listening)
