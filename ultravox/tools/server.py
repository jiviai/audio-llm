import uvicorn
import asyncio
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from typing import Optional, Generator
import tempfile
import shutil

from ultravox import data as datasets
from ultravox.inference import ultravox_infer
from ultravox.inference import base as infer_base

# Initialize FastAPI app
app = FastAPI(title="Ultravox Streaming LLM API")


SYS_PROMPT = """As a medical professional, your task is to ask most relevant questions from the patient so that we can reach to a conclusive diagnosis. Talk to patient and follow the below guidelines:

Stage 1: Gather all necessary information related to the chief complaint keeping SOCRATES principle (DO NOT state reference of SOCRATES in output question)
Stage 2: Ask specific questions to eliminate diseases to narrow down differential diagnosis
Keep question short and to the point.
Ask only 1 question at a time and wait for the response."""

# Model configuration
class ModelConfig:
    model_path: str = "/home/akshat/audio-llm/checkpoint-25000"
    device: Optional[str] = None
    data_type: Optional[str] = None
    conversation_mode: bool = True

config = ModelConfig()
inference = ultravox_infer.UltravoxInference(
    config.model_path, device=config.device, data_type=config.data_type, conversation_mode=True, system_prompt=SYS_PROMPT
)

# ✅ Fix: Extract text from `InferenceChunk` objects before yielding
async def async_generator_wrapper(sync_generator: Generator):
    """Wraps a synchronous generator and extracts text from InferenceChunk."""
    for chunk in sync_generator:
        if isinstance(chunk, infer_base.InferenceChunk):  # Extract text from chunk
            yield chunk.text + "\n"
        else:
            yield str(chunk) + "\n"
        await asyncio.sleep(0)  # Ensure non-blocking behavior

# ✅ Unified streaming endpoint for text & audio inference
@app.post("/stream_infer")
async def stream_infer(
    prompt: str = Form(...),
    audio: Optional[UploadFile] = File(None),
    max_new_tokens: int = Form(200),
    temperature: float = Form(0.0),
    reset: bool = Form(False),
):
    """Handles both text & audio input, streaming the response."""
    
    # Handle Audio Input
    if audio:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            shutil.copyfileobj(audio.file, temp_audio)
            audio_path = temp_audio.name

        if "<|audio|>" not in prompt:
            prompt += "<|audio|>"

        sample = datasets.VoiceSample.from_prompt_and_file(prompt, audio_path)
    else:
        sample = datasets.VoiceSample.from_prompt(prompt)

    if len(sample.messages) != 1:
        return {"error": f"Expected exactly 1 message but got {len(sample.messages)}"}

    # ✅ Fix: Wrap `infer_stream()` in an async generator
    if reset:
        inference.update_conversation()
    return StreamingResponse(
        async_generator_wrapper(
            inference.infer_stream(sample, max_tokens=max_new_tokens, temperature=temperature)
        ),
        media_type="text/plain",
    )

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
