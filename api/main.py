import os
from typing import List, Optional
from enum import Enum
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from openai import OpenAI
import soundfile as sf
import gigaam
import librosa
import numpy as np
import httpx
from fastapi import Body

from io import BytesIO


VOXTRAL_API="http://158.160.33.135:8000/v1"

class ModelType(str, Enum):
    GIGAAM = "gigaam"
    WHISPER_V3 = "whisper-v3"
    VOXTRAL_MINI = "voxtral-mini"

class ArrayUploadFile:
    def __init__(self, filename: str, file, content_type: str = "audio/wav"):
        self.filename = filename
        self.file = file
        self.content_type = content_type
        self.size = None
    
    async def read(self) -> bytes:
        if hasattr(self.file, 'read'):
            return self.file.read()
        return b''
    
    def close(self):
        if hasattr(self.file, 'close'):
            self.file.close()

try:
    gigaam_model = gigaam.load_model("v2_rnnt")
except Exception as e:
    raise RuntimeError(f"Could not load GigaAM model: {e}")


def validate_and_preprocess_audio(file_path: str, model_type: ModelType) -> str:
    try:
        audio, sr = librosa.load(file_path, sr=None)
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
        
        if len(audio) / sr < 0.1:
            raise ValueError(f"Audio too short: {len(audio) / sr:.2f}s (minimum 0.1s required)")
        
        if model_type == ModelType.GIGAAM:
            target_sr = 16000
            if sr != target_sr:
                print(f"Resampling from {sr}Hz to {target_sr}Hz for GigaAM")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                
                processed_path = file_path.replace('.mp3', '_processed.wav')
                sf.write(processed_path, audio, target_sr)
                return processed_path
        
        return file_path
        
    except Exception as e:
        raise ValueError(f"Audio validation failed: {str(e)}")

try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-large-v3"
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    whisper_model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model=whisper_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
except Exception as e:
    raise RuntimeError(f"Could not load Whisper-large-v3 model: {e}")


try:
    voxtral_client = OpenAI(
        api_key="EMPTY", 
        base_url=VOXTRAL_API
    )
    print("✅ Voxtral-Mini client initialized successfully")
except Exception as e:
    print(f"⚠️ Warning: Could not initialize Voxtral-Mini client: {e}")
    voxtral_client = None


app = FastAPI(
    title="Multi-Model Speech-to-Text API",
    description="API service that transcribes audio using GigaAM v2_rnnt, Whisper-large-v3, or Voxtral-Mini.",
    version="3.0.0"
)


async def is_voxtral_healthy(base_url: str = VOXTRAL_API, timeout_sec: int = 5) -> bool:
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code == 200
    except Exception:
        return False

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...), 
    model_type: ModelType = Query(ModelType.GIGAAM, description="Choose transcription model"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'ru', 'en')"),
    prompt: Optional[str] = Query(None, description="Prompt for Voxtral model guidance"),
    temperature: float = Query(0.0, description="Temperature for Voxtral model (0.0-1.0)")
) -> JSONResponse:
    if file.content_type.split("/")[0] != "audio":
        raise HTTPException(status_code=415, detail="Unsupported file type")

    audio_bytes = await file.read()
    
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    tmp_path = f"/tmp/{file.filename}"
    processed_path = tmp_path
    
    try:
        with open(tmp_path, "wb") as tmp:
            tmp.write(audio_bytes)
        
        processed_path = validate_and_preprocess_audio(tmp_path, model_type)
        
        if model_type == ModelType.GIGAAM:
            try:
                result = gigaam_model.transcribe_longform(processed_path)
                
                if not result or len(result) == 0:
                    raise ValueError("GigaAM returned empty result")
                    
            except Exception as gigaam_error:
                print(f"GigaAM longform failed: {gigaam_error}")
                
                try:
                    result = gigaam_model.transcribe(processed_path) 
                except:
                    raise HTTPException(
                        status_code=500, 
                        detail=f"GigaAM transcription failed: {str(gigaam_error)}. Try using a different audio file or model."
                    )
            
        elif model_type == ModelType.WHISPER_V3:
            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language
            
            whisper_result = whisper_pipeline(
                tmp_path,
                return_timestamps=True,
                generate_kwargs=generate_kwargs
            )
            
            result = [{
                "transcription": whisper_result["text"],
                "boundaries": [0, len(whisper_result["text"])],  
                "chunks": whisper_result.get("chunks", [])
            }]
            
        elif model_type == ModelType.VOXTRAL_MINI:
            if voxtral_client is None:
                raise HTTPException(status_code=503, detail="Voxtral-Mini client not available")
            
            try:
                with open(tmp_path, "rb") as audio_file:
                    transcription_params = {
                        "file": audio_file,
                        "model": "mistralai/Voxtral-Mini-3B-2507",
                        "temperature": temperature,
                    }
                    
                    if language:
                        transcription_params["language"] = language
                    if prompt:
                        transcription_params["prompt"] = prompt
                    
                    response = voxtral_client.audio.transcriptions.create(**transcription_params)
                
                result = [{
                    "transcription": response.text,
                    "boundaries": [0, len(response.text)],
                    "chunks": [] 
                }]
                
            except Exception as api_error:
                raise HTTPException(status_code=502, detail=f"Voxtral API error: {str(api_error)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        for path in [tmp_path, processed_path]:
            if path and os.path.exists(path) and path != tmp_path:
                os.remove(path)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return JSONResponse({
        "model_used": model_type.value,
        "result": result
    })

@app.post("/transcribe/array")
async def transcribe_array(
    audio_array: List[float] = Body(..., description="Flat float32 audio array"),
    sr: int = Body(..., ge=8000, le=48000, description="Sample rate (e.g., 16000)"),
    model_type: ModelType = Query(ModelType.GIGAAM, description="Model for transcription: gigaam, whisper-v3, voxtral-mini"),
    language: Optional[str] = Query(None, description="Optional transcription language hint"),
    prompt: Optional[str] = Query(None, description="Prompt for Voxtral"),
    temperature: float = Query(0.0, ge=0.0, le=1.0, description="Temperature for Voxtral")
) -> JSONResponse:
    try:
        audio_np = np.array(audio_array, dtype=np.float32)

        if len(audio_np) == 0:
            raise HTTPException(status_code=400, detail="Empty audio array.")
        if len(audio_np) / sr < 0.1:
            raise HTTPException(status_code=400, detail=f"Audio too short: {len(audio_np)/sr:.2f}s")

        with BytesIO() as buf:
            sf.write(buf, audio_np, sr, format='WAV')
            buf.seek(0)
            tmp_path = "/tmp/array_input.wav"
            with open(tmp_path, "wb") as f:
                f.write(buf.read())
        
        file_wrapper = ArrayUploadFile(
            filename="array.wav", 
            file=open(tmp_path, "rb"),
            content_type="audio/wav"
        )

        response = await transcribe(
            file=file_wrapper,
            model_type=model_type,
            language=language,
            prompt=prompt,
            temperature=temperature
        )

        file_wrapper.close()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process array audio: {str(e)}")


@app.post("/transcribe/gigaam")
async def transcribe_gigaam(file: UploadFile = File(...)) -> JSONResponse:
    return await transcribe(file, ModelType.GIGAAM)


@app.post("/transcribe/whisper")
async def transcribe_whisper(
    file: UploadFile = File(...),
    language: Optional[str] = Query(None, description="Language code (e.g., 'ru', 'en', 'es')")
) -> JSONResponse:
    return await transcribe(file, ModelType.WHISPER_V3, language)


@app.post("/transcribe/voxtral")
async def transcribe_voxtral(
    file: UploadFile = File(...),
    language: Optional[str] = Query("ru", description="Language code (e.g., 'ru', 'en')"),
    prompt: Optional[str] = Query(None, description="Prompt to guide transcription"),
    temperature: float = Query(0.0, description="Temperature for model creativity (0.0-1.0)")
) -> JSONResponse:
    return await transcribe(file, ModelType.VOXTRAL_MINI, language, prompt, temperature)


@app.get("/models/info")
async def get_models_info():
    voxtral_status = "available" if voxtral_client is not None else "unavailable"
    
    return JSONResponse({
        "available_models": {
            "gigaam": {
                "name": "GigaAM v2 RNN-T",
                "description": "Optimized for Russian speech recognition",
                "best_for": "Russian language audio with high accuracy",
                "features": ["Long-form transcription", "Precise timestamps", "Russian optimization"],
                "status": "available"
            },
            "whisper-v3": {
                "name": "OpenAI Whisper Large v3",
                "description": "Multilingual speech recognition and translation",
                "best_for": "Multi-language support, international content",
                "features": ["99+ languages", "Automatic language detection", "Translation capabilities"],
                "status": "available"
            },
            "voxtral-mini": {
                "name": "Mistral Voxtral-Mini 3B",
                "description": "External API model with prompt support",
                "best_for": "Guided transcription with context prompts",
                "features": ["Prompt guidance", "Temperature control", "External API"],
                "status": voxtral_status,
                "endpoint": VOXTRAL_API
            }
        },
        "device": device,
        "torch_dtype": str(torch_dtype)
    })

@app.get("/health")
async def health_check():
    voxtral_status = await is_voxtral_healthy()
    models_status = {
        "gigaam": True,
        "whisper-v3": True,
        "voxtral-mini": voxtral_status
    }
    return JSONResponse({
        "status": "healthy" if all(models_status.values()) else "degraded",
        "models": models_status,
        "external_services": {
            "voxtral_api": voxtral_status
        }
    })