from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import io
import os
import uvicorn

app = FastAPI(title="SoundMirror Phoneme Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Lazy load model to avoid Railway startup timeout
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        print(f"Loading model: {MODEL_ID}")
        processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
        print("Model loaded successfully!")

def load_audio(file_bytes):
    audio, sr = torchaudio.load(io.BytesIO(file_bytes))
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    return audio.squeeze()

@app.get("/")
async def root():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "language": LANGUAGE,
        "service": "SoundMirror Phoneme Backend"
    }

@app.post("/phonemes")
async def analyze_phonemes(
    file: UploadFile = File(...),
    lang: str = Query(LANGUAGE)
):
    try:
        audio_bytes = await file.read()
        waveform = load_audio(audio_bytes)
        
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
        
        tokens = transcription.strip().split()
        
        return {
            "lang": lang,
            "model": MODEL_ID,
            "phonemes": transcription,
            "phoneme_list": tokens,
            "tokens": tokens,
            "clean_tokens": tokens,
            "primary": tokens[0] if tokens else "",
            "raw_transcription": transcription,
            "ipa_units": tokens,
            "raw_text": transcription
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/expected_ipa")
async def get_expected_ipa(text: str = Query(...), lang: str = Query(LANGUAGE)):
    return {
        "text": text,
        "lang": lang,
        "ipa": text.lower()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
