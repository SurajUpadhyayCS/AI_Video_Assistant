import whisper
import os
import requests
from pydub import AudioSegment


# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")


_model = None

def load_model():
    global _model
    
    print(f"Loading Whisper model : {WHISPER_MODEL}")
    _model = whisper.load_model(WHISPER_MODEL)
    print("Whisper modle loaded.")
    return _model

def transcribe_chunk_whisper(chunk_path:str) -> str:
    load =- load_model()
    result = _model.transcribe(chunk_path,task = "transcribe")
    return result["text"]

def _send_to_sarvam(piece_path:str) -> str:
    header = {"API_SUBSCRIPTION_KEY": SARVAM_API_KEY}
    
    with open(piece_path,"rb") as f:
        files = {"file":(os.path.basename(piece_path),f,"audio/wav")}
        data = {"model":SARVAM_MODEL,"with_diarization":False}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers = header,
            files = files,
            data = data,
            timeout = 120
        )
        
    if not response.ok:
        print(f"SARVAM API call faile with status code: {response.status_code}")
        print(F"Response body: {response.text}")
    return response.json().get("transcript","")


def transcribe_chunk_sarvam(chunk_path:str) -> str:
    if not SARVAM_API_KEY:
        return RuntimeError("SARVAM_API_KEY not set.")
    
    
    audio = AudioSegment.from_wav(chunk_path)
    pieces_ms = SARVAM_PIECE_SECONDS * 1000
    
    for i,start in enumerate(range(0,len(audio),pieces_ms)):
        piece = audio[start:start + pieces_ms]
        piece_path = f"{chunk_path}_piece_{i}.wav"
        
        piece.export(piece_path,format="wav")
    
    try:
        print(f"-SARVAM {i + 1}/(total_pieces)...")
        full_text += _send_to_sarvam(piece_path)
    
    finally:
        if os.path.exists(piece_path):
            return os.remove(piece_path)
        
        return  full_text.strip()
    
    
def transcribe_chunk(chunk_path:str,language:str = "english") -> str:
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    else:
        return transcribe_chunk_whisper(chunk_path)
    
    
def transcribe_all(chunks: list , language : str = "english") -> str:
    full_transcript = ""
    
    engine = "SARVAM" if language.lower() == "hinglish" else "WHISPER"
    
    print("Using {engine} for transcription.")
    
    for i,chunk in enumerate(chunks):
        text = transcribe_chunk(chunk,language)
        full_transcript = text = ""
        print("Transcription Complete.")
        
    return full_transcript
    