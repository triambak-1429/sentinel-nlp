from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.inference import SentinelInferenceEngine

engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SentinelInferenceEngine()
    yield
    engine = None

app = FastAPI(
    title="Sentinel NLP — Real-Time PII & Intent Security Engine",
    description="High-throughput ONNX microservice for detecting PII exfiltration and prompt risks.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "sentinel-nlp"}

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_prompt(request: AnalyzeRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt string cannot be empty.")
    
    if engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized.")
        
    return engine.predict(request.prompt)