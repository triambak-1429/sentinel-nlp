import os
import shutil
import time
import urllib.request
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from src.config import MODEL_DIR, ONNX_MODEL_PATH, ID2LABEL
from app.schemas import DetectedEntity, AnalyzeResponse

WEIGHTS_DATA_PATH = Path("models/pii_detector.onnx.data")
WEIGHTS_URL = "https://huggingface.co/triambak-1429/sentinel-nlp-onnx/resolve/main/pii_detector.onnx.data"

class SentinelInferenceEngine:
    def __init__(self):
        if not WEIGHTS_DATA_PATH.exists():
            print("ONNX NLP weights not found. Downloading from Hugging Face...")
            WEIGHTS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            req = urllib.request.Request(WEIGHTS_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(WEIGHTS_DATA_PATH, "wb") as f:
                # Stream in 8MB chunks to prevent 512MB RAM spikes on Render
                shutil.copyfileobj(response, f, length=8 * 1024 * 1024)
            print("Download completed successfully!")

        tokenizer_json_path = MODEL_DIR / "best_pii_model" / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128)

        self.session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

    def predict(self, prompt: str) -> AnalyzeResponse:
        start_time = time.perf_counter()
        
        encoded = self.tokenizer.encode(prompt)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        logits = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        predictions = np.argmax(logits, axis=-1)[0]
        
        tokens = encoded.tokens
        offsets = encoded.offsets
        
        detected_entities = []
        current_entity_tokens = []
        current_entity_type = None
        has_pii = False

        for token, pred_id, (start, end) in zip(tokens, predictions, offsets):
            if start == end or token in ["<s>", "</s>", "<pad>", "<unk>"]:
                continue
                
            tag = ID2LABEL.get(pred_id, "O")
            clean_tag = tag.split("-")[-1] if tag != "O" else "O"

            if clean_tag != "O":
                has_pii = True
                if current_entity_type == clean_tag and not tag.startswith("B-"):
                    current_entity_tokens.append(prompt[start:end])
                else:
                    if current_entity_tokens and current_entity_type:
                        full_value = "".join(current_entity_tokens).strip()
                        detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))
                    current_entity_tokens = [prompt[start:end]]
                    current_entity_type = clean_tag
            else:
                if current_entity_tokens and current_entity_type:
                    full_value = "".join(current_entity_tokens).strip()
                    detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))
                    current_entity_tokens = []
                    current_entity_type = None

        if current_entity_tokens and current_entity_type:
            full_value = "".join(current_entity_tokens).strip()
            detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))

        latency = (time.perf_counter() - start_time) * 1000
        risk_score = round(min(1.0, len(detected_entities) * 0.35 + (0.5 if has_pii else 0.0)), 2)

        return AnalyzeResponse(
            prompt=prompt,
            contains_pii=has_pii,
            risk_score=risk_score,
            latency_ms=round(latency, 2),
            detected_entities=detected_entities
        )