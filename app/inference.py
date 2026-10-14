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
        import time
        start_time = time.perf_counter()
        
        encoded = self.tokenizer.encode(prompt)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        logits = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})[0]
        predictions = np.argmax(logits, axis=-1)[0]
        
        tokens = encoded.tokens
        offsets = encoded.offsets
        
        detected_entities = []
        current_entity_type = None
        current_start = None
        current_end = None

        for token, pred_id, (start, end) in zip(tokens, predictions, offsets):
            # Skip special tokens or zero-length offsets
            if start == end or token in ["<s>", "</s>", "<pad>", "<unk>"]:
                continue
                
            tag = ID2LABEL.get(pred_id, "O")
            # Strip B- and I- prefixes to get the raw entity type
            clean_tag = tag.split("-")[-1] if tag != "O" else "O"

            if clean_tag != "O":
                # AGGRESSIVE MERGE: If it's the same entity type as the previous token, combine them
                if current_entity_type == clean_tag:
                    current_end = max(current_end, end)
                else:
                    # Different entity type -> save the old one and start a new one
                    if current_entity_type is not None:
                        full_value = prompt[current_start:current_end].strip()
                        if full_value:
                            detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))
                    
                    current_entity_type = clean_tag
                    current_start = start
                    current_end = end
            else:
                # Hit an "O" tag -> close out the current entity
                if current_entity_type is not None:
                    full_value = prompt[current_start:current_end].strip()
                    if full_value:
                        detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))
                    current_entity_type = None

        # Flush any trailing entity after the loop ends
        if current_entity_type is not None:
            full_value = prompt[current_start:current_end].strip()
            if full_value:
                detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))

        latency = (time.perf_counter() - start_time) * 1000
        has_pii = len(detected_entities) > 0
        risk_score = round(min(1.0, len(detected_entities) * 0.35 + (0.5 if has_pii else 0.0)), 2)

        return AnalyzeResponse(
            prompt=prompt,
            contains_pii=has_pii,
            risk_score=risk_score,
            latency_ms=round(latency, 2),
            detected_entities=detected_entities
        )