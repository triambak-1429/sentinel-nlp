import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from src.config import MODEL_DIR, ONNX_MODEL_PATH, ID2LABEL
from app.schemas import DetectedEntity, AnalyzeResponse

class SentinelInferenceEngine:
    def __init__(self):
        tokenizer_json_path = MODEL_DIR / "best_pii_model" / "tokenizer.json"
        
        # Pure Rust/C++ tokenizer backend (~5MB RAM footprint)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json_path))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(length=128)
        
        # ONNX Runtime Session
        self.session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])

    def predict(self, prompt: str) -> AnalyzeResponse:
        start_time = time.perf_counter()
        
        # Encode prompt
        encoded = self.tokenizer.encode(prompt)
        
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Run ONNX inference
        logits = self.session.run(None, ort_inputs)[0]
        predictions = np.argmax(logits, axis=-1)[0]
        
        tokens = encoded.tokens
        
        detected_entities = []
        has_pii = False
        
        for token, pred_id in zip(tokens, predictions):
            tag = ID2LABEL.get(pred_id, "O")
            if tag != "O" and token not in ["<s>", "</s>", "<pad>", "<unk>"]:
                has_pii = True
                clean_token = token.replace("Ġ", "")
                detected_entities.append(DetectedEntity(token=clean_token, entity_type=tag))

        latency = (time.perf_counter() - start_time) * 1000
        risk_score = round(min(1.0, len(detected_entities) * 0.35 + (0.5 if has_pii else 0.0)), 2)

        return AnalyzeResponse(
            prompt=prompt,
            contains_pii=has_pii,
            risk_score=risk_score,
            latency_ms=round(latency, 2),
            detected_entities=detected_entities
        )