import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from src.config import MODEL_DIR, ONNX_MODEL_PATH, ID2LABEL
from app.schemas import DetectedEntity, AnalyzeResponse

class SentinelInferenceEngine:
    def __init__(self):
        pytorch_model_path = MODEL_DIR / "best_pii_model"
        self.tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)
        self.session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

    def predict(self, prompt: str) -> AnalyzeResponse:
        start_time = time.perf_counter()
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="np", 
            truncation=True, 
            max_length=128
        )
        
        ort_inputs = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        
        logits = self.session.run(None, ort_inputs)[0]
        predictions = np.argmax(logits, axis=-1)[0]
        
        input_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        detected_entities = []
        has_pii = False
        
        for token, pred_id in zip(tokens, predictions):
            tag = ID2LABEL.get(pred_id, "O")
            if tag != "O" and token not in ["<s>", "</s>", "<pad>"]:
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