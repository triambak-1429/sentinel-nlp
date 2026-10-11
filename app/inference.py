import time
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from src.config import MODEL_DIR, ONNX_MODEL_PATH, ID2LABEL
from app.schemas import DetectedEntity, AnalyzeResponse

class SentinelInferenceEngine:
    def __init__(self):
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
        offsets = encoded.offsets  # Character start/end boundaries in original prompt
        
        detected_entities = []
        current_entity_tokens = []
        current_entity_type = None
        has_pii = False

        for token, pred_id, (start, end) in zip(tokens, predictions, offsets):
            # Skip special tokens and zero-length offsets
            if start == end or token in ["<s>", "</s>", "<pad>", "<unk>"]:
                continue
                
            tag = ID2LABEL.get(pred_id, "O")
            
            # Extract base tag without prefix (e.g., 'B-EMAIL' -> 'EMAIL')
            clean_tag = tag.split("-")[-1] if tag != "O" else "O"

            if clean_tag != "O":
                has_pii = True
                
                # Check if continuing an entity sequence or starting a new one
                if current_entity_type == clean_tag and not tag.startswith("B-"):
                    current_entity_tokens.append(prompt[start:end])
                else:
                    # Flush completed entity
                    if current_entity_tokens and current_entity_type:
                        full_value = "".join(current_entity_tokens).strip()
                        detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))
                    
                    # Start new entity sequence
                    current_entity_tokens = [prompt[start:end]]
                    current_entity_type = clean_tag
            else:
                # Flush active entity when encountering non-PII token 'O'
                if current_entity_tokens and current_entity_type:
                    full_value = "".join(current_entity_tokens).strip()
                    detected_entities.append(DetectedEntity(token=full_value, entity_type=current_entity_type))
                    current_entity_tokens = []
                    current_entity_type = None

        # Flush any trailing entity
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