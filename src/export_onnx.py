import time
import torch
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForTokenClassification
from src.config import MODEL_DIR, ONNX_MODEL_PATH

def export_pytorch_to_onnx():
    print("--- Starting Phase 3: ONNX Export & Benchmarking ---")
    
    pytorch_model_path = MODEL_DIR / "best_pii_model"
    
    if not pytorch_model_path.exists():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found at {pytorch_model_path}. Please complete Phase 2 first."
        )

    print(f"[1/3] Loading PyTorch model from {pytorch_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_path)
    model = AutoModelForTokenClassification.from_pretrained(pytorch_model_path)
    model.eval()

    # Create dummy inputs for graph tracing
    dummy_text = "Please contact admin at admin@company.com with key sk-live-12345"
    inputs = tokenizer(dummy_text, return_tensors="pt")
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    print(f"[2/3] Exporting model graph to ONNX format at {ONNX_MODEL_PATH}...")
    
    # Export model with dynamic batch size and dynamic sequence length
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        ONNX_MODEL_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
        do_constant_folding=True
    )
    print("ONNX model exported successfully!")

    # 3. Benchmark PyTorch vs ONNX Runtime Inference Latency
    print("[3/3] Running latency benchmark (PyTorch vs ONNX Runtime)...")
    
    sample_prompt = "User key is sk-live-abc123456789 and support email is support@enterprise.io"
    encoded = tokenizer(sample_prompt, return_tensors="np", truncation=True, max_length=128)
    
    pt_inputs = {
        "input_ids": torch.tensor(encoded["input_ids"]),
        "attention_mask": torch.tensor(encoded["attention_mask"])
    }

    # PyTorch Inference Latency
    start_pt = time.perf_counter()
    with torch.no_grad():
        pt_outputs = model(**pt_inputs).logits.numpy()
    pt_latency = (time.perf_counter() - start_pt) * 1000

    # ONNX Runtime Inference Latency
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }
    
    start_ort = time.perf_counter()
    ort_outputs = ort_session.run(None, ort_inputs)[0]
    ort_latency = (time.perf_counter() - start_ort) * 1000

    # Verify output consistency
    np.testing.assert_allclose(pt_outputs, ort_outputs, rtol=1e-3, atol=1e-4)
    
    print("\n--- Latency Benchmark Results ---")
    print(f"PyTorch CPU Execution Latency : {pt_latency:.2f} ms")
    print(f"ONNX Runtime Execution Latency: {ort_latency:.2f} ms")
    print(f"Latency Reduction             : {((pt_latency - ort_latency) / pt_latency) * 100:.1f}%\n")

if __name__ == "__main__":
    export_pytorch_to_onnx()