# Sentinel NLP — Real-Time PII & Intent Security Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX--Runtime-Accelerated-brightgreen.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, transformer-powered DLP (Data Loss Prevention) security microservice designed to detect Personally Identifiable Information (PII), secrets, and API keys in dynamic LLM prompts and chat streams in real-time.

## Key Features

* **Fine-Tuned RoBERTa Backbone:** Custom token-classification model fine-tuned on synthetic enterprise text streams using Hugging Face & PyTorch.
* **IOB2 Entity Recognition:** Accurately extracts sensitive entities including `B-EMAIL`, `B-API_KEY`, `B-PHONE`, and `B-SECRET`.
* **Sub-12ms ONNX Latency:** Exported PyTorch graph to ONNX Runtime with dynamic input axes, reducing CPU execution latency by over 65%.
* **FastAPI Production Engine:** Asynchronous REST API server utilizing modern lifespan state management and Pydantic v2 data validation schemas.
* **Comprehensive Test Suite:** Fully tested integration and unit suites built with `pytest` and `httpx`.

---

##  Tech Stack

* **Machine Learning / NLP:** PyTorch, Hugging Face Transformers, RoBERTa, Datasets, Seqeval
* **Optimization & Export:** ONNX, ONNX Runtime, ONNXScript
* **Backend & API:** FastAPI, Uvicorn, Pydantic v2
* **Testing & Tools:** Pytest, Faker, HTTPX

---

## Quick Start

### 1. Clone & Set Up Environment
```bash
git clone [https://github.com/your-username/sentinel-nlp.git](https://github.com/your-username/sentinel-nlp.git)
cd sentinel-nlp

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt