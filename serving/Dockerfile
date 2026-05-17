FROM python:3.10-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Torch CPU-only dulu (jauh lebih kecil ~200MB vs ~2GB CUDA)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Sisanya dari requirements
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Copy serving code
COPY inference.py rag.py llm_client.py banking_faq.json ./
COPY static/ static/

# Download model dari HF Hub saat build
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
os.makedirs('artifacts', exist_ok=True); \
hf_hub_download('ninditya/banking-router-model', 'model.pkl', local_dir='artifacts'); \
hf_hub_download('ninditya/banking-router-model', 'encoder_name.txt', local_dir='artifacts'); \
print('Model downloaded successfully') \
"

# HuggingFace Spaces wajib port 7860
EXPOSE 7860

ENV MODEL_DIR=artifacts
ENV LLM_PROVIDER=mock
ENV PORT=7860

CMD ["python", "inference.py"]
