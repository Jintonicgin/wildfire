import os
import torch
from transformers import pipeline

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# (가벼운) 텍스트 생성 / 대화
GEN_MODEL = os.getenv("HF_GEN_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
gen_pipe = pipeline(
    task="text-generation",
    model=GEN_MODEL,
    device=0 if DEVICE in ["cuda", "mps"] else -1,
    torch_dtype=torch.float16 if DEVICE in ["cuda", "mps"] else None,
    max_new_tokens=int(os.getenv("GEN_MAX_NEW_TOKENS", "512")),
    do_sample=True,
    temperature=float(os.getenv("GEN_TEMPERATURE", "0.7")),
    top_p=float(os.getenv("GEN_TOP_P", "0.9"))
)

# 번역(영↔한)
TRANS_EN2KO = os.getenv("HF_TRANS_EN2KO", "Helsinki-NLP/opus-mt-en-ko")
TRANS_KO2EN = os.getenv("HF_TRANS_KO2EN", "Helsinki-NLP/opus-mt-ko-en")
trans_en2ko = pipeline("translation", model=TRANS_EN2KO, device=0 if DEVICE in ["cuda", "mps"] else -1)
trans_ko2en = pipeline("translation", model=TRANS_KO2EN, device=0 if DEVICE in ["cuda", "mps"] else -1)

# 요약(경량)
SUM_MODEL = os.getenv("HF_SUM_MODEL", "sshleifer/distilbart-cnn-12-6")
sum_pipe = pipeline(
    task="summarization",
    model=SUM_MODEL,
    device=0 if DEVICE in ["cuda", "mps"] else -1
)