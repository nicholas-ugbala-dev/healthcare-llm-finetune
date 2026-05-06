# Healthcare LLM Fine-Tuning

Fine-tuning Llama 3.2 3B on a medical QA dataset using LoRA/PEFT 
for healthcare question answering.

## Stack
- Base model: meta-llama/Llama-3.2-3B-Instruct
- Dataset: MedQuAD (NIH-sourced medical QA)
- Fine-tuning: LoRA via PEFT + BitsAndBytes
- Training runtime: Google Colab (T4 GPU)
- Deployment: FastAPI + Docker + Hugging Face Hub

## Status
🟡 Week 1 — Environment setup & baseline inference