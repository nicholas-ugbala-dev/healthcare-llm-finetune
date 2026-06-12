# Healthcare LLM Fine-Tuning

Fine-tuning Llama 3.2 3B on a medical question-answering dataset using LoRA/PEFT.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Week%203%20Complete-brightgreen?style=flat)

---

## What This Is

This is a portfolio project. I am a backend and full-stack engineer with production experience in healthtech, and I am building toward AI/ML engineering roles. Fine-tuning open-source LLMs is the specific skill gap I am closing.

The project takes Llama 3.2 3B Instruct, fine-tunes it on a cleaned medical QA dataset, deploys it as a public inference API, and documents every decision along the way. The domain is healthcare because it is the domain I work in. The pipeline itself is domain-agnostic.

This is not a production medical AI. It is a complete, end-to-end fine-tuning pipeline built in public.

---

## The Result So Far

The clearest measure of progress is a single question asked before and after fine-tuning.

**Base model (Week 1), asked about early symptoms of type 2 diabetes:**

> "When your body produces more insulin, it can cause your body to hold onto more water, leading to increased thirst."

That mechanism is wrong. Increased thirst is caused by high blood glucose pulling fluid from tissues, not insulin.

**Fine-tuned model (Week 3), same question:**

> "Increased thirst and urination: High blood sugar levels can cause the body to produce more urine, leading to dehydration and increased thirst."

The hallucination is gone, and the filler preamble the base model opened every answer with ("As a medical assistant, I'd be happy to help...") is gone across all test responses.

---

## Technical Approach

| Decision | Choice | Reasoning |
|---|---|---|
| Base model | Llama 3.2 3B Instruct | Capable enough for meaningful baselines, small enough for free-tier GPU |
| Fine-tuning method | LoRA via PEFT (r=16, alpha=32) | Trains 0.23% of parameters, feasible on T4 VRAM |
| Quantization | 4-bit NF4 via BitsAndBytes | Reduces model footprint from ~12GB to ~2GB |
| Dataset | ChatDoctor HealthCareMagic 100k | Conversational prose format, matches target output style |
| Training compute | Kaggle T4 GPU (15.6GB VRAM) | Free, 12-hour session limit |
| Deployment | FastAPI + Docker + Hugging Face Hub | Reproducible, public, callable |

---

## Progress

**Week 1 (complete)**
- Environment configured, Llama 3.2 3B loaded with 4-bit quantization
- Baseline inference across 5 medical test questions
- Identified a factual hallucination in the baseline (diabetes symptom mechanism)

**Week 2 (complete)**
- Switched from a multiple-choice dataset to ChatDoctor conversational prose
- Built a cleaning pipeline: 112,165 raw rows filtered to 45,205 clean samples
- Sampled 10,000, formatted into the Llama 3.2 chat template, split 90/10
- Cleaned dataset published to Hugging Face Hub

**Week 3 (complete)**
- Configured LoRA adapters via PEFT and SFTTrainer training loop
- Worked through a BFloat16 / fp16 gradient scaler conflict on T4; migrated Colab to Kaggle
- Trained 1 epoch over 4,937 samples, 309 steps, ~73 minutes
- Eval loss declined 2.558 to 2.495 with no overfitting
- Fine-tuned adapter weights published to Hugging Face Hub
- Baseline hallucination fixed; residual errors honestly documented

**Week 4 (in progress)**
- Switch to Llama's built-in pad token to shrink the adapter file
- Tighter data cleaning, second dataset (WikiDoc) for factual grounding
- Full dataset, 2 epochs, reproducible training seed
- Quantitative evaluation: base vs fine-tuned perplexity, greedy-decoded comparison

**Weeks 5 to 6 (planned)**
- Week 5: FastAPI inference endpoint, Docker containerisation, public deployment
- Week 6: Final blog series, interview summary, repo polish

---

## Artifacts

- **Fine-tuned model:** [nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned](https://huggingface.co/nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned)
- **Cleaned dataset:** [nicholas-ugbala-hf/chatdoctor-cleaned-10k](https://huggingface.co/datasets/nicholas-ugbala-hf/chatdoctor-cleaned-10k)

---

## Repository Structure

```
healthcare-llm-finetune/
├── notebooks/
│   ├── week1_baseline.ipynb         # Model loading, tokenization, baseline inference
│   ├── week2_data_prep.ipynb        # Dataset cleaning, formatting, train/eval split
│   └── week3_finetuning.ipynb       # LoRA config, training, baseline comparison
├── results/
│   ├── baseline_outputs.txt         # Pre-training outputs on 5 test questions
│   └── finetuned_outputs.txt        # Post-training outputs on the same 5 questions
├── data/
│   └── README.md                    # Dataset documentation
└── README.md
```

---

## Setup and Reproduction

**Requirements**

- Python 3.11+
- A Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct` (gated)
- A T4 GPU (Kaggle free tier or Colab)

**Install**

```bash
pip install transformers bitsandbytes accelerate peft trl datasets huggingface_hub
```

**Load the fine-tuned model**

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained("nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned")
model = PeftModel.from_pretrained(base, "nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned")
model.eval()
```

---

## Stack

- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning:** LoRA via PEFT, supervised fine-tuning via TRL SFTTrainer
- **Quantization:** BitsAndBytes 4-bit NF4
- **Dataset:** lavita/ChatDoctor-HealthCareMagic-100k (cleaned subset)
- **Training runtime:** Kaggle T4 GPU
- **Inference API:** FastAPI (planned, Week 5)
- **Containerisation:** Docker (planned, Week 5)
- **Model hosting:** Hugging Face Hub

---

## Writing

Each week has a corresponding article documenting the decisions made and the results.

- Week 1: Setup, model loading, quantization, tokenization, baseline results [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-1-setup-and-baseline-inference-3k25)
- Week 2: Dataset selection, cleaning pipeline, formatting, train/eval split [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-2-data-preparation-393n-temp-slug-518945?preview=4eb4e6003bc460941fe8f8442251582e2a6f33ed121a757230ca4820aa3d7d4d7b8494a8c8cc861f9982d63872aa98f28710286c1798528f7d2c1f90)
- Week 3: LoRA configuration, the training run, hardware debugging, before/after results [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-3-the-first-training-run-14pl)

---

## About

I am Nicholas Ugbala, a full-stack software engineer with production experience in healthtech and AI integration. This project is part of a deliberate skill-building track toward AI/ML engineering roles.

Follow the build: [dev.to article series link](http://dev.to/nicholas-ugbala-dev) or connect on [LinkedIn](https://www.linkedin.com/in/nicholas-ugbala/)