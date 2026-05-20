# Healthcare LLM Fine-Tuning

Fine-tuning Llama 3.2 3B on a medical question-answering dataset using LoRA/PEFT.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Week%201%20Complete-brightgreen?style=flat)

---

## What This Is

This is a portfolio project. I am a backend and full-stack engineer with production experience in healthtech, and I am building toward AI/ML engineering roles. Fine-tuning open-source LLMs is the specific skill gap I am closing.

The project takes Llama 3.2 3B Instruct, fine-tunes it on MedQuAD (a NIH-sourced medical QA dataset), deploys it as a public inference API, and documents every decision made along the way. The domain is healthcare because it is the domain I work in. The pipeline itself is domain-agnostic.

This is not a production medical AI. It is a complete, end-to-end fine-tuning pipeline built in public.

---

## The Problem

General-purpose LLMs produce medically plausible text that is not always clinically accurate. Asked about the early symptoms of type 2 diabetes, a base Llama 3.2 3B model responded:

> "When your body produces more insulin, it can cause your body to hold onto more water, leading to increased thirst."

That mechanism is wrong. Increased thirst in diabetes is caused by high blood glucose pulling fluid from tissues through osmosis, not insulin. The model arrived there through superficial pattern matching on co-occurring keywords in general web text.

Fine-tuning on curated, authoritative medical data penalizes these correlations and anchors responses to verified clinical pathways. That is what this project tests and demonstrates.

---

## Technical Approach

| Decision | Choice | Reasoning |
|---|---|---|
| Base model | Llama 3.2 3B Instruct | Capable enough for meaningful baselines, small enough for free-tier GPU |
| Fine-tuning method | LoRA via PEFT | Trains 1-5% of parameters, feasible on T4 VRAM |
| Quantization | 4-bit NF4 via BitsAndBytes | Reduces model footprint from ~12GB to ~2GB |
| Dataset | MedQuAD (lavita/medical-qa-datasets) | NIH-sourced, clean instruction pairs, defensible provenance |
| Training compute | Google Colab T4 GPU (15.8GB VRAM) | Free tier, right size for this scale |
| Deployment | FastAPI + Docker + Hugging Face Hub | Reproducible, public, callable |

---

## Progress

**Week 1 (complete)**
- Development environment configured on Colab
- Llama 3.2 3B Instruct loaded with 4-bit quantization
- Baseline inference confirmed working across 5 medical test questions
- Baseline outputs saved to `results/baseline_outputs.txt` for post-training comparison
- Identified one factual hallucination in baseline (diabetes symptom mechanism)

**Week 2 (in progress)**
- Load and inspect MedQuAD dataset
- Format into instruction-following template
- Set up train and eval splits
- Establish evaluation baseline (ROUGE scores on eval set)

**Weeks 3 to 6 (planned)**
- Week 3: First LoRA fine-tuning run, compare against baseline
- Week 4: Larger run with tuned hyperparameters, track failure modes
- Week 5: Push model to Hugging Face Hub, build FastAPI inference endpoint, containerise with Docker
- Week 6: Write README, blog post, and interview summary. Polish repo for portfolio use.

---

## Repository Structure

```
healthcare-llm-finetune/
├── notebooks/
│   └── week1_baseline.ipynb       # Model loading, tokenization, baseline inference
├── results/
│   └── baseline_outputs.txt       # Pre-training model outputs on 5 test questions
├── data/
│   └── prep.py                    # Dataset formatting scripts (Week 2)
├── src/
│   └── api/                       # FastAPI inference endpoint (Week 5)
└── README.md
```

---

## Setup and Reproduction

**Requirements**

- Python 3.11+
- A Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct` (gated model, request access at huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- Google Colab with T4 GPU runtime, or a local GPU with at least 8GB VRAM

**Install dependencies**

```bash
pip install transformers bitsandbytes accelerate peft trl datasets huggingface_hub
```

**Authenticate with Hugging Face**

```python
from huggingface_hub import login
login()  # paste your HF token when prompted
```

**Run the baseline notebook**

Open `notebooks/week1_baseline.ipynb` in Colab. Set runtime to T4 GPU. Run all cells in order.

The notebook loads the model, runs inference on 5 test questions, and saves outputs to `results/baseline_outputs.txt`.

---

## Stack

- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning:** LoRA via PEFT, supervised fine-tuning via TRL SFTTrainer
- **Quantization:** BitsAndBytes 4-bit NF4
- **Dataset:** lavita/medical-qa-datasets (medical_meadow_medqa subset)
- **Training runtime:** Google Colab T4 GPU
- **Inference API:** FastAPI (planned, Week 5)
- **Containerisation:** Docker (planned, Week 5)
- **Model hosting:** Hugging Face Hub (planned, Week 5)

---

## Writing

Technical writeup of Week 1 (setup, model loading, quantization, tokenization, baseline results) is published on dev.to:

[Fine-Tuning Llama 3.2 3B on Medical QA: Week 1 Setup and Baseline Inference](https://dev.to/sarko07/fine-tuning-llama-32-3b-on-medical-qa-week-1-setup-and-baseline-inference-3k25)

Each week will have a corresponding article documenting the decisions made and what the results showed.

---

## About

I am Nicholas Ugbala, a full-stack software engineer with production experience in healthtech and AI integration. This project is part of a deliberate skill-building track toward AI/ML engineering roles.

If you want to follow the build: [dev.to article series link](https://dev.to/sarko07/fine-tuning-llama-32-3b-on-medical-qa-week-1-setup-and-baseline-inference-3k25) or connect on LinkedIn: [Nicholas Ugbala](https://www.linkedin.com/in/nicholas-ugbala/)