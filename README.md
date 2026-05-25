# Healthcare LLM Fine-Tuning

Fine-tuning Llama 3.2 3B on a medical question-answering dataset using LoRA/PEFT.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Week%202%20Complete-brightgreen?style=flat)

---

## What This Is

This is a portfolio project. I am a backend and full-stack engineer with production experience in healthtech, and I am building toward AI/ML engineering roles. Fine-tuning open-source LLMs is the specific skill gap I am closing.

The project takes Llama 3.2 3B Instruct, fine-tunes it on a cleaned medical QA dataset, deploys it as a public inference API, and documents every decision along the way. The domain is healthcare because it is the domain I work in. The pipeline itself is domain-agnostic.

This is not a production medical AI. It is a complete, end-to-end fine-tuning pipeline built in public.

---

## The Problem

General-purpose LLMs produce medically plausible text that is not always clinically accurate. Asked about the early symptoms of type 2 diabetes, a base Llama 3.2 3B model responded:

> "When your body produces more insulin, it can cause your body to hold onto more water, leading to increased thirst."

That mechanism is wrong. Increased thirst in diabetes is caused by high blood glucose pulling fluid from tissues through osmosis, not insulin. The model arrived there through superficial pattern matching on co-occurring keywords in general web text.

Fine-tuning on curated medical data penalizes these correlations and anchors responses to verified clinical pathways. That is what this project tests and demonstrates.

---

## Technical Approach

| Decision | Choice | Reasoning |
|---|---|---|
| Base model | Llama 3.2 3B Instruct | Capable enough for meaningful baselines, small enough for free-tier GPU |
| Fine-tuning method | LoRA via PEFT | Trains 1-5% of parameters, feasible on T4 VRAM |
| Quantization | 4-bit NF4 via BitsAndBytes | Reduces model footprint from ~12GB to ~2GB |
| Dataset | ChatDoctor HealthCareMagic 100k | Conversational prose format, matches target output style, real engineering cleaning problem |
| Training compute | Google Colab T4 GPU (15.8GB VRAM) | Cloud-hosted GPU, right size for this scale |
| Deployment | FastAPI + Docker + Hugging Face Hub | Reproducible, public, callable |

---

## Dataset Decision

The original plan used `lavita/medical-qa-datasets` with the `medical_meadow_medqa` subset. Inspecting the samples revealed it was the wrong shape: outputs were USMLE multiple choice answer selections, not clinical prose. Training on it would produce a model that selects answer letters, not one that answers patient questions.

**ChatDoctor HealthCareMagic 100k** was chosen for three specific reasons. First, the output format matches the goal: conversational prose responses to patient-described symptoms. PubMedQA produces yes/no research answers. MedQA is multiple choice. Neither matches the target output style. Second, it is publicly available, ungated, and immediately loadable. Augmented chain-of-thought versions of MedQA do not exist as clean public datasets and would require GPT-4 generation to create, introducing a proprietary dependency. Third, the cleaning problem is real and representative: filtering 112k noisy forum rows to 45k usable samples is closer to production data engineering than loading a pre-sanitised benchmark.

---

## Progress

**Week 1 (complete)**
- Development environment configured on Colab
- Llama 3.2 3B Instruct loaded with 4-bit quantization
- Baseline inference confirmed working across 5 medical test questions
- Baseline outputs saved to `results/baseline_outputs.txt` for post-training comparison
- Identified one factual hallucination in baseline (diabetes symptom mechanism)

**Week 2 (complete)**
- Identified and switched from wrong dataset (USMLE multiple choice) to correct dataset (ChatDoctor conversational prose)
- Built cleaning pipeline: 112,165 raw rows filtered to 45,205 clean samples
- Removed platform filler from output openings and closings
- Removed platform name artifacts from inputs
- Sampled 10,000 rows with seed=42 for reproducibility
- Formatted into Llama 3.2 chat template
- Token length distribution confirmed: average 261 tokens, max 794, zero samples over 1024
- max_seq_length set to 512 (only 1.1% of samples exceed it)
- Train/eval split: 9,000 train, 1,000 eval
- Cleaned dataset published to Hugging Face Hub

**Week 3 (planned)**
- Configure LoRA adapters via PEFT
- Set up SFTTrainer training loop
- First fine-tuning run on T4
- Compare outputs against Week 1 baseline

**Weeks 4 to 6 (planned)**
- Week 4: Larger run with tuned hyperparameters, add second dataset, track failure modes
- Week 5: Push model to Hugging Face Hub, build FastAPI inference endpoint, containerise with Docker
- Week 6: Final README, blog post series, interview summary

---

## Repository Structure

```
healthcare-llm-finetune/
├── notebooks/
│   ├── week1_baseline.ipynb         # Model loading, tokenization, baseline inference
│   └── week2_data_prep.ipynb        # Dataset cleaning, formatting, train/eval split
├── results/
│   └── baseline_outputs.txt         # Pre-training model outputs on 5 test questions
├── data/
│   └── README.md                    # Dataset documentation
├── src/
│   └── api/                         # FastAPI inference endpoint (Week 5)
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
login()
```

**Run Week 1 baseline notebook**

Open `notebooks/week1_baseline.ipynb` in Colab. Set runtime to T4 GPU. Run all cells in order.

**Run Week 2 data preparation notebook**

Open `notebooks/week2_data_prep.ipynb` in Colab. The cleaned dataset is also available directly on Hugging Face Hub and can be loaded without re-running the cleaning pipeline:

```python
from datasets import load_dataset

dataset  = load_dataset("nicholas-ugbala-hf/chatdoctor-cleaned-10k")
train_dataset = dataset['train']
eval_dataset  = dataset['eval']
```

---

## Stack

- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning:** LoRA via PEFT, supervised fine-tuning via TRL SFTTrainer
- **Quantization:** BitsAndBytes 4-bit NF4
- **Dataset:** lavita/ChatDoctor-HealthCareMagic-100k (cleaned, 10k subset)
- **Cleaned dataset:** [your-username/chatdoctor-cleaned-10k on Hugging Face]
- **Training runtime:** Google Colab T4 GPU
- **Inference API:** FastAPI (planned, Week 5)
- **Containerisation:** Docker (planned, Week 5)
- **Model hosting:** Hugging Face Hub (planned, Week 5)

---

## Writing

Each week has a corresponding article documenting the decisions made and what the results showed.

- Week 1: Setup, model loading, quantization, tokenization, baseline results [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-1-setup-and-baseline-inference-3k25)
- Week 2: Dataset selection, cleaning pipeline, formatting, train/eval split [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-2-data-preparation-393n-temp-slug-518945?preview=4eb4e6003bc460941fe8f8442251582e2a6f33ed121a757230ca4820aa3d7d4d7b8494a8c8cc861f9982d63872aa98f28710286c1798528f7d2c1f90)

---

## About

I am Nicholas Ugbala, a full-stack software engineer with production experience in healthtech and AI integration. This project is part of a deliberate skill-building track toward AI/ML engineering roles.

Follow the build: [dev.to article series link](http://dev.to/nicholas-ugbala-dev) or connect on [LinkedIn](https://www.linkedin.com/in/nicholas-ugbala/)