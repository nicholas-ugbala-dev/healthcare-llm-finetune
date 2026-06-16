# Healthcare LLM Fine-Tuning

Fine-tuning Llama 3.2 3B on a medical question-answering dataset using LoRA/PEFT.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?style=flat&logo=huggingface&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Status-Week%204%20Complete-brightgreen?style=flat)

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

That mechanism is wrong. Increased thirst is caused by high blood glucose pulling fluid from tissues through osmosis, not insulin.

**Fine-tuned model (final), same question:**

> "Increased thirst (polydipsia), frequent urination, fatigue, blurred vision, slow healing wounds. High blood sugar levels cause damage to small vessels in the body, leading to slow recovery."

The hallucination is gone, the filler preamble is gone, and the answer is bounded and accurate.

---

## Technical Approach

| Decision | Choice | Reasoning |
|---|---|---|
| Base model | Llama 3.2 3B Instruct | Capable enough for meaningful baselines, small enough for free-tier GPU, richest fine-tuning ecosystem |
| Fine-tuning method | LoRA via PEFT (r=16, alpha=32) | Trains a small fraction of parameters, feasible on T4 VRAM |
| LoRA targets | Attention + feed-forward layers | Feed-forward targeting improved factual recall and reduced confabulation |
| Quantization | 4-bit NF4 via BitsAndBytes | Reduces model footprint from ~12GB to ~2GB |
| Dataset | ChatDoctor + WikiDoc (cleaned, rebalanced) | Conversational style plus factual grounding |
| Training compute | Kaggle T4 GPU (15.6GB VRAM) | Free, 12-hour session limit |
| Decoding | Greedy + EOS + repetition penalty | Reproducible, stable, no degeneration |
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
- Formatted into the Llama 3.2 chat template, split 90/10
- Cleaned dataset published to Hugging Face Hub

**Week 3 (complete)**
- Configured LoRA adapters via PEFT and SFTTrainer
- Worked through a BFloat16 / fp16 gradient scaler conflict on T4; migrated Colab to Kaggle
- Trained 1 epoch, eval loss 2.558 to 2.495
- Baseline hallucination fixed; residual errors and sampling instability documented

**Week 4 (complete)**
- Combined two datasets (ChatDoctor + WikiDoc) for conversational style and factual grounding
- Switched to Llama's built-in pad token, shrinking the adapter from 3.19GB to ~50MB
- Two-epoch run improved eval loss to 2.275 but regressed generation into list confabulation
- Diagnosed the failure: list-format overfitting plus a repetition penalty that drove degeneration
- Fixed it with data rebalancing, expanded LoRA targets (feed-forward layers), and corrected greedy decoding
- Final model produces accurate, bounded, reproducible answers across all 5 test questions

**Week 5 (planned)**
- FastAPI inference endpoint
- Docker containerisation
- Public deployment

**Week 6 (planned)**
- Final blog series, interview summary, repo polish

---

## Key Finding From Week 4

Lower eval loss did not mean a better model. The two-epoch run had the best loss number of the project (2.275) and the worst generation quality, collapsing into invented drug names and repetition loops. Loss measures next-token prediction, not factual accuracy or whether the model knows when to stop. The fix was a combination of data rebalancing, targeting the feed-forward layers with LoRA, and switching to greedy decoding with a proper stop token. The full writeup is in the Week 4 article.

---

## Artifacts

- **Final model:** [nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned-v2](https://huggingface.co/nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned-v2)
- **Final dataset:** [nicholas-ugbala-hf/medical-qa-narrative-10k](https://huggingface.co/datasets/nicholas-ugbala-hf/medical-qa-narrative-10k)

---

## Repository Structure

```
healthcare-llm-finetune/
├── notebooks/
│   ├── week1_baseline.ipynb
│   ├── week2_data_prep.ipynb
│   ├── week3_finetuning.ipynb
│   └── week4_finetuning.ipynb
├── results/
│   ├── baseline_outputs.txt           # Pre-training outputs
│   ├── finetuned_outputs.txt          # Week 3 outputs
│   ├── finetuned_v2_outputs.txt       # Week 4 final outputs (greedy, reproducible)
│   ├── training_log.json              # Week 4 loss history
│   └── README.md
├── data/
│   └── README.md
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
pip install "transformers>=4.44,<5.0" bitsandbytes accelerate peft trl datasets huggingface_hub
```

Note: transformers is pinned below 5.0. The 5.x release changed import paths that the fine-tuning stack has not fully caught up to.

**Load the final fine-tuned model**

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
tokenizer = AutoTokenizer.from_pretrained("nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned-v2")
model = PeftModel.from_pretrained(base, "nicholas-ugbala-hf/llama-3.2-3b-medical-finetuned-v2")
model.eval()
```

**Recommended generation settings** (these prevent the degeneration documented in Week 4):

```python
outputs = model.generate(
    input_ids=encoded_inputs["input_ids"],
    attention_mask=encoded_inputs["attention_mask"],
    max_new_tokens=256,
    do_sample=False,
    repetition_penalty=1.3,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    pad_token_id=tokenizer.pad_token_id,
)
```

---

## Stack

- **Model:** meta-llama/Llama-3.2-3B-Instruct
- **Fine-tuning:** LoRA via PEFT (attention + feed-forward targets), TRL SFTTrainer
- **Quantization:** BitsAndBytes 4-bit NF4
- **Datasets:** lavita/ChatDoctor-HealthCareMagic-100k + medalpaca/medical_meadow_wikidoc (cleaned, rebalanced)
- **Training runtime:** Kaggle T4 GPU
- **Inference API:** FastAPI (planned, Week 5)
- **Containerisation:** Docker (planned, Week 5)
- **Model hosting:** Hugging Face Hub

---

## Writing

Each week has a corresponding article documenting the decisions and the results.

- Week 1: Setup, model loading, quantization, tokenization, baseline results [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-1-setup-and-baseline-inference-3k25)
- Week 2: Dataset selection, cleaning pipeline, formatting, split [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-2-data-preparation-393n-temp-slug-518945?preview=4eb4e6003bc460941fe8f8442251582e2a6f33ed121a757230ca4820aa3d7d4d7b8494a8c8cc861f9982d63872aa98f28710286c1798528f7d2c1f90)
- Week 3: LoRA configuration, the training run, hardware debugging, before/after results [link](https://dev.to/nicholas-ugbala-dev/fine-tuning-llama-32-3b-on-medical-qa-week-3-the-first-training-run-14pl)

---

## About

I am Nicholas Ugbala, a full-stack software engineer with production experience in healthtech and AI integration. This project is part of a deliberate skill-building track toward AI/ML engineering roles.

Follow the build: [dev.to article series link](http://dev.to/nicholas-ugbala-dev) or connect on [LinkedIn](https://www.linkedin.com/in/nicholas-ugbala/)