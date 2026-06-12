# Results

## baseline_outputs.txt
Base Llama 3.2 3B outputs on 5 test questions, before fine-tuning.

## finetuned_outputs_run1.txt
Fine-tuned model outputs, run 1.

## finetuned_outputs_run2.txt
Fine-tuned model outputs, run 2, same weights.

Note: both fine-tuned runs used sampling (do_sample=True, temperature=0.7).
The same model produced different outputs across runs, and run 2 shows a
repetition collapse on the heart attack question. This is a known failure
mode of lightly fine-tuned small models under stochastic decoding. Week 4
addresses it with greedy decoding for evaluation and a repetition penalty
for inference.