# Results

Model outputs and training logs across the project.

## baseline_outputs.txt
Base Llama 3.2 3B Instruct outputs on 5 test questions, before any fine-tuning.
Generated with sampling. Notable for a factual hallucination on the diabetes question,
where the base model attributed increased thirst to insulin causing water retention
rather than to high blood glucose.

## finetuned_outputs_run1.txt
Week 3 model. One epoch on cleaned ChatDoctor data. The diabetes hallucination is
fixed and the filler openers are removed. Some residual factual errors and trailing
filler remain. Generated with sampling.

## finetuned_outputs_run2.txt
Week 4 two-epoch model, trained on the combined dataset (8,000 ChatDoctor + 4,000
WikiDoc). This run achieved the best eval loss of the project (2.275) but the worst
generation quality. Under decoding it degenerated into repetition loops and
confabulation, inventing drug names and failing to stop on enumeration questions.
Generated with sampling. Kept as evidence of the central finding: a lower loss number
did not mean a better model.

## finetuned_v2_outputs.txt
Final model. One epoch on the rebalanced dataset (8,500 ChatDoctor + 1,500 WikiDoc),
with LoRA applied to both attention and feed-forward layers. Generated with greedy
decoding (do_sample=False) plus repetition_penalty=1.3 and an explicit EOS token, which
makes the outputs reproducible: running the same question twice produces identical text.
All five answers are accurate, bounded, and stable, including the hypertension and heart
attack questions that degenerated in run2.

## A note on what these files show together

The progression from run1 to run2 to v2 is the story of the project. run1 was a decent
one-epoch model. run2 had a better loss but worse output, because more training plus a
list-heavy dataset overfit the model into runaway enumeration. v2 fixed it by rebalancing
the data, targeting the feed-forward layers, and switching to greedy decoding with a
proper stop token. The lesson: eval loss measures next-token prediction, not factual
accuracy or whether the model knows when to stop, and generation settings matter as much
as the trained weights.