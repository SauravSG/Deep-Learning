# Fine-tuning BERT on QNLI with the HuggingFace Trainer

Fine-tuned `bert-base-uncased` to decide whether a sentence contains the answer to a question (GLUE QNLI).

**Model on the Hub:** https://huggingface.co/GoingDeep/bert-base-uncased-finetuned-for-qnli

## Results (QNLI validation split, 5,463 pairs)

| | Accuracy | Validation loss |
|---|---|---|
| Always predicting `not_entailment` (majority class) | 50.5% | n/a |
| Before fine-tuning (random classifier head, seed 42) | 45.4% | 0.741 |
| Epoch 1 (published checkpoint) | 87.2% | 0.312 |
| Epoch 2 | 87.4% | 0.363 |
| Epoch 3 | 87.8% | 0.554 |

- The published checkpoint is the one with the lowest validation loss (epoch 1), not the highest accuracy.
- The validation split was also used to select the checkpoint, so these scores are slightly optimistic.
- The accuracy gap between epochs is about 30 examples out of 5,463, which is within run-to-run noise.

## Task

- Input: a **question** and a **sentence**, in that order (`[CLS] question [SEP] sentence [SEP]`).
- Labels: `0` = `entailment` (the sentence answers the question), `1` = `not_entailment`. This is the reverse of MRPC, where `1` was the positive class.
- The test split has no public labels (all `-1`), so validation is the evaluation split.

## Setup

- Data: GLUE QNLI, with a random 20,000-pair subsample of the 104,743 training pairs (shuffled with seed 42, label balance checked: 10,012 vs 9,988). Evaluation uses the full validation split.
- Preprocessing: question/sentence pairs tokenized with `.map(batched=True)` and `truncation=True`; raw text columns removed; `label` renamed to `labels`; dynamic per-batch padding with `DataCollatorWithPadding`.
- Model: `AutoModelForSequenceClassification` from `bert-base-uncased`, `num_labels=2`, with `id2label`/`label2id` set so predictions show names. `set_seed(42)` was called before loading.
- Training: 3 epochs, batch size 16, learning rate 5e-5 (3,750 steps, about 21 minutes on a Colab GPU), evaluated and saved every epoch, `load_best_model_at_end=True`.
- Metric: accuracy via `evaluate.load("glue", "qnli")` (QNLI returns accuracy only).

## Experiment log

- **Hypothesis:** validation loss lowest at epoch 3, accuracy in the 70s. **Not supported on either point.**
- **Result:** validation loss was lowest at epoch 1 and rose afterwards while training loss kept falling (0.407 → 0.214 → 0.098), so the model overfit from epoch 2. Accuracy was 87%+, higher than predicted.
- **Same pattern as MRPC:** with lr 5e-5, validation loss was lowest after epoch 1 on both datasets, even though QNLI had about 5 times more training data.
- **Next ideas:** train for 1 to 2 epochs, try the full training set to see whether more data helps, and compare `metric_for_best_model='eval_accuracy'` against the default loss.

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "GoingDeep/bert-base-uncased-finetuned-for-qnli"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)

inputs = tokenizer(
    "What is the capital of France?",
    "Paris is the capital and largest city of France.",
    return_tensors="pt",
)
with torch.inference_mode():
    pred = model(**inputs).logits.argmax(dim=-1).item()
