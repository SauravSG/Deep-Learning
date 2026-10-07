# Week 7: Fine-tuning BERT on MRPC with the HuggingFace Trainer

Fine-tuned `bert-base-uncased` to detect whether two sentences are paraphrases (GLUE MRPC).

**Model on the Hub:** https://huggingface.co/GoingDeep/bert-base-uncased-finetuned-mrpc

## Results (MRPC validation split)

- Before fine-tuning (random classifier head, seed 42): 68.4% accuracy, F1 0.812
  - This equals always predicting "equivalent", so it is the majority-class floor.
- After fine-tuning (epoch 1 checkpoint, picked by lowest validation loss): 85.5% accuracy, F1 0.897
- Scores are on the validation split, which was also used to pick the best checkpoint, so they are slightly optimistic.

## Setup

- Data: GLUE MRPC (3,668 train / 408 validation sentence pairs)
- Preprocessing: tokenized sentence pairs with `.map(batched=True)`; dynamic per-batch padding with `DataCollatorWithPadding`
- Model: `AutoModelForSequenceClassification`, `num_labels=2`
- Training: 3 epochs, batch size 16, learning rate 5e-5, evaluate and save every epoch, `load_best_model_at_end=True`
- Metrics: accuracy and F1 via `evaluate.load("glue", "mrpc")`

## Experiments

- Baseline run: validation loss rose every epoch (0.35 → 0.44 → 0.56) while accuracy stayed flat, a sign of overfitting. The epoch 1 checkpoint was kept.
- Learning rate 2e-5 (only change): overfitting slowed, but the loaded checkpoint (epoch 2) scored lower (82.4% / 0.878). Hypothesis not supported; one run, no seed control.
- One earlier run collapsed to predicting a single class (68.4% accuracy, frozen across epochs). Most likely a setup or learning-rate problem, not confirmed.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

repo = "GoingDeep/bert-base-uncased-finetuned-mrpc"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo)

inputs = tokenizer("This cup of coffee was quite tasty!", "The coffee in this cup tasted pretty good!", return_tensors="pt")
with torch.inference_mode():
    pred = torch.argmax(model(**inputs).logits, dim=-1).item()
print({0: "not_equivalent", 1: "equivalent"}[pred])
```

## Limitations

- English only; trained on a small set of news-style sentence pairs, so it may be unreliable on casual or domain-specific text.
- MRPC "equivalent" means near-identical information, not just a similar topic.
