import logging

logger = logging.getLogger(__name__)

# NOTE: This module is dead code — it is not imported anywhere.
# Previously it eagerly loaded a DistilBERT model at module level,
# slowing down Django startup by 5-10 seconds. The body is now
# guarded so it only runs when executed directly.

if __name__ == "__main__":
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer("Hello, I had a great day", return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    logger.info(model.config.id2label[predicted_class_id])
