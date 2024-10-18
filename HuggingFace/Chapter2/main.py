import torch.nn.functional
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.logits.shape)

print(outputs.logits)

predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
print(predictions)

print(model.config.id2label)
