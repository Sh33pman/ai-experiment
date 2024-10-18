from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a super car my whole life, and I still can't afford one.", "So have I!"]

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(sequences, padding=True, return_tensors="pt", truncation=True)

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
# model_inputs = tokenizer(sequences, padding="max_length")

# Will pad the sequences up to the specified max length
# model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

output = model(**model_inputs)
print(output)