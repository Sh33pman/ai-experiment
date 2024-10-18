from transformers import AutoTokenizer, AutoModel
import torch

checkpoint = "bert-base-cased"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

model.save_pretrained("my-" + checkpoint)
tokenizer.save_pretrained("my-" + checkpoint)

raw_input = "Using a Transformer network is simple"

# Long way
tokens = tokenizer.tokenize(raw_input)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
final_inputs = tokenizer.prepare_for_model(ids)
print(final_inputs)
print(tokenizer.decode(final_inputs.input_ids))

# Short way
inputs = tokenizer(raw_input)
print(inputs)

model_inputs = torch.tensor([final_inputs.input_ids])
print(model_inputs)

output = model(model_inputs)

print(output)