from mamba_lm import from_pretrained
from transformers import AutoTokenizer

model = from_pretrained('state-spaces/mamba-130m').to("cuda")
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

# output = model.generate(tokenizer, "There once was a")
output = model.generate(tokenizer, "There", sample=None)

print(output)