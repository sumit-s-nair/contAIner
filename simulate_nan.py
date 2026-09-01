import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

# Simulate NaN weights from exploding gradients
print('Corrupting model weights with NaN...')
for param in model.parameters():
    param.data.fill_(float('nan'))

inputs = tokenizer('Hello world', return_tensors='pt').to(model.device)

print('Running generation with sampling...')
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    print('Generation successful!')
except Exception as e:
    print(f'Generation failed with error: {type(e).__name__}: {e}')
