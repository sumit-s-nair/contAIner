import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.rl_env.env import PlannerEnv, EnvConfig
from src.system2_planner.models import RepoManifest

print('Loading model and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-1.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-Coder-1.5B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)

print('Initializing env...')
config = EnvConfig(dry_run=True, corpus_manifest_path='cache/corpus_manifest.json')
env = PlannerEnv(config)
corpus = env._load_corpus()

repo_entry = next(r for r in corpus if r['repo'] == 'deanishe/alfred-appscripts')
print(f'Found repo: {repo_entry}')

obs, ep_info = env.reset(repo_entry=repo_entry)
print('Reset successful.')

chat_messages = env._serializer.to_chat_prompt(obs, tokenizer=tokenizer)
prompt_str = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
print(f'Prompt length (chars): {len(prompt_str)}')

inputs = tokenizer(prompt_str, return_tensors='pt').to(model.device)
input_ids = inputs['input_ids']

print(f'Shape: {input_ids.shape}')
print(f'Min token ID: {input_ids.min().item()}')
print(f'Max token ID: {input_ids.max().item()}')
print(f'Vocab size: {tokenizer.vocab_size}')

print('Running generation...')
try:
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        pad_token_id=tokenizer.eos_token_id
    )
    print('Generation successful!')
except Exception as e:
    print(f'Generation failed: {e}')

