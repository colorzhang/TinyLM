"""
Sample from a trained model
"""
from contextlib import nullcontext
import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling, GenerationConfig
from model.configuration_mistral import MistralConfig
from model.modeling_mistral import MistralForCausalLM

num_samples = 10 # number of samples to draw
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = MistralForCausalLM.from_pretrained("output-5-6epochs")
model.to(device)

model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# tokenizer.pad_token = tokenizer.eos_token

gen_config=GenerationConfig(
	temperature=0.8,
	top_k=20,
	top_p=0.5,
	do_sample=True,
	num_beams=1,
	repetition_penalty=1.1,
	max_new_tokens=400,
	eos_token_id=tokenizer.eos_token_id,
	pad_token_id=tokenizer.pad_token_id,
	)

# prompt = "李白乘舟将欲行，忽闻岸上踏歌声。"
# prompt = '英文名称'
# prompt = "床前明月光，疑是地上霜"
# prompt = "关山月"
prompt = "花间一壶酒，独酌无相亲。"
model_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

for k in range(num_samples):
    generated_ids = model.generate(**model_inputs, generation_config=gen_config)
    res = tokenizer.batch_decode(generated_ids)[0]

    print(res)
