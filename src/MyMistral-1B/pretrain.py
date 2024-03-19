"""
从头训练一个语言模型TinyLM，直接利用Mistral-7B的模型架构和tokenizer，训练数据使用56315首唐诗，
训练机器使用AWS EC2-G5.4xlarge 1台(1 A10G GPU)
TinyLM-400M
TinyLM-127M
"""

# %% 
from transformers import TrainingArguments, Trainer, AutoTokenizer, DataCollatorForLanguageModeling
from model.configuration_mistral import MistralConfig
from model.modeling_mistral import MistralForCausalLM
from datasets import load_dataset, Dataset
import numpy as np


# %%
config = MistralConfig.from_pretrained("model/config.json")
config


# %%
model = MistralForCausalLM(config)
model


# %%
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    add_eos_token=True,
    add_bos_token=True,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token


# %%
def to_token_ids(sample: dict) -> dict:
    all_poem = sample['title'] + '\n' + sample['author'] + '\n' + ''.join(sample['content'])
    # print(all_poem)
    outputs = tokenizer(
        all_poem,
        truncation=False,
        padding=False,
    )
    # print(outputs)

    input_ids = [np.array(item, dtype=np.uint16) for item in outputs["input_ids"]]

    return {
            "input_ids": input_ids
        }


# %%
train = load_dataset("json", data_files="/home/ubuntu/GenAI/nanoGPT/data/poem-tangsong/poet.tang.json", split='train[:90%]')
eval = load_dataset("json", data_files="/home/ubuntu/GenAI/nanoGPT/data/poem-tangsong/poet.tang.json", split='train[90%:]')
# dataset
train_dataset = train.map(to_token_ids, remove_columns=train.column_names)
eval_dataset = eval.map(to_token_ids, remove_columns=eval.column_names)
print(train_dataset, eval_dataset)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


# %%
args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=6,
    weight_decay=0.1,
    warmup_steps=0,
    learning_rate=1e-5,
    evaluation_strategy='steps',
    eval_steps=5000,
    save_steps=2000,
    save_strategy='steps',
    save_total_limit=4,
    optim="adamw_torch",
    lr_scheduler_type='cosine',
    bf16=True,
    logging_steps=1000,
    log_level='info',
    logging_first_step=True,
    report_to="wandb",
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)


# %%
trainer.train()


# %%
trainer.save_model("output-5-6epochs")

# %%
