import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# # Load datasets
# train_dataset = load_dataset('json', data_files='/content/train.jsonl')#, split="train")
# valid_dataset = load_dataset('json', data_files='/content/test.jsonl')#, split="train")
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Features, Value
import gpt_llm_dataprep

#CHECK if examples have been generated
#GET generted examples and system message
if os.path.exists('output.json'):
    examples, system_message = gpt_llm_dataprep.load_data_from_file()
else:
    examples, system_message = gpt_llm_dataprep.main()
print("Examples:", examples)
print("System message:", system_message)

# Define Hyperparameters
model_name = "NousResearch/llama-2-7b-chat-hf" # use this if you have access to the official LLaMA 2 model "meta-llama/Llama-2-7b-chat-hf", though keep in mind you'll need to pass a Hugging Face key argument
dataset_name = "train.jsonl"
new_model = "llama-2-7b-custom"
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "constant"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 25
logging_steps = 5
max_seq_length = None
packing = False
device_map = {"": 0}

# Load datasets

# dataset features
features = Features({
    'prompt': Value('string'),
    'response': Value('string'),
})

# load_dataset parameters
train_path = 'train.jsonl'
valid_path = 'test.jsonl'
file_type = 'json'
split = 'train'
train_dataset = load_dataset(file_type, data_files=train_path, features=features, split=split)
valid_dataset = load_dataset(file_type, data_files=valid_path, features=features, split=split)
print(train_dataset)

# Preprocess datasets
train_dataset_mapped = train_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
valid_dataset_mapped = valid_dataset.map(lambda examples: {'text': [f'[INST] <<SYS>>\n{system_message.strip()}\n<</SYS>>\n\n' + prompt + ' [/INST] ' + response for prompt, response in zip(examples['prompt'], examples['response'])]}, batched=True)
print("train_dataset_mapped BELOW:")
print(train_dataset_mapped)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)
# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="all",
    evaluation_strategy="steps",
    eval_steps=5  # Evaluate every 20 steps
)
# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset_mapped,
    eval_dataset=valid_dataset_mapped,  # Pass validation dataset here
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)
trainer.train()
trainer.model.save_pretrained(new_model)

# Cell 4: Test the model
logging.set_verbosity(logging.CRITICAL)
prompt = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nWrite a function that reverses a string. [/INST]" # replace the command here with something relevant to your task
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(prompt)
print(result[0]['generated_text'])