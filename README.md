# local_gpt_llm_trainer

## Overview 
This is a local python implementation of the [gpt-llm-trainer](https://github.com/mshumer/gpt-llm-trainer/tree/main) by [mshumer](https://github.com/mshumer).
## Features
You can now specify the dataset path and train a model from a previously generated dataset.
Otherwise the feature set is the same as the original gpt-llm-traininer:
  
- **Dataset Generation**: Using GPT-4, `gpt-llm-trainer` will generate a variety of prompts and responses based on the provided use-case.

- **System Message Generation**: `gpt-llm-trainer` will generate an effective system prompt for your model.

- **Fine-Tuning**: After your dataset has been generated, the system will automatically split it into training and validation sets, fine-tune a model for you, and get it ready for inference.
## Installation:
- **Required Software**
	- Python 3.9+ (I use anaconda)
	- openAI API key (**A dataset of around 1000 examples seemed to almost cost 10$**)
		- This is probably an exaggeration (I was generating more than just the data), but this is not free to run by any means
	- IDE (I use vscode)
- **Install steps**
	- Create anaconda environment (optional but why not)
	- conda install pip (only if using anaconda environment)
	- clone this repo
	- pip install -r requirements.txt
## Usage
The only file you need to run is the gpt_llm_trainer.py, but in each file there will be things you need to change to work for your personal use
Open each file, search for the terms, make appropriate edits
- gpt_llm_dataprep.py 
	- **openai.api_key** (required to run)
	- prompt
	- temperature
	- number_of_examples
	- The data generation outputs will also be here
- data_generator.py
	- os.environ\[CUDA**] set device order and visible devices 
	- the prompt and system message are generated here (I wouldnt change the generate_system_message content, but here is where it is specified)
```python
messages = [
	{
	"role": "system",
	"content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
	
	}
]	
```

``` python
def generate_system_message(self,
prompt,
temperature,
):
	response = openai.ChatCompletion.create(
	model="gpt-4",
	messages=[
	{
	"role": "system",
	"content": "You will be given a high-level description of the model we are training, and from that, you will generate a simple system prompt for that model to use. Remember, you are not generating the system message for data generation -- you are generating the system message to use for inference. A good format to follow is `Given $INPUT_DATA, you will $WHAT_THE_MODEL_SHOULD_DO.`.\n\nMake it as concise as possible. Include nothing but the system prompt in your response.\n\nFor example, never write: `\"$SYSTEM_PROMPT_HERE\"`.\n\nIt should be like: `$SYSTEM_PROMPT_HERE`."
	},
	{
	"role": "user",
	"content": prompt.strip(),
	}
	],
	temperature=temperature,
	max_tokens=500,
)
```

- gpt_llm_trainer.py
	- This is the only file you need to run 
	- Hyper parameters are specified here 
``` python
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
```

