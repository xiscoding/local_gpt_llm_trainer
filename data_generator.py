import os
import openai
import random
import time
from transformers import GPT2Tokenizer

class DataGenerator:
    def __init__(self, api_key, cuda_visible_devices="2"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        openai.api_key = api_key
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_example(self, prompt, 
                         prev_examples, 
                         temperature=.5):
        messages = [
            {
                "role": "system",
                "content": f"You are generating data which will be used to train a machine learning model.\n\nYou will be given a high-level description of the model we want to train, and from that, you will generate data samples, each with a prompt/response pair.\n\nYou will do so in this format:\n```\nprompt\n-----------\n$prompt_goes_here\n-----------\n\nresponse\n-----------\n$response_goes_here\n-----------\n```\n\nOnly one prompt/response pair should be generated per turn.\n\nFor each turn, make the example slightly more complex than the last, while ensuring diversity.\n\nMake sure your samples are unique and diverse, yet high-quality and complex enough to train a well-performing model.\n\nHere is the type of model we want to train:\n`{prompt}`"
            }
        ]
        if len(prev_examples) > 0:
            if len(prev_examples) > 10:
                prev_examples = random.sample(prev_examples, 10)
            for example in prev_examples:
                messages.append({
                    "role": "assistant",
                    "content": example
                })
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=temperature,
            max_tokens=1354,
        )
        return response.choices[0].message['content']

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
        return response.choices[0].message['content']

    def generate_examples(self, prompt, number_of_examples, temperature):
        prev_examples = []
        for i in range(number_of_examples):
            try:
                print(f'Generating example {i}')
                prompt_tokens = self.tokenizer.tokenize(prompt)
                prev_examples_tokens = [self.tokenizer.tokenize(example) for example in prev_examples]
                total_tokens = len(prompt_tokens) + sum(len(tokens) for tokens in prev_examples_tokens)
                print(f'Tokens in prompt and previous examples: {total_tokens}')
                example = self.generate_example(prompt, prev_examples, temperature)
                print(example)
                prev_examples.append(example)
            except openai.error.RateLimitError:
                print("RATELIMITREACHED: waiting 10 seconds")
                time.sleep(10)
        print(prev_examples)
        return prev_examples
