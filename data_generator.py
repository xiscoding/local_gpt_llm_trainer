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

    def generate_example(self, 
                         prompt, 
                         content,
                         prev_examples, 
                         temperature=.5):
        messages = [
            {
                "role": "system",
                "content": f"{content}"
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
                                content, 
                                temperature, 
                                ):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"{content}" 
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

    def generate_examples(self, prompt,content, number_of_examples, temperature):
        prev_examples = []
        for i in range(number_of_examples):
            try:
                print(f'Generating example {i}')
                prompt_tokens = self.tokenizer.tokenize(prompt)
                prev_examples_tokens = [self.tokenizer.tokenize(example) for example in prev_examples]
                total_tokens = len(prompt_tokens) + sum(len(tokens) for tokens in prev_examples_tokens)
                print(f'Tokens in prompt and previous examples: {total_tokens}')
                example = self.generate_example(prompt, content, prev_examples, temperature)
                print(example)
                prev_examples.append(example)
            except openai.error.RateLimitError:
                print("RATELIMITREACHED: waiting 10 seconds")
                time.sleep(10)
        print(prev_examples)
        return prev_examples
