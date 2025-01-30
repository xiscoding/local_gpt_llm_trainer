import os
from openai import OpenAI
from openai import RateLimitError
import random
import time
from transformers import GPT2Tokenizer


####### NOTE: CUDA SETTTINGS HERE REDUNDANT?##########################


class DataGenerator:
    def __init__(self, api_key, cuda_visible_devices="2", model_name ="gpt-4o-mini"):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
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
        response = self.client.chat.completions.create(model= self.model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=100)
        return response.choices[0].message.content

    def generate_system_message(self, 
                                prompt,
                                content, 
                                temperature, 
                                ):
        response = self.client.chat.completions.create(model= self.model_name,
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
        max_tokens=100)
        return response.choices[0].message.content

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
            except RateLimitError:
                print("RATELIMITREACHED: waiting 10 seconds")
                time.sleep(10)
        print(prev_examples)
        return prev_examples
    
    def create_query_response_pairs(self, data):
        """
        Takes a dictionary of the form:
        
        {
        "system_message": "<string>",
        "examples": [
            "```\nprompt\n-----------\n<PROMPT_TEXT>\n-----------\n\nresponse\n-----------\n<RESPONSE_TEXT>\n-----------\n```",
            ...
        ]
        }
        
        Returns a dictionary:
        
        {
        "system_message": "<string>",
        "examples": [
            {"query": "<PROMPT_TEXT>", "response": "<RESPONSE_TEXT>"},
            ...
        ]
        }
        """
        system_message = data.get("system_message", "")
        raw_examples = data.get("examples", [])
        
        parsed_examples = []
        for ex in raw_examples:
            # 1) Remove triple backticks to simplify parsing
            ex_clean = ex.replace("```", "").strip()
            
            # 2) Split on the literal "-----------"
            #    The data format is something like:
            #    prompt
            #    -----------
            #    <PROMPT_TEXT>
            #    -----------
            #
            #    response
            #    -----------
            #    <RESPONSE_TEXT>
            #    -----------
            parts = ex_clean.split("-----------")

            # Expecting at least 4 chunks if well-formed:
            # parts[0] might contain "prompt\n"
            # parts[1] is the raw prompt text
            # parts[2] might contain "response\n"
            # parts[3] is the raw response text
            if len(parts) < 4:
                # If it's not well-formed, skip or handle differently
                continue
            
            # Strip extra whitespace/newlines
            prompt_text = parts[1].strip()
            response_text = parts[3].strip()
            
            parsed_examples.append({
                "query": prompt_text,
                "response": response_text
            })
        
        return {
            "system_message": system_message,
            "examples": parsed_examples
        }

