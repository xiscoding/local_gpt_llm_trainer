import json
import openai
import os
from data_generator import DataGenerator
from secret import SECRET
"""
The main function is responsible for generating examples and a system message based on a defined prompt.
It uses the following elements:
- prompt: A description used to generate examples and a system message.
- temperature: A value between 0 and 1 that controls randomness in generation (lower value means more deterministic).
- number_of_examples: The number of examples to generate.
- openai.api_key: The API key for OpenAI.

The generated examples and system message are saved to a JSON file 'output.json'.
"""
def main():

    # Load content and prompt strings from JSON
    with open("prompts.json", "r") as file:
        prompts_data = json.load(file)

    group_name = "generate_puzzles"
    example_content = prompts_data[group_name]["example_content"]
    system_message_content = prompts_data[group_name]["system_message_content"]
    prompt = prompts_data[group_name]["prompt"]

    prompt = prompt
    temperature = 0.4
    number_of_examples = 10
    openai.api_key = SECRET
    
#### NOTE: GENERATE DATA ##########
    data_gen = DataGenerator(openai.api_key)
    # System message
    system_message = data_gen.generate_system_message(
        prompt=prompt, 
        content=system_message_content, 
        temperature=temperature)
    print(f'The system message is: `{system_message}`. Re-run if you want a better result.')

    # Examples
    examples = data_gen.generate_examples(
        prompt=prompt,
        content=example_content,
        number_of_examples=number_of_examples,
        temperature=temperature
    )
    ###### NOTE: FINAL DATA ##################
    data = {
        "system_message": system_message,
        "examples": examples
    }
    data = data_gen.create_query_response_pairs(data)
    with open('output.json', 'w') as file:
        json.dump(data, file)
    return examples, system_message

def load_data_from_file():
    with open('output.json', 'r') as file:
        data = json.load(file)
        return data['examples'], data['system_message']

if __name__ == "__main__":
    main()
