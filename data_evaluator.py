import os
import openai
import time
from openai import RateLimitError

class DataEvaluator:
    """
    A class to evaluate query/response pairs using the OpenAI ChatCompletion API.
    """

    def __init__(self, api_key=None, model_name="gpt-3.5-turbo"):
        """
        :param api_key: Your OpenAI API key (can also be set via environment variable).
        :param model_name: Which OpenAI model to use for evaluation (default "gpt-3.5-turbo").
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.model_name = model_name

    def evaluate_example(self, query, response, 
                         eval_instructions="Provide a concise evaluation of the correctness, clarity, and completeness of the given response to the query.", 
                         temperature=0.3, 
                         max_tokens=200):
        """
        Uses the ChatCompletion API to generate an evaluation of a single query/response pair.

        :param query: The user query or prompt.
        :param response: The answer or response being evaluated.
        :param eval_instructions: Instructions explaining how the model should evaluate the response.
        :param temperature: Controls randomness in generation (0-1).
        :param max_tokens: The maximum number of tokens in the evaluation response.
        :return: A string containing the evaluation.
        """
        # Build the conversation
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful but critical evaluator. "
                    "You will be given a user query and a proposed response. "
                    "Follow the given evaluation instructions to assess the response."
                )
            },
            {
                "role": "user",
                "content": f"{eval_instructions}\n\nQuery:\n{query}\n\nResponse:\n{response}"
            },
        ]

        try:
            # Call OpenAI's chat completion endpoint
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            # Return the model's evaluation
            return completion.choices[0].message.content.strip()

        except RateLimitError:
            print("Rate limit reached. Waiting 10 seconds before retrying...")
            time.sleep(10)
            # Retry once
            completion = openai.ChatCompletion.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return completion.choices[0].message.content.strip()

if __name__ == "__main__":
    evaluator = DataEvaluator(api_key="YOUR_API_KEY", model_name="gpt-3.5-turbo")

    query_example = "What are the primary benefits of regular exercise?"
    response_example = ("Regular exercise helps improve cardiovascular health, boosts mood, and "
                        "maintains healthy body weight. It can also reduce risk of certain diseases.")

    evaluation_instructions = (
        "Evaluate the correctness, clarity, and completeness of the given response. "
        "Highlight any strengths, weaknesses, or missing elements."
    )

    evaluation_result = evaluator.evaluate_example(
        query=query_example,
        response=response_example,
        eval_instructions=evaluation_instructions,
        temperature=0.3,
        max_tokens=150
    )

    print("Evaluation Result:")
    print(evaluation_result)
