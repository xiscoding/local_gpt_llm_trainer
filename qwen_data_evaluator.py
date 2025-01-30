import openai
import os
from typing import Dict, Any
from secret import SECRET
class DataEvaluator:
    def __init__(self, api_key, model_name="gpt-4o-mini"):
        openai.api_key = api_key
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=api_key)

    def evaluate(self, query: str, response: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": "You are an AI evaluator. You will receive a query and a response. Your task is to evaluate the response based on the following criteria: relevance, accuracy, completeness, and clarity. Provide a score from 1 to 5 for each criterion and an overall comment."},
            {"role": "user", "content": f"Query: {query}\nResponse: {response}"},
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200
            )
            evaluation = response.choices[0].message.content
            return self.parse_evaluation(evaluation)
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {}

    def parse_evaluation(self, evaluation: str) -> Dict[str, Any]:
        # This is a placeholder for actual parsing logic
        # Assuming the model returns a structured response like:
        # Relevance: 4
        # Accuracy: 5
        # Completeness: 3
        # Clarity: 4
        # Overall Comment: The response is relevant and accurate but could be more complete.
        
        lines = evaluation.split('\n')
        parsed = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                if key in ['relevance', 'accuracy', 'completeness', 'clarity']:
                    try:
                        parsed[key] = int(value)
                    except ValueError:
                        parsed[key] = value
                else:
                    parsed['overall_comment'] = value
        return parsed

# Example Usage
if __name__ == "__main__":     
    api_key = SECRET
    #os.environ.get("OPENAI_API_KEY")
    evaluator = DataEvaluator(api_key)
    query = "What is the capital of France?"
    response = "The capital of France is Paris."
    evaluation = evaluator.evaluate(query, response)
    print(evaluation)