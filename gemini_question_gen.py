import base64
import os
from google import genai
from google.genai import types


TEMPERATURE = 0.15
THINKING_BUDGET = 12700
TOP_P = 0.9
MAX_OUTPUT_TOKENS = 15500

class GeminiQuestionGen:
    
    
    def __init__(self):
        pass
    
    
    def _build_prompt(self, course: str, topic: str, difficulty: str, is_calculation: True):
        if is_calculation:
            return f"""Generate {difficulty} difficulty questions of which 5 calculation multiple choice questions on the topic {topic} for an {course} course. Do this in a json format with the following format:
    
    {{
        "question": "question string",
        "explanation": "explanation string",
        "steps": {{
            "1": "step 1",
            "2": "step 2"
        }},
        "options": ["a", "b", "c", "d"],
        "answer": "answer string"
    }}
    
    Generate questions, answers, explanations, options and the optional calculation steps.
    Ensure the calculation steps have at least 3 steps and no more than 8 steps keeping them short and concise.
    Always solve before providing the answer.
    Ensure proper latex formatting
    """
    
        else:
             return f"""Generate {difficulty} difficulty questions of which 10 theoretical multiple choice questions on the topic {topic} for an {course} course aimed at building an understanding for students. Do this in a json format with the following format:
    
    {{
        "question": "question string",
        "explanation": "explanation string",
        "options": ["a", "b", "c", "d"],
        "answer": "answer string"
    }}
    
    Generate questions, answers, explanations and options.
    Ensure proper latex formatting
    """
    
    


    def generate(self, course="Contol Systems", topic="Laplace Transform", difficulty="medium", is_calculation = False):
        client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY"),
        )

        model = "gemini-2.5-flash-lite"
        prompt = self._build_prompt(course, topic, difficulty, is_calculation=is_calculation)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            thinking_config = types.ThinkingConfig(
                thinking_budget=THINKING_BUDGET,
            ),
        )

        response = ""
        for chunk in client.models.generate_content_stream(model=model, contents=contents, config=generate_content_config):
            if hasattr(chunk, "text"):
                response += chunk.text
            elif hasattr(chunk, "parts"):
                for part in chunk.parts:
                    if hasattr(part, "text"):
                        response += part.text
        print(response)

if __name__ == "__main__":
    gemini = GeminiQuestionGen().generate(is_calculation=True, difficulty="hard")
