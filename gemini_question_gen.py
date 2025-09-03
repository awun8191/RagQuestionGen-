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
    
    def _sanitize(self, text: str):
        """Extract JSON from `text` and return the parsed Python object.

        Strategy:
        - First, try to find a fenced code block that starts with ```json and ends with ```.
        - If not found, try to parse the entire text as JSON.
        - If that fails, try to extract JSON-like content using regex for { ... }.
        - Then, try to parse the extracted content with json.loads.
        - On JSON decode error, attempt fallbacks (escape backslashes, or use ast.literal_eval).
        """
        import re
        import json

        code = None
        # Try to find the first ```json ... ``` block
        m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL | re.IGNORECASE)
        if m:
            code = m.group(1).strip()
        else:
            # Try to parse the whole text as JSON
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                pass
            # Try to extract JSON using regex for outermost { ... }
            m2 = re.search(r'\{.*\}', text, re.DOTALL)
            if m2:
                code = m2.group(0).strip()
            else:
                raise ValueError("No JSON content found in text")

        # Now parse the code
        # Try direct JSON parse first
        try:
            return json.loads(code)
        except json.JSONDecodeError:
            # Try escaping single backslashes (common when LaTeX backslashes are present)
            try:
                fixed = code.replace('\\', '\\\\')
                return json.loads(fixed)
            except json.JSONDecodeError:
                # Last resort: try ast.literal_eval after converting JSON literals to Python
                import ast
                pyish = code.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                try:
                    return ast.literal_eval(pyish)
                except Exception as e:
                    raise ValueError('Failed to parse JSON content') from e



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
        response = self._sanitize(response)
        print(response)

if __name__ == "__main__":
    gemini = GeminiQuestionGen().generate(is_calculation=True, difficulty="hard")
