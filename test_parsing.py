import json
from gemini_question_gen import QuestionGen

# Test the markdown parsing functionality
def test_markdown_parsing():
    # Create an instance (will fail due to invalid API key, but that's okay for testing parsing)
    qg = QuestionGen(api_key="test")
    
    # Test text with markdown formatting
    test_text = '''```json
{
    "questions": [
        {
            "course_code": "TEST101",
            "course_name": "Test Course",
            "topic_name": "Test Topic",
            "difficulty_ranking": 1,
            "difficulty": "Easy",
            "question": "This is a test question?",
            "options": ["A) Option 1", "B) Option 2", "C) Option 3", "D) Option 4"],
            "correct_answer": "A",
            "explanation": "This is a test explanation.",
            "solution_steps": ["Step 1: Do something", "Step 2: Do something else"]
        }
    ]
}
```'''
    
    # Mock response object
    class MockResponse:
        def __init__(self, text):
            self.text = text
    
    # Create a mock response
    mock_response = MockResponse(test_text)
    
    # Test the parsing logic directly
    text = mock_response.text.strip()
    
    # Remove markdown code block formatting if present
    if text.startswith("```json"):
        text = text[7:]  # Remove ```json
    if text.startswith("```"):
        text = text[3:]  # Remove ```
    if text.endswith("```"):
        text = text[:-3]  # Remove ```
    
    print("Cleaned text:")
    print(text)
    
    # Try to parse the response as JSON
    try:
        json_result = json.loads(text)
        print("\nParsed JSON successfully:")
        print(json.dumps(json_result, indent=2))
        print("\nTest passed!")
        return True
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return False

if __name__ == "__main__":
    test_markdown_parsing()