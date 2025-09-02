import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gemini_question_gen import QuestionGen
from unittest.mock import Mock, patch

def test_question_gen_without_api_key():
    """Test the QuestionGen class without requiring an actual API key"""
    # Mock the genai module and its methods
    with patch('gemini_question_gen.genai') as mock_genai:
        # Mock the GenerativeModel and its generate_content method
        mock_model = Mock()
        # Simulate a response with markdown formatting
        mock_model.generate_content.return_value.text = '''```json
{
    "questions": [
        {
            "course_code": "PHY101",
            "course_name": "Physics Fundamentals",
            "topic_name": "Kinematics",
            "difficulty_ranking": 2,
            "difficulty": "Easy",
            "question": "A ball is thrown vertically upwards with an initial velocity of 20 m/s. Ignoring air resistance, what is the maximum height the ball reaches? (Use g = 10 m/s²)",
            "options": [
                "A) 10 m",
                "B) 20 m",
                "C) 30 m",
                "D) 40 m"
            ],
            "correct_answer": "B",
            "explanation": "At the maximum height, the ball's final velocity (v) will be 0 m/s. We can use the following kinematic equation to solve for the height (h): v² = u² + 2as, where u is the initial velocity, a is the acceleration due to gravity (-10 m/s² since it's acting downwards), and s is the displacement (height).",
            "solution_steps": [
                "Step 1: Identify known values: u = 20 m/s, v = 0 m/s, a = -10 m/s²",
                "Step 2: Use the equation v² = u² + 2as",
                "Step 3: Substitute values: 0² = 20² + 2(-10)s",
                "Step 4: Simplify: 0 = 400 - 20s",
                "Step 5: Solve for s: 20s = 400, so s = 20 m"
            ]
        }
    ]
}
```'''
        
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        
        # Create an instance with a mock API key
        question_gen = QuestionGen(api_key="mock-api-key")
        
        # Generate a question (this will test the markdown parsing)
        result = question_gen.generate_questions(
            course_code="PHY101",
            course_name="Physics Fundamentals",
            topic="Kinematics",
            difficulty="Easy",
            num_questions=1
        )
        
        # Print the result
        print("Generated Questions (JSON format):")
        print(json.dumps(result, indent=2))
        
        # Verify the result structure
        assert "questions" in result
        assert len(result["questions"]) > 0
        question = result["questions"][0]
        assert "course_code" in question
        assert "course_name" in question
        assert "topic_name" in question
        assert "difficulty_ranking" in question
        assert "difficulty" in question
        assert "question" in question
        assert "options" in question
        assert "correct_answer" in question
        assert "explanation" in question
        assert "solution_steps" in question
        print("\nTest passed! The QuestionGen class works correctly.")

if __name__ == "__main__":
    test_question_gen_without_api_key()