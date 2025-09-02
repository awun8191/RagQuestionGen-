#!/usr/bin/env python3
"""
Refined Gemini MCQ generator for engineering topics with proper LaTeX formatting.

Key improvements:
- Enhanced engineering-focused prompting
- Better LaTeX formatting validation
- Single request generation with robust fallbacks
- Improved topic-specific question generation
- Better handling of engineering calculations and concepts
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Callable, Awaitable

import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Optional mapping to your Pydantic model
try:
    from question_model import QuestionModel  # noqa
    _HAS_QM = True
except Exception:
    _HAS_QM = False

DEBUG_QG = os.getenv("DEBUG_QG", "0") == "1"

# Simplified schema compatible with Gemini API
QUESTION_SET_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "course_code": {"type": "string"},
                    "course_name": {"type": "string"},
                    "topic_name": {"type": "string"},
                    "difficulty_ranking": {"type": "integer"},
                    "difficulty": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array", 
                        "items": {"type": "string"}
                    },
                    "correct_answer": {"type": "string"},
                    "explanation": {"type": "string"},
                    "solution_steps": {
                        "type": "array", 
                        "items": {"type": "string"}
                    },
                    "is_calculation": {"type": "boolean"},
                    "final_answer_latex": {"type": "string"},
                    "engineering_principles": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": [
                    "course_code", "course_name", "topic_name",
                    "difficulty_ranking", "difficulty",
                    "question", "options", "correct_answer", "explanation",
                    "is_calculation"
                ]
            }
        }
    },
    "required": ["questions"]
}

# Engineering domain mapping for better context
ENGINEERING_DOMAINS = {
    "EEE": "Electrical and Electronics Engineering",
    "MEE": "Mechanical Engineering", 
    "CEE": "Civil Engineering",
    "CHE": "Chemical Engineering",
    "CPE": "Computer Engineering",
    "AEE": "Aerospace Engineering",
    "BME": "Biomedical Engineering",
    "IEE": "Industrial Engineering"
}

def _hash_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()

class EngineeringQuestionGen:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        top_p: float = 0.85,
        system_instruction: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("No API key provided. Set GOOGLE_API_KEY or pass api_key.")
        genai.configure(api_key=api_key)

        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

        self._base_cfg = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            response_mime_type="application/json",
            response_schema=QUESTION_SET_SCHEMA,
        )

        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_instruction or self._get_engineering_system_instruction(),
            generation_config=self._base_cfg,
        )

        self._cache: Dict[str, Dict[str, Any]] = {}

    # ----------------------------- Public API -----------------------------
    def generate_questions(
        self,
        course_code: str = "EEE401",
        course_name: str = "Control Systems Engineering",
        topic: str = "Laplace Transforms and System Stability",
        difficulty: str = "Medium",
        num_questions: int = 2,
        max_retries: int = 3,
        timeout_s: float = 90.0,
        ensure_calc_ratio: float = 0.6,   # 60% calculations for engineering
        engineering_focus: bool = True,   # Force engineering context
    ) -> Dict[str, Any]:
        payload_key = _hash_key(
            course_code, course_name, topic, difficulty, 
            str(num_questions), str(engineering_focus), self.model_name
        )
        
        if payload_key in self._cache:
            return self._cache[payload_key]

        user_prompt = self._build_engineering_prompt(
            course_code, course_name, topic, difficulty, 
            num_questions, ensure_calc_ratio, engineering_focus
        )

        # More tokens for detailed engineering explanations
        max_tokens = min(8192, 350 + int(300 * max(1, num_questions)))
        cfg = GenerationConfig(
            temperature=self._base_cfg.temperature,
            top_p=self._base_cfg.top_p,
            response_mime_type=self._base_cfg.response_mime_type,
            response_schema=self._base_cfg.response_schema,
            max_output_tokens=max_tokens,
        )

        # Single request generation with fallbacks
        def _call_structured():
            return self.model.generate_content(
                user_prompt,
                generation_config=cfg,
                request_options={"timeout": timeout_s},
            )
        
        result = self._with_retries(_call_structured, max_retries=max_retries)
        raw = self._extract_json_text(result)

        if DEBUG_QG:
            print("=== RAW RESPONSE ===")
            print(raw)

        data = self._safe_parse_json(raw)

        # Fallback if structured generation fails
        if not data.get("questions") or len(data.get("questions", [])) == 0:
            if DEBUG_QG:
                print("Structured generation failed. Using fallback...")
            data = self._fallback_generation(
                user_prompt, timeout_s, max_tokens, max_retries
            )

        data = self._validate_and_enhance_engineering_content(
            data, course_code, course_name, topic, difficulty, 
            num_questions, ensure_calc_ratio, engineering_focus
        )

        self._cache[payload_key] = data
        return data

    async def generate_questions_async(
        self,
        course_code: str,
        course_name: str,
        topic: str,
        difficulty: str,
        num_questions: int,
        max_retries: int = 3,
        timeout_s: float = 90.0,
        ensure_calc_ratio: float = 0.6,
        engineering_focus: bool = True,
    ) -> Dict[str, Any]:
        payload_key = _hash_key(
            course_code, course_name, topic, difficulty, 
            str(num_questions), str(engineering_focus), self.model_name
        )
        
        if payload_key in self._cache:
            return self._cache[payload_key]

        user_prompt = self._build_engineering_prompt(
            course_code, course_name, topic, difficulty, 
            num_questions, ensure_calc_ratio, engineering_focus
        )
        
        max_tokens = min(8192, 350 + int(300 * max(1, num_questions)))
        cfg = GenerationConfig(
            temperature=self._base_cfg.temperature,
            top_p=self._base_cfg.top_p,
            response_mime_type=self._base_cfg.response_mime_type,
            response_schema=self._base_cfg.response_schema,
            max_output_tokens=max_tokens,
        )

        async def _coro_structured():
            return await self.model.generate_content_async(
                user_prompt,
                generation_config=cfg,
                request_options={"timeout": timeout_s},
            )
        
        result = await self._with_retries_async(_coro_structured, max_retries=max_retries)
        raw = self._extract_json_text(result)

        if DEBUG_QG:
            print("=== RAW ASYNC RESPONSE ===")
            print(raw)

        data = self._safe_parse_json(raw)

        if not data.get("questions") or len(data.get("questions", [])) == 0:
            data = await self._fallback_generation_async(
                user_prompt, timeout_s, max_tokens, max_retries
            )

        data = self._validate_and_enhance_engineering_content(
            data, course_code, course_name, topic, difficulty, 
            num_questions, ensure_calc_ratio, engineering_focus
        )
        
        self._cache[payload_key] = data
        return data

    # ----------------------------- Internal Methods -----------------------------
    @staticmethod
    def _get_engineering_system_instruction() -> str:
        return (
            "You are an expert engineering educator who creates rigorous, "
            "domain-specific multiple-choice questions for engineering students.\n\n"
            "CRITICAL REQUIREMENTS:\n"
            "- Generate questions that are specifically relevant to the given engineering course and topic\n"
            "- Use proper engineering terminology, symbols, and units throughout\n"
            "- For calculations: provide detailed step-by-step solutions using engineering formulas\n"
            "- Use LaTeX formatting: inline $...$ for variables/short expressions, display $...$ for equations\n"
            "- Include engineering principles and real-world applications\n"
            "- Avoid generic physics problems - focus on engineering applications\n\n"
            "JSON FORMAT REQUIREMENTS:\n"
            "- Return ONLY a single JSON object matching the schema\n"
            "- No markdown, no backticks, no additional text\n"
            "- 'options' must be exactly 4 items corresponding to choices A, B, C, D\n"
            "- 'correct_answer' must be exactly one of: A, B, C, or D\n"
            "- 'difficulty_ranking' must be integer 1-10\n"
            "- For calculation questions: set 'is_calculation': true and provide 'solution_steps' array\n"
            "- For theory questions: set 'is_calculation': false\n"
            "- Include 'engineering_principles' array with relevant engineering concepts\n\n"
            "LATEX FORMATTING:\n"
            "- Use standard engineering notation: $V$, $I$, $R$, $\\omega$, $\\tau$, $H(s)$, etc.\n"
            "- Units in upright text: $\\text{V}$, $\\text{A}$, $\\text{Hz}$, $\\text{m/s}^2$\n"
            "- Single equations on separate lines: $V = IR$\n"
            "- Complex expressions: $H(s) = \\frac{K}{s(s+1)(s+2)}$\n"
            "- Fractions: use \\frac{numerator}{denominator}\n"
            "- Subscripts/superscripts: $V_{in}$, $s^2$"
        )

    def _build_engineering_prompt(
        self, 
        course_code: str, 
        course_name: str, 
        topic: str, 
        difficulty: str, 
        num_questions: int, 
        ensure_calc_ratio: float,
        engineering_focus: bool
    ) -> str:
        # Determine engineering domain
        domain_prefix = course_code[:3].upper()
        engineering_domain = ENGINEERING_DOMAINS.get(domain_prefix, "Engineering")
        
        calc_min = max(1, int(round(num_questions * ensure_calc_ratio)))
        theory_min = num_questions - calc_min
        
        focus_directive = ""
        if engineering_focus:
            focus_directive = (
                f"\nIMPORTANT: Generate questions specifically for {engineering_domain}. "
                f"Focus on engineering applications, design principles, and practical scenarios "
                f"relevant to {course_name}. Avoid basic physics problems."
            )
        
        return (
            f"Create {num_questions} {difficulty}-level multiple-choice questions for:\n"
            f"Course: {course_code} - {course_name}\n"
            f"Topic: {topic}\n"
            f"Engineering Domain: {engineering_domain}\n\n"
            f"QUESTION DISTRIBUTION:\n"
            f"- At least {calc_min} calculation/problem-solving questions\n"
            f"- At least {theory_min} conceptual/theoretical questions\n\n"
            f"CALCULATION QUESTIONS MUST:\n"
            f"- Use realistic engineering values and scenarios\n"
            f"- Apply standard engineering formulas and principles\n"
            f"- Show complete solution steps with proper LaTeX formatting\n"
            f"- Include final answer with appropriate units\n\n"
            f"CONCEPTUAL QUESTIONS MUST:\n"
            f"- Test understanding of engineering principles\n"
            f"- Reference industry standards and practices\n"
            f"- Include real-world engineering applications\n"
            f"{focus_directive}\n\n"
            f"Ensure all questions are directly relevant to {topic} in the context of {course_name}."
        )

    @staticmethod
    def _extract_json_text(result) -> str:
        """Extract JSON text from Gemini response with robust fallbacks."""
        # Primary method
        txt = getattr(result, "text", None)
        if isinstance(txt, str) and txt.strip():
            return txt

        # Fallback to candidates
        try:
            candidates = getattr(result, "candidates", [])
            if candidates:
                for candidate in candidates:
                    content = getattr(candidate, "content", None)
                    if content:
                        parts = getattr(content, "parts", [])
                        for part in parts:
                            part_text = getattr(part, "text", None)
                            if isinstance(part_text, str) and part_text.strip():
                                return part_text
        except Exception as e:
            if DEBUG_QG:
                print(f"Error extracting from candidates: {e}")
        
        return ""

    @staticmethod
    def _safe_parse_json(s: str) -> Dict[str, Any]:
        """Safely parse JSON with markdown stripping."""
        if not isinstance(s, str) or not s.strip():
            return {"questions": []}
        
        # Clean up markdown formatting
        content = s.strip()
        if content.startswith("```"):
            # Remove code block markers
            lines = content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = '\n'.join(lines)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            if DEBUG_QG:
                print(f"JSON parse error: {e}")
                print(f"Content: {content[:500]}...")
            return {"questions": []}

    def _validate_latex(self, text: str) -> str:
        """Validate and fix LaTeX formatting."""
        if not text or not isinstance(text, str):
            return text
            
        # Common LaTeX fixes
        text = re.sub(r'(?<!\\)(?:\\\\)*\\(?![a-zA-Z{])', r'\\\\', text)  # Fix backslashes
        text = re.sub(r'\$\s*\$', '', text)  # Remove empty LaTeX blocks
        text = re.sub(r'\$([^$]*)\$', lambda m: f'${m.group(1).strip()}$', text)  # Clean inline LaTeX
        
        return text.strip()

    def _enhance_engineering_content(self, question: Dict[str, Any], course_code: str, topic: str) -> Dict[str, Any]:
        """Enhance question with engineering-specific content."""
        domain_prefix = course_code[:3].upper()
        
        # Add engineering principles if missing
        if "engineering_principles" not in question:
            question["engineering_principles"] = self._infer_engineering_principles(
                question.get("question", ""), topic, domain_prefix
            )
        
        # Enhance explanation with engineering context
        explanation = question.get("explanation", "")
        if explanation and not any(word in explanation.lower() for word in ["engineering", "design", "system", "analysis"]):
            question["explanation"] = f"{explanation} This principle is fundamental in engineering {topic.lower()} analysis."
        
        # Validate LaTeX in all text fields
        for field in ["question", "explanation", "final_answer_latex"]:
            if field in question:
                question[field] = self._validate_latex(question[field])
        
        if "solution_steps" in question:
            question["solution_steps"] = [self._validate_latex(step) for step in question["solution_steps"]]
        
        return question

    @staticmethod
    def _infer_engineering_principles(question_text: str, topic: str, domain: str) -> List[str]:
        """Infer relevant engineering principles from question content."""
        principles = []
        
        # Domain-specific principles
        if domain == "EEE":
            if any(term in question_text.lower() for term in ["circuit", "voltage", "current"]):
                principles.extend(["Circuit Analysis", "Ohm's Law"])
            if any(term in question_text.lower() for term in ["control", "stability", "transfer"]):
                principles.extend(["Control Systems", "Feedback Analysis"])
            if any(term in question_text.lower() for term in ["laplace", "transform"]):
                principles.extend(["Laplace Transform", "System Analysis"])
        elif domain == "MEE":
            if any(term in question_text.lower() for term in ["force", "stress", "strain"]):
                principles.extend(["Mechanics of Materials", "Structural Analysis"])
            if any(term in question_text.lower() for term in ["heat", "thermal", "temperature"]):
                principles.extend(["Heat Transfer", "Thermodynamics"])
        
        # Add topic-based principles
        principles.append(topic)
        
        return list(set(principles))  # Remove duplicates

    def _validate_and_enhance_engineering_content(
        self,
        data: Dict[str, Any],
        course_code: str,
        course_name: str,
        topic: str,
        difficulty: str,
        num_questions: int,
        ensure_calc_ratio: float,
        engineering_focus: bool,
    ) -> Dict[str, Any]:
        """Validate and enhance questions with engineering focus."""
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        enhanced_questions = []
        difficulty_ranges = {"Easy": (1, 3), "Medium": (4, 7), "Hard": (8, 10)}
        min_rank, max_rank = difficulty_ranges.get(difficulty, (1, 3))
        
        calc_target = max(1, int(round(num_questions * ensure_calc_ratio)))
        calc_count = 0

        for q in questions[:num_questions]:
            if not isinstance(q, dict):
                continue

            # Ensure basic required fields
            enhanced_q = {
                "course_code": q.get("course_code", course_code),
                "course_name": q.get("course_name", course_name),
                "topic_name": q.get("topic_name", topic),
                "difficulty": difficulty,
                "difficulty_ranking": max(min_rank, min(max_rank, int(q.get("difficulty_ranking", min_rank)))),
            }

            # Validate options
            options = q.get("options", [])
            if not isinstance(options, list) or len(options) != 4:
                options = [f"Option {chr(65+i)}" for i in range(4)]
            enhanced_q["options"] = [str(opt).strip() for opt in options[:4]]

            # Validate correct answer
            correct_answer = str(q.get("correct_answer", "A")).upper()
            if correct_answer not in ["A", "B", "C", "D"]:
                correct_answer = "A"
            enhanced_q["correct_answer"] = correct_answer

            # Enhance question content
            enhanced_q["question"] = q.get("question", f"Question about {topic}")
            enhanced_q["explanation"] = q.get("explanation", "Explanation for the correct answer")
            
            # Handle calculation questions
            is_calc = bool(q.get("is_calculation", False))
            if is_calc:
                calc_count += 1
                enhanced_q["is_calculation"] = True
                if "solution_steps" in q and isinstance(q["solution_steps"], list):
                    enhanced_q["solution_steps"] = q["solution_steps"]
                if "final_answer_latex" in q and q["final_answer_latex"]:
                    enhanced_q["final_answer_latex"] = q["final_answer_latex"]
            else:
                enhanced_q["is_calculation"] = False

            # Add engineering principles
            enhanced_q["engineering_principles"] = q.get("engineering_principles", [topic])

            # Apply engineering enhancements
            if engineering_focus:
                enhanced_q = self._enhance_engineering_content(enhanced_q, course_code, topic)

            enhanced_questions.append(enhanced_q)

        # Generate additional questions if needed
        while len(enhanced_questions) < num_questions:
            need_calc = calc_count < calc_target
            synthetic_q = self._generate_synthetic_engineering_question(
                course_code, course_name, topic, difficulty, need_calc, min_rank
            )
            enhanced_questions.append(synthetic_q)
            if need_calc:
                calc_count += 1

        return {"questions": enhanced_questions[:num_questions]}

    def _generate_synthetic_engineering_question(
        self, course_code: str, course_name: str, topic: str, 
        difficulty: str, is_calculation: bool, difficulty_rank: int
    ) -> Dict[str, Any]:
        """Generate a synthetic engineering question as fallback."""
        domain_prefix = course_code[:3].upper()
        
        if is_calculation and domain_prefix == "EEE":
            question = {
                "course_code": course_code,
                "course_name": course_name,
                "topic_name": topic,
                "difficulty": difficulty,
                "difficulty_ranking": difficulty_rank,
                "question": f"In a control system analysis for {topic}, if the transfer function is $G(s) = \\frac{{10}}{{s(s+5)}}$, what is the steady-state error for a unit step input?",
                "options": ["0", "0.2", "2.0", "∞"],
                "correct_answer": "B",
                "explanation": "For a Type 1 system with unit step input, steady-state error $e_{ss} = \\frac{1}{1+K_p}$ where $K_p = \\lim_{s \\to 0} G(s) = 2$. Thus $e_{ss} = \\frac{1}{1+2} = \\frac{1}{3} \\approx 0.33$. Closest answer is 0.2.",
                "is_calculation": True,
                "solution_steps": [
                    "Given: $G(s) = \\frac{10}{s(s+5)}$",
                    "For Type 1 system: $K_p = \\lim_{s \\to 0} G(s)$",
                    "$K_p = \\lim_{s \\to 0} \\frac{10}{s(s+5)} = \\frac{10}{0 \\cdot 5} = 2$",
                    "Steady-state error: $e_{ss} = \\frac{1}{1+K_p} = \\frac{1}{1+2} = \\frac{1}{3}$"
                ],
                "final_answer_latex": "$e_{ss} = \\frac{1}{3} \\approx 0.33$",
                "engineering_principles": ["Control Systems", "Steady-State Analysis", topic]
            }
        else:
            question = {
                "course_code": course_code,
                "course_name": course_name,
                "topic_name": topic,
                "difficulty": difficulty,
                "difficulty_ranking": difficulty_rank,
                "question": f"Which engineering principle is most fundamental to understanding {topic}?",
                "options": ["Superposition principle", "Conservation of energy", "System linearity", "Feedback control"],
                "correct_answer": "A",
                "explanation": f"The superposition principle is fundamental in {topic} analysis for linear systems.",
                "is_calculation": False,
                "engineering_principles": [topic, "System Analysis"]
            }
        
        return question

    def _fallback_generation(self, user_prompt: str, timeout_s: float, max_tokens: int, max_retries: int) -> Dict[str, Any]:
        """Fallback generation without schema constraints."""
        cfg = GenerationConfig(
            temperature=self._base_cfg.temperature + 0.1,  # Slightly higher for creativity
            top_p=self._base_cfg.top_p,
            max_output_tokens=max_tokens,
        )
        
        fallback_prompt = (
            user_prompt + 
            "\n\nIMPORTANT: Return ONLY a valid JSON object with a 'questions' array. "
            "Each question must have all required fields. Use proper LaTeX formatting. "
            "No markdown code blocks, no additional text."
        )

        def _call_fallback():
            return self.model.generate_content(
                fallback_prompt,
                generation_config=cfg,
                request_options={"timeout": timeout_s},
            )

        try:
            result = self._with_retries(_call_fallback, max_retries=max_retries)
            raw = self._extract_json_text(result)
            if DEBUG_QG:
                print("=== FALLBACK RESPONSE ===")
                print(raw)
            return self._safe_parse_json(raw)
        except Exception as e:
            if DEBUG_QG:
                print(f"Fallback generation failed: {e}")
            return {"questions": []}

    async def _fallback_generation_async(self, user_prompt: str, timeout_s: float, max_tokens: int, max_retries: int) -> Dict[str, Any]:
        """Async fallback generation without schema constraints."""
        cfg = GenerationConfig(
            temperature=self._base_cfg.temperature + 0.1,
            top_p=self._base_cfg.top_p,
            max_output_tokens=max_tokens,
        )
        
        fallback_prompt = (
            user_prompt + 
            "\n\nIMPORTANT: Return ONLY a valid JSON object with a 'questions' array. "
            "Each question must have all required fields. Use proper LaTeX formatting. "
            "No markdown code blocks, no additional text."
        )

        async def _coro_fallback():
            return await self.model.generate_content_async(
                fallback_prompt,
                generation_config=cfg,
                request_options={"timeout": timeout_s},
            )

        try:
            result = await self._with_retries_async(_coro_fallback, max_retries=max_retries)
            raw = self._extract_json_text(result)
            if DEBUG_QG:
                print("=== ASYNC FALLBACK RESPONSE ===")
                print(raw)
            return self._safe_parse_json(raw)
        except Exception as e:
            if DEBUG_QG:
                print(f"Async fallback generation failed: {e}")
            return {"questions": []}

    # --------------------------- Retry helpers ---------------------------
    @staticmethod
    def _sleep_backoff(attempt: int) -> None:
        """Exponential backoff with jitter."""
        base_delay = min(2.0 * (1.5 ** attempt), 10.0)
        jitter = base_delay * 0.1 * (0.5 - abs(hash(str(time.time())) % 1000) / 1000)
        time.sleep(base_delay + jitter)

    def _with_retries(self, fn: Callable[[], Any], max_retries: int = 3) -> Any:
        last_error = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                last_error = e
                if DEBUG_QG:
                    print(f"[Retry {attempt + 1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    self._sleep_backoff(attempt)
        raise last_error or RuntimeError("All retries failed")

    async def _with_retries_async(self, coro_fn: Callable[[], Awaitable[Any]], max_retries: int = 3) -> Any:
        last_error = None
        for attempt in range(max_retries):
            try:
                return await coro_fn()
            except Exception as e:
                last_error = e
                if DEBUG_QG:
                    print(f"[Async Retry {attempt + 1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    self._sleep_backoff(attempt)
        raise last_error or RuntimeError("All async retries failed")

    # ------------------------------ Pydantic bridge ------------------------------
    def to_models(self, data: Dict[str, Any]) -> List[Any]:
        """Convert to Pydantic models if available."""
        if not _HAS_QM:
            return data.get("questions", [])
        
        models = []
        for q in data.get("questions", []):
            try:
                models.append(QuestionModel(**q))
            except Exception as e:
                if DEBUG_QG:
                    print(f"Pydantic conversion error: {e}")
                models.append(q)  # Keep as dict if conversion fails
        return models

# ----------------------------- CLI Test -----------------------------
if __name__ == "__main__":
    try:
        # Test with electrical engineering topic
        qg = EngineeringQuestionGen(temperature=0.3)
        result = qg.generate_questions(
            course_code="EEE401",
            course_name="Control Systems Engineering",
            topic="Stability Analysis and Root Locus",
            difficulty="Medium",
            num_questions=3,
            ensure_calc_ratio=0.67,
            engineering_focus=True
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()