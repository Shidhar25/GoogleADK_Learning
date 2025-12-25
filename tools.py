from pydantic import BaseModel, Field
from typing import List
from pypdf import PdfReader

class QuizInput(BaseModel):
    paragraph: str = Field(..., description="Paragraph to generate quiz from")
    num_questions: int = Field(5, ge=1, le=10)

# ---------- Output Schema ----------
class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str

class QuizOutput(BaseModel):
    questions: List[QuizQuestion]
    status: str


# ---------- Tool ----------
def generate_quiz(input: QuizInput) -> QuizOutput:
    """
    Generates quiz questions from a paragraph.
    (Mock logic – replace with LLM/RAG later)
    """

    sentences = [s.strip() for s in input.paragraph.split(".") if s.strip()]
    questions = []

    for i in range(min(input.num_questions, len(sentences))):
        sentence = sentences[i]

        question = f"What is the main idea of: '{sentence[:50]}...?'"
        options = [
            sentence,
            "Unrelated concept",
            "Opposite meaning",
            "None of the above"
        ]

        questions.append(
            QuizQuestion(
                question=question,
                options=options,
                answer=sentence
            )
        )

    return QuizOutput(
        questions=questions,
        status="success"
    )
def get_current_time(city: str) -> dict:
    return {"city": city, "time": "10:30 AM"}

def get_current_weather(city: str) -> dict:
    return {"city": city, "weather": "Sunny"}

def get_google_search(query: str) -> dict:
    return {"query": query, "results": ["Result 1", "Result 2"]}

def get_pdf_content(file_path: str) -> dict:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return {"content": text[:10000]}  # Limit to first 10k characters
    except Exception as e:
        return {"error": str(e)}