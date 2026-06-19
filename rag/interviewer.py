import json
from openai import OpenAI
from rag.prompts import (
    QUESTION_GENERATION_PROMPT,
    EVALUATION_PROMPT,
    FINAL_FEEDBACK_PROMPT,
)


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def generate_questions(jd: str, resume_context: str, api_key: str) -> list[str]:
    prompt = QUESTION_GENERATION_PROMPT.format(
        resume_context=resume_context,
        jd=jd,
    )
    response = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7,
    )
    result = json.loads(response.choices[0].message.content)
    return result["questions"]


def evaluate_answer(
    question: str,
    answer: str,
    resume_context: str,
    history: list[dict],
    api_key: str,
) -> str:
    history_text = "\n".join(
        f"{'면접관' if m['role'] == 'assistant' else '지원자'}: {m['content']}"
        for m in history[-6:]
    )
    prompt = EVALUATION_PROMPT.format(
        resume_context=resume_context,
        question=question,
        answer=answer,
        history=history_text,
    )
    response = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_final_feedback(
    jd: str,
    resume_context: str,
    history: list[dict],
    api_key: str,
) -> str:
    history_text = "\n".join(
        f"{'면접관' if m['role'] == 'assistant' else '지원자'}: {m['content']}"
        for m in history
    )
    prompt = FINAL_FEEDBACK_PROMPT.format(
        resume_context=resume_context,
        jd=jd,
        history=history_text,
    )
    response = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return response.choices[0].message.content
