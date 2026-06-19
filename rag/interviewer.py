from openai import OpenAI

SYSTEM_PROMPT = """당신은 지원자의 이력서를 기반으로 면접관의 질문에 답하는 AI 어시스턴트입니다.
이력서에 근거하여 명확하고 간결하게 답변하세요.
이력서에 없는 내용은 절대 추측하거나 지어내지 마세요.
관련 내용이 없다면 "이력서에서 관련 내용을 찾을 수 없습니다."라고 솔직하게 답하세요.
한국어로 답변하세요."""


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def chat_with_resume(
    history: list[dict],
    new_query: str,
    resume_context: str,
    api_key: str,
) -> str:
    # 최신 질문에만 검색된 이력서 맥락을 붙여서 전달
    augmented_query = f"[참고 이력서 내용]\n{resume_context}\n\n[질문]\n{new_query}"

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    api_messages += history  # 이전 대화 맥락 유지
    api_messages.append({"role": "user", "content": augmented_query})

    response = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=api_messages,
        temperature=0.4,
    )
    return response.choices[0].message.content
