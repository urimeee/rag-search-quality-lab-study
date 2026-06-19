from openai import OpenAI


def _client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def search_and_summarize(query: str, resume_context: str, api_key: str) -> str:
    prompt = f"""면접관이 지원자에 대해 다음을 알고 싶어합니다: "{query}"

아래는 지원자 이력서에서 관련 내용을 검색한 결과입니다:

{resume_context}

위 내용을 바탕으로 면접관이 바로 활용할 수 있도록 핵심만 명확하게 정리해주세요.
이력서에 없는 내용은 절대 추가하지 마세요.
내용이 없다면 "이력서에서 관련 내용을 찾을 수 없습니다."라고 답하세요."""

    response = _client(api_key).chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content
