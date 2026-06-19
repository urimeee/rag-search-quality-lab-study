import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.embedder import load_vectorstore, get_relevant_context
from rag.interviewer import generate_questions, evaluate_answer, generate_final_feedback
from rag.experimenter import preview_chunks, search_with_scores

st.set_page_config(page_title="AI 면접관 + RAG 실험실", page_icon="🎙️", layout="wide")

MAX_ANSWERS_PER_QUESTION = 2

API_KEY = os.getenv("OPENAI_API_KEY", "")


def init_session():
    defaults = {
        "stage": "setup",
        "vectorstore": None,
        "resume_text": "",
        "jd": "",
        "questions": [],
        "current_q_idx": 0,
        "answers_for_current_q": 0,
        "messages": [],
        "pdf_loaded": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_session():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


init_session()

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")

    if not API_KEY:
        st.error(".env 파일에 OPENAI_API_KEY가 설정되지 않았습니다.")
        st.stop()

    # 저장된 인덱스 자동 로드
    if not st.session_state.pdf_loaded:
        saved = load_vectorstore(API_KEY)
        if saved:
            st.session_state.vectorstore, st.session_state.resume_text = saved
            st.session_state.pdf_loaded = True

    if st.session_state.pdf_loaded:
        st.success(f"이력서 로드됨 ({len(st.session_state.resume_text):,}자)")
    else:
        st.error("이력서 인덱스가 없습니다.\n\n`python3 setup_resume.py 이력서.pdf` 를 먼저 실행하세요.")
        st.stop()

    st.divider()
    if st.button("🔄 처음부터 다시"):
        reset_session()


# ── 탭 구성 ───────────────────────────────────────────────────────────────────
tab_lab, tab_interview = st.tabs(["🔬 RAG 실험실", "🎙️ 면접 연습"])


# ════════════════════════════════════════════════════════════════════
# TAB 1: 면접 연습
# ════════════════════════════════════════════════════════════════════
with tab_interview:
    st.title("AI 면접관")
    st.caption("이력서 + JD 기반 맞춤형 모의 면접")

    if st.session_state.stage == "setup":
        col1, col2 = st.columns(2)
        with col1:
            st.info("**1단계**\n\n왼쪽 사이드바에서 이력서 PDF 업로드")
        with col2:
            st.info("**2단계**\n\n아래 JD 입력 후 면접 시작")

        jd_text = st.text_area(
            "📋 채용공고 (JD) 입력",
            placeholder="채용공고 전체 내용을 붙여넣으세요...",
            height=250,
        )

        can_start = st.session_state.pdf_loaded and jd_text.strip()
        if st.button("🚀 면접 시작", type="primary", disabled=not can_start):
            st.session_state.jd = jd_text.strip()
            with st.spinner("맞춤 면접 질문 생성 중..."):
                context = get_relevant_context(st.session_state.vectorstore, jd_text, k=8)
                questions = generate_questions(jd_text, context, API_KEY)
            st.session_state.questions = questions
            st.session_state.stage = "interviewing"
            st.session_state.current_q_idx = 0
            st.session_state.answers_for_current_q = 0
            st.session_state.messages = []

            opening = (
                "안녕하세요! 오늘 면접에 지원해 주셔서 감사합니다. "
                "이력서와 채용공고를 검토했으며, 총 5개의 질문을 드릴 예정입니다.\n\n"
                f"**[1/5] {questions[0]}**"
            )
            st.session_state.messages.append({"role": "assistant", "content": opening})
            st.rerun()

    elif st.session_state.stage in ("interviewing", "finished"):
        total = len(st.session_state.questions)
        current = min(st.session_state.current_q_idx + 1, total)
        if st.session_state.stage == "interviewing":
            st.progress(current / total, text=f"진행률: {current}/{total} 질문")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if st.session_state.stage == "interviewing":
            user_input = st.chat_input("답변을 입력하세요...")
            if user_input:
                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.answers_for_current_q += 1

                current_q = st.session_state.questions[st.session_state.current_q_idx]
                context = get_relevant_context(st.session_state.vectorstore, current_q, k=5)

                with st.spinner("면접관이 검토 중..."):
                    response = evaluate_answer(
                        question=current_q,
                        answer=user_input,
                        resume_context=context,
                        history=st.session_state.messages,
                        api_key=API_KEY,
                    )

                move_next = (
                    "다음 질문으로 넘어가겠습니다" in response
                    or st.session_state.answers_for_current_q >= MAX_ANSWERS_PER_QUESTION
                )
                st.session_state.messages.append({"role": "assistant", "content": response})

                if move_next:
                    st.session_state.current_q_idx += 1
                    st.session_state.answers_for_current_q = 0

                    if st.session_state.current_q_idx < total:
                        next_q = st.session_state.questions[st.session_state.current_q_idx]
                        idx = st.session_state.current_q_idx + 1
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"**[{idx}/{total}] {next_q}**"}
                        )
                    else:
                        with st.spinner("종합 피드백 생성 중..."):
                            all_context = get_relevant_context(
                                st.session_state.vectorstore, st.session_state.jd, k=8
                            )
                            feedback = generate_final_feedback(
                                jd=st.session_state.jd,
                                resume_context=all_context,
                                history=st.session_state.messages,
                                api_key=API_KEY,
                            )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": "수고하셨습니다! 종합 피드백을 드립니다.\n\n---\n\n" + feedback}
                        )
                        st.session_state.stage = "finished"
                st.rerun()

        elif st.session_state.stage == "finished":
            st.success("면접이 완료되었습니다!")
            transcript = "\n\n".join(
                f"[{'면접관' if m['role'] == 'assistant' else '지원자'}]\n{m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                label="📥 면접 내용 다운로드",
                data=transcript,
                file_name="interview_transcript.txt",
                mime="text/plain",
            )


# ════════════════════════════════════════════════════════════════════
# TAB 2: RAG 실험실
# ════════════════════════════════════════════════════════════════════
with tab_lab:
    st.title("RAG 실험실")
    st.caption("청킹 전략과 검색 파라미터가 검색 품질에 미치는 영향을 직접 실험합니다.")

    if not st.session_state.pdf_loaded:
        st.warning("사이드바에서 이력서 PDF를 먼저 업로드하세요.")
        st.stop()

    # ── 실험 1: 청킹 시각화 ──────────────────────────────────────────
    st.subheader("실험 1. Chunk Size & Overlap — 청킹 전략 비교")
    st.markdown(
        "문서를 어떻게 자르느냐에 따라 검색 가능한 정보의 단위가 달라집니다. "
        "슬라이더를 조절하며 청크 수와 내용이 어떻게 변하는지 관찰하세요."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        chunk_size = st.slider("Chunk Size (글자 수)", min_value=100, max_value=1500, value=400, step=100)
    with col_b:
        overlap = st.slider("Overlap (글자 수)", min_value=0, max_value=300, value=80, step=25)

    if overlap >= chunk_size:
        st.error("Overlap은 Chunk Size보다 작아야 합니다.")
    else:
        chunks = preview_chunks(st.session_state.resume_text, chunk_size, overlap)
        avg_len = sum(c["length"] for c in chunks) / len(chunks) if chunks else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("생성된 청크 수", f"{len(chunks)}개")
        m2.metric("평균 청크 길이", f"{avg_len:.0f}자")
        m3.metric("원본 대비 오버헤드", f"{(sum(c['length'] for c in chunks) / len(st.session_state.resume_text) - 1) * 100:.1f}%")

        st.markdown(
            f"> **해석:** Chunk Size={chunk_size}, Overlap={overlap}으로 이력서가 **{len(chunks)}개** 조각으로 분할됩니다. "
            f"Overlap으로 인해 원본보다 총 텍스트량이 늘어나는 것이 정상입니다 — 이것이 청크 경계 맥락 손실을 방지하는 비용입니다."
        )

        with st.expander(f"청크 미리보기 전체 ({len(chunks)}개)"):
            for c in chunks:
                st.markdown(f"**[청크 {c['index']}]** `{c['length']}자`")
                st.text(c["content"])
                st.divider()

    st.divider()

    # ── 실험 2: 검색 품질 실험 ───────────────────────────────────────
    st.subheader("실험 2. Top-K 조정 — 검색 결과 변화 분석")
    st.markdown(
        "같은 질문이라도 Top-K를 몇으로 설정하느냐에 따라 검색 결과가 달라집니다. "
        "유사도 점수(Score)가 낮을수록 더 유사한 청크입니다 (FAISS L2 거리 기준)."
    )

    query_input = st.text_input(
        "검색 쿼리 입력",
        placeholder="예: 프로젝트 경험, 사용한 기술 스택, 팀 협업 경험...",
    )
    top_k = st.slider("Top-K (검색 결과 수)", min_value=1, max_value=10, value=3, step=1)

    if query_input and st.session_state.vectorstore:
        results = search_with_scores(st.session_state.vectorstore, query_input, k=top_k)

        st.markdown(f"**'{query_input}'** 쿼리로 상위 **{top_k}개** 청크 검색 결과:")

        for r in results:
            score_bar = min(r["score"] / 2.0, 1.0)  # L2 거리를 0~1 범위로 정규화 (시각화용)
            relevance = "높음" if r["score"] < 0.5 else "중간" if r["score"] < 1.0 else "낮음"

            with st.expander(f"**[{r['rank']}위]** 유사도 점수: `{r['score']:.4f}` | 관련도: {relevance} | `{r['length']}자`"):
                st.progress(1 - score_bar, text=f"L2 거리: {r['score']:.4f} (낮을수록 유사)")
                st.text(r["content"])

        st.markdown(
            "> **해석:** Top-K를 늘리면 더 많은 맥락을 LLM에 제공할 수 있지만, "
            "하위 순위 청크의 관련도가 낮으면 오히려 노이즈가 되어 답변 품질을 떨어뜨릴 수 있습니다. "
            "이것이 검색 품질과 답변 품질을 **분리해서** 평가해야 하는 이유입니다."
        )

    st.divider()

    # ── 실험 3: Hallucination 개념 ───────────────────────────────────
    st.subheader("실험 3. Hallucination 발생 원인 분석")
    st.markdown("""
    Hallucination은 LLM이 검색된 맥락에 **없는 내용을 지어내는 현상**입니다.

    **주요 발생 원인:**
    - 관련 청크가 검색되지 않았을 때 (Retrieval 실패)
    - Top-K가 너무 작아 핵심 정보가 누락될 때
    - Chunk Size가 너무 작아 맥락이 잘렸을 때
    - 질문과 관련된 내용이 문서 자체에 없을 때

    **진단 방법 — 검색 실패 vs 답변 실패 분리:**

    | 상황 | 원인 | 해결 방향 |
    |------|------|----------|
    | 올바른 청크가 검색됐지만 답변이 틀림 | Generation 실패 | 프롬프트 개선, 모델 교체 |
    | 관련 청크가 검색 안 됨 | Retrieval 실패 | 청킹 전략, 임베딩 모델, Top-K 조정 |
    | 검색 결과 자체가 빈약 | 데이터 품질 문제 | 문서 전처리, 청크 설계 재검토 |

    > 위 실험 2에서 쿼리를 입력하고 검색된 청크를 직접 눈으로 확인하는 것이 Retrieval 품질 분석의 첫 단계입니다.
    """)
