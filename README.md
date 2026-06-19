# RAG 검색 품질 실험 플랫폼

이력서 기반 RAG(Retrieval-Augmented Generation) 파이프라인을 직접 구축하고,  
청킹 전략·검색 파라미터가 검색 품질에 미치는 영향을 실험하며,  
이력서 내용을 대화형으로 탐색할 수 있는 플랫폼입니다.

---

## 프로젝트 개요

이 프로젝트는 **RAG 파이프라인의 핵심 파라미터를 직접 조작하고 결과를 관찰**하는 실험 환경입니다.

- Chunk Size / Overlap / Top-K 조정에 따른 검색 결과 변화를 시각적으로 비교
- 검색된 청크의 유사도 점수를 기반으로 Retrieval 품질 직접 분석
- Hallucination 발생 원인(Retrieval 실패 vs Generation 실패) 구분 프레임워크 제공
- 이력서 내용을 RAG 기반 멀티턴 채팅으로 자유롭게 탐색

---

## RAG 파이프라인 구조

```
[문서 입력]
    │
    ▼
[PDF 파싱]           pypdf로 텍스트 추출 및 정제
    │
    ▼
[청킹 (Chunking)]    RecursiveCharacterTextSplitter
    │                chunk_size=400, overlap=80 (기본값)
    ▼
[임베딩 (Embedding)] OpenAI text-embedding-3-small
    │                텍스트 → 1536차원 벡터로 변환
    ▼
[벡터 DB 저장]       FAISS 인덱스로 로컬 저장
    │
    ════════════════════════════════
    │ (질문 시)
    ▼
[쿼리 임베딩]        질문을 동일한 벡터 공간으로 변환
    │
    ▼
[유사도 검색]        FAISS L2 거리 기반 Top-K 청크 검색
    │
    ▼
[답변 생성]          검색된 맥락 + 대화 히스토리 + 질문 → GPT-4o → 최종 답변
```

---

## 주요 기능

### 🔬 RAG 실험실 (기본 화면)

**실험 1. 청킹 전략 비교**
- Chunk Size(100~1500자)와 Overlap(0~300자)을 슬라이더로 조정
- 실시간으로 생성되는 청크 수, 평균 길이, 오버헤드 확인
- 청크별 내용 미리보기로 문서 분할 결과 직접 관찰

**실험 2. Top-K 검색 품질 분석**
- 검색 쿼리 입력 후 Top-K(1~10) 조정
- 각 검색 결과의 유사도 점수(L2 거리) 시각화
- 순위별 청크 내용 확인으로 Retrieval 품질 직접 평가

**실험 3. Hallucination 원인 분석**
- 검색 실패(Retrieval 문제)와 답변 실패(Generation 문제) 구분 기준 제시
- 원인별 해결 방향 정리

### 🔍 이력서 Q&A (채팅)

- 이력서 내용을 자유롭게 질문하는 대화형 인터페이스
- 매 질문마다 FAISS에서 관련 청크를 검색해 맥락으로 활용 (RAG)
- 이전 대화 맥락을 유지하여 후속 질문 가능 (멀티턴)
- 대화 초기화 버튼으로 새 세션 시작 가능

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| Web UI | Streamlit |
| LLM | GPT-4o (OpenAI) |
| 임베딩 모델 | text-embedding-3-small (OpenAI) |
| 벡터 DB | FAISS (로컬) |
| PDF 파싱 | pypdf |
| RAG 파이프라인 | LangChain |

---

## 설치 및 실행

### 1. 패키지 설치

```bash
pip3 install -r requirements.txt
```

### 2. 환경변수 설정

```bash
cp .env.example .env
# .env 파일에 OpenAI API 키 입력
```

```
OPENAI_API_KEY=sk-...
```

### 3. 이력서 임베딩 (최초 1회)

```bash
python3 setup_resume.py 이력서.pdf
```

`faiss_index/` 폴더에 벡터 인덱스가 저장됩니다. 이후 실행 시 자동으로 로드되므로 재실행 불필요.

### 4. 앱 실행

```bash
python3 -m streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

---

## 프로젝트 구조

```
RAG_PROJECT/
├── app.py                  # Streamlit 메인 앱
├── setup_resume.py         # 이력서 최초 임베딩 스크립트 (오너 전용)
├── requirements.txt
├── .env.example
├── faiss_index/            # 벡터 인덱스 저장 위치 (.gitignore 제외)
│   ├── index.faiss
│   ├── index.pkl
│   └── resume.txt
└── rag/
    ├── pdf_loader.py       # PDF 텍스트 추출
    ├── embedder.py         # 청킹 → 임베딩 → FAISS 저장/로드
    ├── experimenter.py     # 청킹 미리보기, 유사도 검색
    ├── interviewer.py      # 멀티턴 채팅 응답 생성
    └── prompts.py          # 프롬프트 템플릿
```

---

## 주요 파라미터 설명

| 파라미터 | 설명 | 권장 범위 |
|----------|------|----------|
| Chunk Size | 청크 하나의 최대 글자 수 | 200~800 |
| Overlap | 인접 청크 간 겹치는 글자 수 | Chunk Size의 10~20% |
| Top-K | 검색 결과로 가져올 청크 수 | 3~5 |

> Chunk Size가 작을수록 검색 정밀도는 높아지지만 맥락이 단절될 위험이 있고,  
> 클수록 맥락은 풍부해지지만 검색 노이즈가 증가합니다.

---

## 유의사항

- `faiss_index/`와 이력서 PDF는 개인정보 보호를 위해 `.gitignore`에서 제외됩니다.
- OpenAI API 사용 비용이 발생합니다 (이력서 임베딩 1회 약 $0.001, Q&A 1회 약 $0.01~0.03).
