"""
이력서를 최초 1회 임베딩하여 faiss_index/에 저장하는 오너 전용 스크립트.
이후 app.py는 이 인덱스를 자동으로 로드하므로 사용자 업로드 불필요.

사용법:
    python3 setup_resume.py 이력서.pdf
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", "")
if not API_KEY:
    print("[오류] .env 파일에 OPENAI_API_KEY를 설정하세요.")
    sys.exit(1)

if len(sys.argv) < 2:
    print("사용법: python3 setup_resume.py 이력서.pdf")
    sys.exit(1)

pdf_path = sys.argv[1]
if not os.path.exists(pdf_path):
    print(f"[오류] 파일을 찾을 수 없습니다: {pdf_path}")
    sys.exit(1)

from rag.pdf_loader import load_pdf
from rag.embedder import create_vectorstore, save_vectorstore

print(f"[1/3] PDF 로드 중: {pdf_path}")
with open(pdf_path, "rb") as f:
    text = load_pdf(f)
print(f"      추출된 텍스트: {len(text):,}자")

print("[2/3] 임베딩 생성 중... (OpenAI API 호출)")
vectorstore = create_vectorstore(text, API_KEY)

print("[3/3] faiss_index/ 에 저장 중...")
save_vectorstore(vectorstore, text)

print("\n완료! 이제 app.py 실행 시 이력서가 자동으로 로드됩니다.")
