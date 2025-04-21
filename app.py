import streamlit as st
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import os

# OpenAI 키는 os.environ["OPENAI_API_KEY"]로 Colab 외부에서 등록했다고 가정합니다.
llm = OpenAI(temperature=0.7)

# Few-shot 예제
examples = [
    {"role": "개발자", "question": "협업 중 문제 해결 경험?", "answer": "Git 충돌 해결하며 커뮤니케이션 개선"},
    {"role": "UX 디자이너", "question": "사용자 피드백 개선 사례?", "answer": "로그인 버튼 위치 변경으로 이탈률 감소"},
    {"role": "기획자", "question": "일정 지연 대처 경험?", "answer": "범위 조정과 외주 커뮤니케이션으로 일정 맞춤"},
]

example_prompt = PromptTemplate(
    input_variables=["role", "question", "answer"],
    template="직무: {role}\n질문: {question}\n답변: {answer}\n"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="직무: {input_role}\n질문:",
    input_variables=["input_role"]
)

response_schemas = [
    ResponseSchema(name="question", description="면접 질문"),
    ResponseSchema(name="answer", description="면접 답변"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

def generate_interview_qa(role):
    full_prompt = prompt.format(input_role=role) + "\n" + output_parser.get_format_instructions()
    response = llm(full_prompt)
    return output_parser.parse(response)

# --------- UI 꾸미기 ---------

# 페이지 설정
st.set_page_config(page_title="AI 면접 생성기", page_icon="🎤", layout="centered")

# 제목 & 설명
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🎤 AI 면접 질문/답변 생성기</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>직무에 맞는 인터뷰 질문과 답변을 AI가 자동으로 생성해드립니다.</p>", unsafe_allow_html=True)
st.markdown("---")

# 선택 옵션
role_color = {
    "개발자": "#2196F3",
    "UX 디자이너": "#9C27B0",
    "기획자": "#4CAF50"
}

job_roles = ["개발자", "UX 디자이너", "기획자"]
selected_role = st.selectbox("🧑‍💼 직무를 선택하세요", job_roles)

# 생성 버튼
if st.button("✨ 질문/답변 생성하기"):
    with st.spinner("AI가 열심히 고민 중입니다..."):
        result = generate_interview_qa(selected_role)

        # 결과 출력
        st.markdown("---")
        st.markdown(
            f"<div style='background-color:{role_color[selected_role]}; padding:20px; border-radius:10px;'>"
            f"<h3 style='color:white;'>📝 질문</h3>"
            f"<p style='color:white; font-size:18px;'>{result['question']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background-color:#eeeeee; padding:20px; border-radius:10px;'>"
            f"<h3>💬 답변</h3>"
            f"<p style='font-size:18px;'>{result['answer']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
