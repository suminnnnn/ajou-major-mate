import streamlit as st
import requests

API_BASE = "http://fastapi-app:8000"
DOMAINS = ["course", "curriculum", "department_intro", "employment_status"]

def run():
    st.title("🛠️ 관리자 도구 - 문서 관리")

    domain = st.selectbox("📚 도메인 선택", DOMAINS)

    col1, col2 = st.columns(2)

    if col1.button("📤 업로드"):
        try:
            res = requests.post(f"{API_BASE}/admin/embed", params={"domain": domain})
            st.success(res.json().get("message"))
        except Exception as e:
            st.error(f"업로드 실패: {e}")

    if col2.button("🗑️ 삭제"):
        try:
            res = requests.delete(f"{API_BASE}/admin/embed", params={"domain": domain})
            st.success(res.json().get("message"))
        except Exception as e:
            st.error(f"삭제 실패: {e}")