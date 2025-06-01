import streamlit as st
import requests

API_BASE = "http://fastapi-app:8000"

def run():
    st.title("📝 회원가입")

    email = st.text_input("이메일")
    name = st.text_input("이름")
    password = st.text_input("비밀번호", type="password")

    if st.button("회원가입"):
        try:
            res = requests.post(f"{API_BASE}/users/signup", json={
                "email": email,
                "name": name,
                "password": password
            })
            res.raise_for_status()
            st.success("🎉 회원가입 완료! 로그인 해주세요.")
        except Exception as e:
            st.error(f"회원가입 실패: {e}")
