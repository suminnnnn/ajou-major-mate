import streamlit as st
import requests

API_BASE = "http://fastapi-app:8000"

if "access_token" not in st.session_state:
    st.session_state.access_token = None
if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("🔐 로그인")
email = st.sidebar.text_input("이메일")
password = st.sidebar.text_input("비밀번호", type="password")

if st.sidebar.button("로그인"):
    try:
        res = requests.post(f"{API_BASE}/users/login", json={"email": email, "password": password})
        res.raise_for_status()
        st.session_state.access_token = res.json()["access_token"]
        st.sidebar.success("✅ 로그인 성공!")
    except Exception as e:
        st.sidebar.error(f"❌ 로그인 실패: {str(e)}")


st.title("🤖 AjouMajorMate")

if not st.session_state.access_token:
    st.info("좌측 사이드바에서 로그인 해주세요.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("질문을 입력하세요")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    try:
        res = requests.post(
            f"{API_BASE}/chat/chat",
            json={"query": query},
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        res.raise_for_status()
        answer = res.json()["response"]
    except Exception as e:
        answer = f"❌ 오류 발생: {str(e)}"

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
