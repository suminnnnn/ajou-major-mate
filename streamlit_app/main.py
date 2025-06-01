import streamlit as st
import admin_page, chat_page, signup_page

st.set_page_config(page_title="AjouMajorMate", layout="wide")

page = st.sidebar.selectbox("📄 페이지 선택", ("챗봇", "회원가입", "관리자"))

if page == "챗봇":
    chat_page.run()
elif page == "회원가입":
    signup_page.run()
elif page == "관리자":
    admin_page.run()
