
import streamlit as st

def get_state():
    if 'app_state' not in st.session_state:
        st.session_state['app_state'] = {}
    return st.session_state['app_state']
