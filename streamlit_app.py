import streamlit as st
from components.state_manager import init_state

st.set_page_config(
    page_title="Transformer Explainer",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()

pages = {
    "Getting Started": [
        st.Page("pages/01_welcome.py", title="Welcome", icon=":material/school:"),
    ],
    "Understand": [
        st.Page("pages/02_tokenization.py", title="Tokenization", icon=":material/text_fields:"),
        st.Page("pages/03_attention.py", title="Attention", icon=":material/visibility:"),
        st.Page("pages/04_output.py", title="Output & Probabilities", icon=":material/bar_chart:"),
    ],
    "Generate": [
        st.Page("pages/05_generation.py", title="Text Generation", icon=":material/auto_awesome:"),
    ],
}

page = st.navigation(pages)
page.run()
