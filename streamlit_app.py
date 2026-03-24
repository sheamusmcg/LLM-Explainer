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
    "Foundations": [
        st.Page("pages/02_neural_networks.py", title="Neural Networks", icon=":material/hub:"),
        st.Page("pages/03_activations.py", title="Activation Functions", icon=":material/show_chart:"),
    ],
    "Understand": [
        st.Page("pages/04_tokenization.py", title="Tokenization", icon=":material/text_fields:"),
        st.Page("pages/05_embeddings.py", title="Embeddings & Position", icon=":material/grid_on:"),
        st.Page("pages/06_attention.py", title="Attention", icon=":material/visibility:"),
        st.Page("pages/07_transformer_block.py", title="Transformer Block", icon=":material/layers:"),
        st.Page("pages/08_output.py", title="Output & Probabilities", icon=":material/bar_chart:"),
    ],
    "Generate": [
        st.Page("pages/09_generation.py", title="Text Generation", icon=":material/auto_awesome:"),
    ],
}

page = st.navigation(pages)
page.run()
