import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components
import os

st.set_page_config(
    page_title="Notebook Report",
    page_icon="📓",
    layout="wide"
)

st.title("📓 Psychology Classification – Full Notebook")
st.subheader("Rendered Jupyter Notebook (Read-Only)")

st.markdown(
    """
    This page displays the **complete Jupyter notebook** used to build,
    evaluate, and explain the fraud detection model.
    
    > 📌 This is a **read-only report view** for transparency and review.
    """
)

st.markdown("---")

NOTEBOOK_PATH = "Introverts_vs_Extroverts.ipynb"

if not os.path.exists(NOTEBOOK_PATH):
    st.error("Notebook file not found.")
    st.stop()

# Load notebook
with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    notebook = nbformat.read(f, as_version=4)

# Convert notebook to HTML
html_exporter = HTMLExporter()
html_exporter.exclude_input_prompt = True
html_exporter.exclude_output_prompt = True

(body, _) = html_exporter.from_notebook_node(notebook)

# Display notebook
components.html(
    body,
    height=1000,
    scrolling=True
)
