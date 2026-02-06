import streamlit as st
import nbformat
from nbconvert import HTMLExporter
import streamlit.components.v1 as components

def show_notebook():
    st.title("📓 Project Notebook Analysis")
    st.markdown("This view renders the complete Jupyter Notebook with all outputs and formatting preserved.")

    notebook_path = "Introverts_vs_Extroverts.ipynb"

    try:
        # 1. Read the Notebook file
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook_content = nbformat.read(f, as_version=4)

        # 2. Initialize the HTML Exporter
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic' # 'lab', 'classic', or 'basic'

        # 3. Convert to HTML
        (body, resources) = html_exporter.from_notebook_node(notebook_content)

        # 4. Display in Streamlit
        # We use a scrollable container to hold the notebook
        st.download_button(
            label=" Download Original Notebook",
            data=open(notebook_path, "rb").read(),
            file_name="Introverts_vs_Extroverts.ipynb",
            mime="application/vnd.jupyter"
        )
        
        st.divider()
        
        # Render the HTML
        components.html(body, height=1000, scrolling=True)

    except FileNotFoundError:
        st.error(f"⚠️ Could not find **{notebook_path}**. Please ensure it is in the same directory as this script.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    show_notebook()
