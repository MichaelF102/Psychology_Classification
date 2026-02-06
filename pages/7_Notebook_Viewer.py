import streamlit as st
import json
import base64

def show_notebook():
    st.title("📓 Project Notebook Viewer")
    st.markdown("This page renders the original analysis notebook directly within the app.")

    # --- 1. Load the Notebook ---
    try:
        with open("Introverts_vs_Extroverts.ipynb", "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except FileNotFoundError:
        st.error("⚠️ The file 'Introverts_vs_Extroverts.ipynb' was not found in the directory.")
        return

    # --- 2. Iterate and Render Cells ---
    for i, cell in enumerate(notebook['cells']):
        
        # --- Handle MARKDOWN Cells ---
        if cell['cell_type'] == 'markdown':
            content = "".join(cell['source'])
            st.markdown(content)
        
        # --- Handle CODE Cells ---
        elif cell['cell_type'] == 'code':
            # Display the code itself
            source_code = "".join(cell['source'])
            if source_code.strip(): # Only show if not empty
                with st.expander(f"📦 Show Code (Cell {i})", expanded=False):
                    st.code(source_code, language='python')
            
            # Display Inputs/Outputs (Text or Images)
            if 'outputs' in cell:
                for output in cell['outputs']:
                    
                    # 1. Handle Text Output (print statements)
                    if 'text' in output:
                        st.text("".join(output['text']))
                    
                    # 2. Handle Stream Output (stdout)
                    if 'name' in output and output['name'] == 'stdout':
                         st.text("".join(output['text']))
                         
                    # 3. Handle Rich Data (Images/Graphs)
                    if 'data' in output:
                        # PNG Images (Standard Matplotlib/Seaborn)
                        if 'image/png' in output['data']:
                            image_data = output['data']['image/png']
                            # If it's a list, join it (rare but possible)
                            if isinstance(image_data, list):
                                image_data = "".join(image_data)
                            st.image(base64.b64decode(image_data))
                        
                        # JPEG Images
                        elif 'image/jpeg' in output['data']:
                            image_data = output['data']['image/jpeg']
                            if isinstance(image_data, list):
                                image_data = "".join(image_data)
                            st.image(base64.b64decode(image_data))

    st.success("✅ End of Notebook")

if __name__ == "__main__":
    show_notebook()
