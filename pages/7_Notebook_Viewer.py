import streamlit as st
from streamlit_option_menu import option_menu # pip install streamlit-option-menu

# Import your page functions
# Ensure these files are in the same folder and have the specific functions defined
# You might need to rename your previous files slightly to match this import structure
# or just copy-paste the functions into one big file.

# Example structure if you kept them as separate files:
# from eda_page import show_eda
# from data_cleaning import show_cleaning
# from feature_engineering import show_feature_engineering
# from model_evaluation import show_model_evaluation
# from live_test import show_live_testing
# from notebook_viewer import show_notebook

st.set_page_config(page_title="Introvert vs Extrovert AI", layout="wide")

# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Home / Notebook", "EDA Analysis", "Data Cleaning", "Feature Eng.", "Model Eval", "Live Prediction"],
        icons=['book', 'bar-chart', 'brush', 'gear', 'trophy', 'person-bounding-box'],
        menu_icon="cast",
        default_index=0,
    )

# Routing Logic
if selected == "Home / Notebook":
    # If you haven't split files yet, just paste the notebook_viewer code here
    import notebook_viewer
    notebook_viewer.show_notebook()

elif selected == "EDA Analysis":
    import eda_page # Assuming you saved the EDA code as eda_page.py
    eda_page.show_eda()

elif selected == "Data Cleaning":
    import data_cleaning
    data_cleaning.show_cleaning()

elif selected == "Feature Eng.":
    import feature_engineering
    feature_engineering.show_feature_engineering()

elif selected == "Model Eval":
    import model_evaluation
    model_evaluation.show_model_evaluation()

elif selected == "Live Prediction":
    import live_test
    live_test.show_live_testing()
