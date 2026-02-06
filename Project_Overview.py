import streamlit as st

def show_overview():
    st.title("🧠 Project Overview: Introverts vs Extroverts")

    st.markdown("Team Members")
    st.code(""" 
    
    ### 1. Michael Fernandes
                
    ### 2. Manav William 
                
    ### 3. Nathan Dsouza
    
    ### 4. Anshul Shashidhar""")
    
    # --- Problem Statement ---
    st.header("1. Problem Statement")
    st.markdown("""
    Understanding human personality is complex. This project aims to **analyze and classify personality types** (Introvert vs. Extrovert) based on behavioral patterns and social preferences.
    
    By examining metrics such as time spent alone, social energy levels, and fear of public speaking, 
    we identify key differentiators between these two personality archetypes.
    """)
    
    st.info("🎯 **Goal:** Analyze behavioral data to distinguish between Introverts and Extroverts.")

    # --- Dataset Description ---
    st.header("2. Dataset Description")
    st.markdown("""
    The dataset contains behavioral responses from individuals, categorized by their personality type.
    Below is a description of the features used in this analysis:
    """)

    # Data Dictionary Table
    data_dict = {
        "Feature Name": [
            "Time_spent_Alone", 
            "Stage_fear", 
            "Social_event_attendance", 
            "Going_outside", 
            "Drained_after_socializing", 
            "Friends_circle_size", 
            "Post_frequency", 
            "Personality"
        ],
        "Description": [
            "Estimated hours spent alone per day/week.",
            "Does the individual have stage fear? (Yes/No).",
            "Frequency of attending social gatherings.",
            "Frequency of going outside for leisure.",
            "Does socializing drain energy? (Yes/No).",
            "Approximate size of the friend circle.",
            "Frequency of posting on social media.",
            "**Target Variable**: The resulting personality type (Introvert/Extrovert)."
        ],
        "Data Type": [
            "Numerical", "Categorical", "Numerical", "Numerical", 
            "Categorical", "Numerical", "Numerical", "Categorical (Target)"
        ]
    }
    
    st.table(data_dict)

    # --- Project Workflow ---
    st.header("3. Analysis Workflow")
    st.markdown("""
    The project follows these standard data science steps:
    1.  **Data Inspection**: Understanding data structure and identifying gaps.
    2.  **Data Cleaning**: Handling missing values (Imputing numerical data with Median, categorical with Mode).
    3.  **Exploratory Data Analysis (EDA)**: Visualizing distributions and correlations.
    4.  **Feature Engineering**: Preparing data for modeling.
    """)

# To run this page individually for testing
if __name__ == "__main__":
    show_overview()
