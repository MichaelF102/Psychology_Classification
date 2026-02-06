import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def show_feature_engineering():
    st.title("⚙️ Feature Engineering & Definitions")
    st.markdown("Transforming raw data into meaningful metrics using domain-specific formulas.")

    # --- 1. Load Data ---
    @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("evi.csv")
            # Basic Imputation to ensure math doesn't fail
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            return df
        except FileNotFoundError:
            return None

    df = load_data()
    if df is None:
        st.error("⚠️ File 'evi.csv' not found.")
        return

    # Create a copy to work on
    df_eng = df.copy()

    # ==========================================
    #       STEP 1: ENCODING (Categorical to Number)
    # ==========================================
    st.header("1. Encoding Variables")
    st.markdown("Converting text labels into numerical values for calculation.")

    
    col1, col2, col3 = st.columns(3)
    
    # 1. Personality Encoding
    with col1:
        st.write("**Target: Personality**")
        df_eng['Personality_encoded'] = df_eng['Personality'].map({'Introvert': 0, 'Extrovert': 1})
        st.code("Introvert: 0\nExtrovert: 1")
        
    # 2. Stage Fear Encoding
    with col2:
        st.write("**Feature: Stage Fear**")
        df_eng['Stage_fear_encoded'] = df_eng['Stage_fear'].map({'No': 0, 'Yes': 1})
        st.code("No: 0\nYes: 1")

    # 3. Drained Encoding
    with col3:
        st.write("**Feature: Drained**")
        df_eng['Drained_after_socializing_encoded'] = df_eng['Drained_after_socializing'].map({'No': 0, 'Yes': 1})
        st.code("No: 0\nYes: 1")
            

    st.divider()

    # ==========================================
    #       STEP 2: CREATING NEW FEATURES
    # ==========================================
    st.header("2. Creating Derived Features")
    st.markdown("Combining multiple columns to create deeper insights.")

    # --- Feature 1: Social Activity Level ---
    st.subheader("🔹 Social Activity Level")
    st.latex(r'''
        \text{Activity Level} = \text{Events} + \text{Going Outside} + \text{Friends Circle}
    ''')
    st.code("df['Social_Activity_Level'] = df['Social_event_attendance'] + df['Going_outside'] + df['Friends_circle_size']")
    df_eng['Social_Activity_Level'] = df_eng['Social_event_attendance'] + df_eng['Going_outside'] + df_eng['Friends_circle_size']
    st.caption("Aggregates all indicators of social busyness into one score.")

    # --- Feature 2: Social Discomfort Index ---
    st.subheader("🔹 Social Discomfort Index")
    st.latex(r'''
        \text{Discomfort Index} = \text{Stage Fear (0/1)} + \text{Drained (0/1)}
    ''')
    st.code("df['Social_Discomfort_Index'] = df['Stage_fear_encoded'] + df['Drained_after_socializing_encoded']")
    df_eng['Social_Discomfort_Index'] = df_eng['Stage_fear_encoded'] + df_eng['Drained_after_socializing_encoded']
    st.caption("A higher score (max 2) indicates higher social anxiety or fatigue.")

    # --- Feature 3: Social Balance ---
    st.subheader("🔹 Social Balance")
    st.latex(r'''
        \text{Social Balance} = \frac{\text{Activity Level}}{\text{Time Spent Alone} + 1}
    ''')
    st.code("df['Social_Balance'] = df['Social_Activity_Level'] / (df['Time_spent_Alone'] + 1)")
    df_eng['Social_Balance'] = df_eng['Social_Activity_Level'] / (df_eng['Time_spent_Alone'] + 1)
    st.caption("Ratio of socialization to solitude. We add +1 to avoid division by zero.")

    # --- Feature 4: Discomfort Efficiency ---
    st.subheader("🔹 Discomfort Efficiency")
    st.latex(r'''
        \text{Efficiency} = \frac{\text{Discomfort Index}}{\text{Activity Level} + 1}
    ''')
    st.code("df['Discomfort_Efficiency'] = df['Social_Discomfort_Index'] / (df['Social_Activity_Level'] + 1)")
    df_eng['Discomfort_Efficiency'] = df_eng['Social_Discomfort_Index'] / (df_eng['Social_Activity_Level'] + 1)
    st.caption("Measures how much 'pain' (discomfort) a person endures per unit of social activity.")

    # --- Feature 5: Posting Impact ---
    st.subheader("🔹 Posting Impact")
    st.latex(r'''
        \text{Impact} = \text{Post Frequency} \times \text{Activity Level}
    ''')
    st.code("df['Posting_Impact'] = df['Post_frequency'] * df['Social_Activity_Level']")
    df_eng['Posting_Impact'] = df_eng['Post_frequency'] * df_eng['Social_Activity_Level']
    st.caption("Correlates online activity with real-world social activity.")

    st.divider()

    # ==========================================
    #       STEP 3: MEANING & DICTIONARY
    # ==========================================
    st.header("3. Feature Dictionary (Interpretations)")
    
    data_dict = pd.DataFrame([
        {
            "Feature": "Social_Activity_Level", 
            "Formula": "Sum(Events, Outside, Friends)", 
            "Interpretation": "High Score = Highly socially active lifestyle."
        },
        {
            "Feature": "Social_Discomfort_Index", 
            "Formula": "Stage Fear + Drained", 
            "Interpretation": "0 = Comfortable, 2 = Highly Anxious/Drained."
        },
        {
            "Feature": "Social_Balance", 
            "Formula": "Activity / (Alone + 1)", 
            "Interpretation": "High Score = Prioritizes socializing over alone time. Low Score = Prioritizes solitude."
        },
        {
            "Feature": "Discomfort_Efficiency", 
            "Formula": "Discomfort / (Activity + 1)", 
            "Interpretation": "High Score = High anxiety despite low activity (Socially sensitive)."
        },
        {
            "Feature": "Posting_Impact", 
            "Formula": "Post Freq * Activity", 
            "Interpretation": "High Score = Active both online and offline (likely Extrovert)."
        }
    ])
    st.table(data_dict)

    # ==========================================
    #       STEP 4: DEFINE X AND y (The Split)
    # ==========================================
    st.header("4. Prepare for Modeling (X and y)")
    st.markdown("Defining the inputs (Features) and the output (Target).")

    # Drop columns as per your snippet
    cols_to_drop = ['id', 'Personality', 'Personality_encoded', 
                    'Stage_fear', 'Drained_after_socializing', 
                    'Stage_fear_encoded', 'Drained_after_socializing_encoded']
    
    # Allow user to verify drops
    final_drop = st.multiselect("Columns to Drop for X:", df_eng.columns, default=cols_to_drop)
    
    X = df_eng.drop(columns=final_drop, errors='ignore')
    y = df_eng['Personality_encoded']

    # Display Shapes (Replicating your print statements)
    col_x, col_y = st.columns(2)
    with col_x:
        st.info(f"**Features (X) Shape:** {X.shape}")
        st.write(X.head())
    with col_y:
        st.info(f"**Target (y) Shape:** {y.shape}")
        st.write(y.head())

    # ==========================================
    #       STEP 5: CORRELATION CHECK
    # ==========================================
    st.header("5. Validate New Features")
    st.markdown("Do these new features actually correlate with Personality?")
    
    # Add target back temporarily for correlation
    corr_check = X.copy()
    corr_check['Personality_Target'] = y
    
    # Calculate correlation
    corr_series = corr_check.corr()['Personality_Target'].sort_values(ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=corr_series.values, y=corr_series.index, palette="coolwarm", ax=ax)
    ax.set_title("Correlation of Engineered Features with Target")
    st.pyplot(fig)
    
    st.info(" **Insight:** Look at 'Social_Activity_Level' and 'Social_Balance'. If they are high positive, they are strong predictors for Extroverts.")

if __name__ == "__main__":
    st.set_page_config(page_title="Feature Engineering", layout="wide")
    show_feature_engineering()
