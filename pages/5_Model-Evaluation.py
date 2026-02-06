import streamlit as st
import pandas as pd
import plotly.express as px

def show_model_evaluation():
    st.title("🏆 Model Evaluation & Benchmarking")
    st.markdown("Comparing the performance of 5 different algorithms to find the best predictor.")

    # --- 1. Load the Provided Results ---
    data = {
        "Model": ["CatBoost", "Logistic Regression", "LightGBM", "XGBoost", "Random Forest"],
        "Accuracy": [0.968961, 0.968691, 0.968421, 0.968151, 0.964103],
        "Precision": [0.968840, 0.968563, 0.968287, 0.968045, 0.963987],
        "Recall": [0.968961, 0.968691, 0.968421, 0.968151, 0.964103],
        "F1-Score": [0.968880, 0.968604, 0.968327, 0.968085, 0.964034]
    }
    
    df_results = pd.DataFrame(data).set_index("Model")

    # ==========================================
    #              SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.header("Settings")
    
    # MOVED: Selectbox is now here
    selected_metric = st.sidebar.selectbox(
        "Select Metric to Compare:", 
        ["Accuracy", "Precision", "Recall", "F1-Score", "All"],
        index=4 # Default to 'All'
    )
    
    st.sidebar.info("Tip: Select 'All' to see the big picture, or specific metrics to zoom in.")

    # ==========================================
    #           MAIN PAGE DASHBOARD
    # ==========================================

    # --- 2. The Leaderboard ---
    st.header("1. The Leaderboard")
    
    # Highlight the winner (Max value per column)
    st.dataframe(
        df_results.style.format("{:.6f}"),
        use_container_width=True
    )
    
    col_win, col_close = st.columns(2)
    with col_win:
        st.success("🥇 **Winner:** CatBoost (Highest F1-Score: 0.968880)")
    with col_close:
        st.info("🥈 **Runner Up:** Logistic Regression (Very close behind!)")

    st.divider()

    # --- 3. Visual Comparison ---
    st.header(f"2. Visualization: {selected_metric}")
    
    # Prepare data for plotting (Melt)
    df_melt = df_results.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    
    if selected_metric == "All":
        # Grouped Bar Chart
        fig = px.bar(
            df_melt, x="Score", y="Model", color="Metric", barmode="group",
            orientation='h', title="All Metrics Comparison",
            range_x=[0.95, 0.98] # Zoom in to see differences
        )
    else:
        # Single Metric Bar Chart
        subset = df_melt[df_melt["Metric"] == selected_metric].sort_values(by="Score", ascending=True)
        fig = px.bar(
            subset, x="Score", y="Model", orientation='h',
            text="Score", title=f"Comparison of {selected_metric}",
            color="Score", color_continuous_scale="Blues",
            range_x=[0.96, 0.97] # Zoom in strictly
        )
        fig.update_traces(texttemplate='%{text:.5f}', textposition='inside')

    st.plotly_chart(fig, use_container_width=True)

    # --- 4. Interpretation ---
    st.header("3. Interpretation of Results")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("🎯 Accuracy")
        st.metric("Score", "96.8%")
        st.info("**\"Overall Correctness\"**\n\nIf the model predicts the personality of 100 random people, it gets the label (Introvert or Extrovert) exactly right for **97 of them**.")

    with col2:
        st.subheader("💎 Precision")
        st.metric("Score", "96.8%")
        st.info("**\"The Trust Factor\"**\n\nWhen the model flags someone and says *'This person is an Extrovert'*, it is correct 96.8% of the time. It rarely makes the mistake of calling a quiet Introvert an Extrovert.")

    with col3:
        st.subheader("🔍 Recall")
        st.metric("Score", "96.9%")
        st.info("**\"The Coverage\"**\n\nOut of all the **actual Extroverts** that exist in your dataset, the model managed to find 96.9% of them. It didn't let many Extroverts 'slip through' and get mislabeled as Introverts.")

    with col4:
        st.subheader("⚖️ F1-Score")
        st.metric("Score", "96.9%")
        st.info("**\"The Reliability\"**\n\nThis confirms the model isn't cheating. It proves the model is good at finding Extroverts (High Recall) *AND* it is honest about it (High Precision). It's a robust predictor.")
    st.divider()

def show_catboost_importance():
    st.header("🏆 CatBoost Feature Importance")
    st.markdown("The specific features that drove the high accuracy (96.9%) of the winning CatBoost model.")

    # --- 1. Define the Data ---
    data = {
        "Feature": [
            "Social_Balance", 
            "Social_Discomfort_Index", 
            "Posting_Impact", 
            "Going_outside", 
            "Friends_circle_size"
        ],
        "Importance (%)": [
            16.417291, 
            15.023017, 
            13.207929, 
            11.251043, 
            9.834892
        ]
    }
    
    df_imp = pd.DataFrame(data).sort_values(by="Importance (%)", ascending=True)

    # --- 2. Create the Plot ---
    fig = px.bar(
        df_imp, 
        x="Importance (%)", 
        y="Feature", 
        orientation='h',
        text_auto='.2f',
        title="Top 5 Features (CatBoost)",
        color="Importance (%)",
        color_continuous_scale="Viridis" # Green/Blue scale looks professional
    )
    
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

    # --- 3. Interpretation (Why these won?) ---
    st.subheader("💡 Why are these the top predictors?")
    
    
    st.write(f"""
    **1. Social_Balance (16.4%)** *Equation: Social Activity / (Time Alone + 1)* This is the #1 predictor because it captures the **trade-off**. It separates people who socialize *despite* loving alone time vs. those who socialize because they *hate* alone time.
    """)
    
    st.write(f"""
    **2. Social_Discomfort_Index (15.0%)** *Equation: Stage Fear + Drained Status* CatBoost loves categorical data. This feature combines the two strongest negative feelings (Fear + Fatigue), making it a massive "Red Flag" indicator for Introversion.
    """)

    st.write(f"""
    **3. Posting_Impact (13.2%)** *Equation: Post Freq × Activity Level* This bridges the gap between digital and physical life. An Extrovert tends to score high on *both*, amplifying this signal significantly.
    """)
    
    st.write("""
    **4 & 5. Raw Behaviors** *Going Outside & Friends Circle* These are the classic raw metrics. The fact that they are lower than the engineered features proves that **your Feature Engineering added significant value** to the model.
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="Model Evaluation", layout="wide")
    show_model_evaluation()


if __name__ == "__main__":
    show_catboost_importance()
