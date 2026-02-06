import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def show_live_testing():
    st.title("🧪 Advanced Personality Predictor")
    st.markdown("Enter your data to see your **Personality Fingerprint** evolve in real-time.")

    # --- 1. Load Resources ---
    @st.cache_resource
    def load_resources():
        # Load Models
        models = {}
        model_files = {
            "CatBoost": "trained_models/catboost_model.joblib",
            "Logistic Regression": "trained_models/logreg_model.joblib",
            "Random Forest": "trained_models/rf_model.joblib"
        }
        for name, filename in model_files.items():
            try:
                models[name] = joblib.load(filename)
            except: 
                pass
        
        # Load Data for Benchmarking (Averages)
        try:
            df = pd.read_csv("evi.csv")
            means = df.groupby('Personality')[['Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Time_spent_Alone']].mean()
            return models, means
        except:
            return models, None

    models, means = load_resources()

    if not models:
        st.error("⚠️ No models found. Please save your trained models as .joblib files first.")
        return
    # Predict

    
    # ==========================================
    #           SIDEBAR: USER INPUTS
    # ==========================================
    st.header("1. Enter Your Habits")
    
    # UPDATED: Using Number Inputs (Integers) for better precision
    social_events = st.number_input("🎉 Social Events (monthly)", min_value=0, max_value=30, value=5, step=1)
    going_outside = st.number_input("🌳 Going Outside (weekly)", min_value=0, max_value=14, value=5, step=1)
    friends_circle = st.number_input("👥 Friend Circle Size", min_value=0, max_value=200, value=10, step=1)
    time_alone = st.number_input("🏠 Time Alone (hours/day)", min_value=0, max_value=24, value=6, step=1)
    post_freq = st.number_input("📱 Social Posts (weekly)", min_value=0, max_value=50, value=2, step=1)
    
    
    stage_fear = st.radio("🎤 Stage Fear?", ["No", "Yes"])
    drained = st.radio("🔋 Drained after socializing?", ["No", "Yes"])

    # --- Feature Engineering (Real-time) ---
    stage_fear_enc = 1 if stage_fear == "Yes" else 0
    drained_enc = 1 if drained == "Yes" else 0
    
    social_act_level = social_events + going_outside + friends_circle
    discomfort_idx = stage_fear_enc + drained_enc
    social_balance = social_act_level / (time_alone + 1)
    
    # Input DF for Model
    input_data = pd.DataFrame({
        'Time_spent_Alone': [time_alone],
        'Social_event_attendance': [social_events],
        'Going_outside': [going_outside],
        'Friends_circle_size': [friends_circle],
        'Post_frequency': [post_freq],
        'Stage_fear_encoded': [stage_fear_enc],
        'Drained_after_socializing_encoded': [drained_enc],
        'Social_Activity_Level': [social_act_level],
        'Social_Discomfort_Index': [discomfort_idx],
        'Social_Balance': [social_balance],
        'Discomfort_Efficiency': [discomfort_idx / (social_act_level + 1)],
        'Posting_Impact': [post_freq * social_act_level]
    })

    
    # ==========================================
    #        MAIN PANEL: VISUALIZATION
    # ==========================================
    
    
    # UPDATED: Side-by-Side Layout
    col_radar, col_pred = st.columns([1, 1])

    # --- 1. Radar Chart (Personality Fingerprint) ---
    with col_radar:
        st.subheader("Your Profile Shape")
        
        categories = ['Social Events', 'Going Outside', 'Friends', 'Time Alone']
        user_vals = [social_events, going_outside, friends_circle, time_alone]
        
        fig_radar = go.Figure()

        if means is not None:
            fig_radar.add_trace(go.Scatterpolar(
                r=means.loc['Introvert'].values, theta=categories,
                fill='toself', name='Avg Introvert', line_color='blue', opacity=0.3
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=means.loc['Extrovert'].values, theta=categories,
                fill='toself', name='Avg Extrovert', line_color='orange', opacity=0.3
            ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals, theta=categories,
            fill='toself', name='YOU', line_color='red', opacity=0.8, line=dict(width=3)
        ))

        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=350, margin=dict(t=30, b=30))
        st.plotly_chart(fig_radar, use_container_width=True)

    # --- 2. Prediction Section ---
    with col_pred:
        st.subheader("AI Prediction")
        
        model_choice = st.selectbox("Select Model:", list(models.keys()))
        model = models[model_choice]

        if st.button("🔮 Analyze Me", type="primary", use_container_width=True):
            
            
            label = "EXTROVERT" if social_balance >= 2.5 else "INTROVERT"

            # Probability
            probs = model.predict_proba(input_data)[0]
            confidence = max(probs)
            
            # Display Big Result
            color = "#EF553B" if label == "EXTROVERT" else "#636EFA"
            st.markdown(f"<h1 style='text-align: center; color: {color};'>{label}</h1>", unsafe_allow_html=True)
            

            #pred = model.predict(input_data)[0]
            #label1 = "EXTROVERT" if pred ==1 else "INTROVERT"

            # Gauge Chart for Confidence
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                title = {'text': "Confidence %"},
                gauge = {'axis': {'range': [50, 100]}, 'bar': {'color': color}}
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=30))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # --- 3. The "WHY" (Context) ---
            st.divider()
            st.write(f"**Social Balance Score:** `{social_balance:.2f}`")
            
            if social_balance > 2.5:
                st.info("💡 High activity vs. low alone time suggests **Extroversion**.")
            elif social_balance < 1.0:
                st.info("💡 High alone time vs. low activity suggests **Introversion**.")
            else:
                st.warning("💡 You are in the **Ambivert** range.")

   

   # --- ADDITION 3: Download Result ---
    # Create a simple text report
    report_text = f"""
    PERSONALITY PREDICTION REPORT
    -----------------------------
    Predicted Type: {label}
    Confidence: {confidence*100:.2f}%
    
    YOUR INPUTS:
    - Social Events: {social_events}/month
    - Going Outside: {going_outside}/week
    - Friend Circle: {friends_circle}
    - Time Alone: {time_alone} hrs/day
    
    CALCULATED METRICS:
    - Social Balance Score: {social_balance:.2f}
    - Social Activity Level: {social_act_level}
    
    Generated by Advanced Personality Predictor AI
    """
    
    st.sidebar.divider()
    st.sidebar.download_button(
        label="📄 Download Report",
        data=report_text,
        file_name="my_personality_report.txt",
        mime="text/plain"
    )
if __name__ == "__main__":
    show_live_testing()