import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def show_cleaning():
    st.title("Tx Data Cleaning & Preprocessing")
    st.markdown("This module handles missing values (Imputation) and prepares the data for analysis.")

    # --- 1. Load Raw Data ---
    @st.cache_data
    def load_raw_data():
        try:
            return pd.read_csv("evi.csv")
        except FileNotFoundError:
            return None

    df_raw = load_raw_data()
    
    if df_raw is None:
        st.error("⚠️ File 'evi.csv' not found.")
        return

    # Create a copy for cleaning
    df_clean = df_raw.copy()

    # Define Columns
    num_cols = df_raw.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
    if 'Personality' in cat_cols: cat_cols.remove('Personality')

    # --- 2. Perform Cleaning (Backend) ---
    # Impute Numerical with Median
    for col in num_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # Impute Categorical with Mode
    for col in cat_cols:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)

    # ==========================================
    #              SIDEBAR CONTROLS
    # ==========================================
   
    
    # --- MOVED: Feature Selector in Sidebar ---
    st.sidebar.subheader(" Inspect Feature")
    # Identify columns that originally had missing values
    cols_with_missing = [col for col in df_raw.columns if df_raw[col].isnull().sum() > 0]
    
    selected_col = None
    if cols_with_missing:
        selected_col = st.sidebar.selectbox(
            "Select Feature to Visualize:", 
            cols_with_missing
        )
    else:
        st.sidebar.success("No missing values in dataset!")

    # ==========================================
    #           MAIN PAGE DASHBOARD
    # ==========================================

    # --- SECTION 1: OVERVIEW ---
    st.header("1. Missing Values Overview")
    
    col1, col2 = st.columns(2)
    
    # BEFORE Chart
    with col1:
        st.subheader("⚠️ Before Cleaning")
        missing_raw = df_raw.isnull().sum().reset_index()
        missing_raw.columns = ['Feature', 'Missing Count']
        missing_raw = missing_raw[missing_raw['Missing Count'] > 0]
        
        if not missing_raw.empty:
            fig_before = px.bar(
                missing_raw, x='Feature', y='Missing Count',
                text_auto=True, title="Count of Null Values",
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig_before, use_container_width=True)
            st.error(f"Total Missing Values: {missing_raw['Missing Count'].sum()}")

    # AFTER Chart
    with col2:
        st.subheader("✅ After Cleaning")
        missing_clean = df_clean.isnull().sum().reset_index()
        missing_clean.columns = ['Feature', 'Missing Count']
        
        # Plot all as 0
        fig_after = px.bar(
            missing_clean, x='Feature', y='Missing Count',
            title="Count of Null Values (Cleaned)",
            color_discrete_sequence=['#00CC96']
        )
        if not missing_raw.empty:
            fig_after.update_layout(yaxis_range=[0, missing_raw['Missing Count'].max()])
            
        st.plotly_chart(fig_after, use_container_width=True)
        st.success(f"Remaining Missing Values: {missing_clean['Missing Count'].sum()}")

    st.divider()

    # --- SECTION 2: DEEP DIVE (Controlled by Sidebar) ---
    if selected_col:
        st.header(f"2. Deep Dive: {selected_col}")
        st.markdown(f"Visualizing how imputation changed the distribution of **{selected_col}**.")
        
        # Determine strategy and value used
        strategy = "Median" if selected_col in num_cols else "Mode"
        fill_value = df_clean[selected_col].mode()[0] if strategy == "Mode" else df_clean[selected_col].median()
        
        c1, c2 = st.columns([2, 1]) # Make chart wider
        
        # --- Visual Comparison ---
        with c1:
            if selected_col in num_cols:
                # Numerical: Histogram Overlay
                fig_overlay = go.Figure()
                fig_overlay.add_trace(go.Histogram(
                    x=df_raw[selected_col], name='Original (with NaNs)',
                    opacity=0.5, marker_color='#EF553B'
                ))
                fig_overlay.add_trace(go.Histogram(
                    x=df_clean[selected_col], name=f'Cleaned (NaNs → {fill_value:.1f})',
                    opacity=0.5, marker_color='#636EFA'
                ))
                fig_overlay.update_layout(barmode='overlay', title=f"Before vs After: {selected_col}")
                st.plotly_chart(fig_overlay, use_container_width=True)
            
            else:
                # Categorical: Grouped Bar
                raw_counts = df_raw[selected_col].value_counts(dropna=False).reset_index()
                raw_counts.columns = [selected_col, 'Count']
                raw_counts['Type'] = 'Original'
                # Fill NaN string for plotting
                raw_counts[selected_col] = raw_counts[selected_col].fillna("Missing (NaN)")
                
                clean_counts = df_clean[selected_col].value_counts().reset_index()
                clean_counts.columns = [selected_col, 'Count']
                clean_counts['Type'] = 'Cleaned'
                
                combined = pd.concat([raw_counts, clean_counts])
                
                fig_cat = px.bar(
                    combined, x=selected_col, y='Count', color='Type',
                    barmode='group', title=f"Changes in Category Counts"
                )
                st.plotly_chart(fig_cat, use_container_width=True)

        # --- Statistical Stats ---
        with c2:
            st.subheader("Impact Stats")
            missing_count = df_raw[selected_col].isnull().sum()
            
            st.metric("Missing Rows Filled", f"{missing_count}")
            st.metric("Imputation Strategy", strategy)
            st.metric("Fill Value", f"{fill_value}")
            
            if selected_col in num_cols:
                mean_before = df_raw[selected_col].mean()
                mean_after = df_clean[selected_col].mean()
                delta = mean_after - mean_before
                st.metric("Mean Shift", f"{mean_after:.2f}", delta=f"{delta:.4f}")

    st.divider()
  

if __name__ == "__main__":
    st.set_page_config(page_title="Data Cleaning", layout="wide")
    show_cleaning()