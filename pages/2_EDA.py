import streamlit as st
import pandas as pd
import plotly.express as px

def show_eda():
    st.title("📊 Complete & Interpreted EDA")
    st.markdown("Understanding the data with automated insights for Numerical, Categorical, and Correlation analysis.")

    # --- Data Loading ---
    @st.cache_data
    def get_clean_data():
        try:
            df = pd.read_csv("evi.csv")
            # Basic Imputation
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object']).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0])
            return df
        except FileNotFoundError:
            return None

    df = get_clean_data()
    if df is None:
        st.error("⚠️ File 'evi.csv' not found.")
        return

    # ==========================================
    #              SIDEBAR CONTROLS
    # ==========================================
    st.sidebar.header("Settings")
    
    # 1. Numerical Selectors
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_num = st.sidebar.selectbox("Select Numerical Feature:", numeric_cols)

    # 2. Categorical Selectors
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c != 'Personality']
    selected_cat = st.sidebar.selectbox("Select Categorical Feature:", cat_cols) if cat_cols else None

    # ==========================================
    #              TABS LAYOUT
    # ==========================================
    tab1, tab2, tab3 = st.tabs(["📈 Numerical Data", "🎭 Categorical Data", "🔥 Correlations"])

    # ==========================================
    #       TAB 1: NUMERICAL ANALYSIS
    # ==========================================
    # ==========================================
    #       TAB 1: NUMERICAL ANALYSIS
    # ==========================================
    with tab1:
        st.header(f"Analyzing: {selected_num}")
        
        # --- Row 1: Average & Distribution ---
        col1, col2 = st.columns(2)

        # Plot 1: Average Comparison (Bar)
        with col1:
            st.subheader("1. Compare Averages")
            avg_df = df.groupby("Personality")[selected_num].mean().reset_index()
            
            fig_avg = px.bar(
                avg_df, x="Personality", y=selected_num, color="Personality",
                text_auto='.2f', title=f"Average {selected_num}",
                color_discrete_map={"Introvert": "#636EFA", "Extrovert": "#EF553B"}
            )
            st.plotly_chart(fig_avg, use_container_width=True)
            
            # Interpretation
            i_mean = avg_df[avg_df['Personality'] == 'Introvert'][selected_num].values[0]
            e_mean = avg_df[avg_df['Personality'] == 'Extrovert'][selected_num].values[0]
            diff = abs(i_mean - e_mean)
            higher_group = "Introverts" if i_mean > e_mean else "Extroverts"
            st.info(f" On average, **{higher_group}** score {diff:.2f} points higher on {selected_num}.")

        # Plot 2: Distribution Spread (Histogram)
        with col2:
            st.subheader("2. Distribution Spread")
            fig_hist = px.histogram(
                df, x=selected_num, color="Personality", 
                barmode="overlay", opacity=0.6,
                title=f"Distribution of {selected_num}"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.info(" Overlapping colors mean similar behavior. Separated colors mean this feature strongly distinguishes the two personalities.")

        st.divider()

        # --- Row 2: Total Sum & Box Plot ---
        col3, col4 = st.columns(2)

        # Plot 3: Total Contribution (Pie Chart)
        with col3:
            st.subheader("3. Share of Total")
            # Calculate Sum
            sum_df = df.groupby("Personality")[selected_num].sum().reset_index()
            
            fig_pie = px.pie(
                sum_df, values=selected_num, names="Personality",
                title=f"Who accounts for more total '{selected_num}'?",
                color="Personality",
                color_discrete_map={"Introvert": "#636EFA", "Extrovert": "#EF553B"},
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

            # Interpretation
            top_contributor = sum_df.sort_values(by=selected_num, ascending=False).iloc[0]
            name = top_contributor['Personality']
            val = top_contributor[selected_num]
            total = sum_df[selected_num].sum()
            pct = (val / total) * 100
            
            st.info(f" **{name}** account for **{pct:.1f}%** of the total cumulative score for this metric.")

        # Plot 4: Box Plot (Median & Outliers)
        with col4:
            st.subheader("4. Median & Outliers")
            fig_box = px.box(
                df, x="Personality", y=selected_num, color="Personality",
                title=f"Box Plot of {selected_num}",
                color_discrete_map={"Introvert": "#636EFA", "Extrovert": "#EF553B"}
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Interpretation
            i_med = df[df['Personality'] == 'Introvert'][selected_num].median()
            e_med = df[df['Personality'] == 'Extrovert'][selected_num].median()
            
            if i_med > e_med:
                st.info(f" The median Introvert ({i_med:.2f}) is higher than the median Extrovert ({e_med:.2f}).")
            else:
                st.info(f" The median Extrovert ({e_med:.2f}) is higher than the median Introvert ({i_med:.2f}).")

    # ==========================================
    #       TAB 2: CATEGORICAL ANALYSIS
    # ==========================================
    # ==========================================
    #       TAB 2: CATEGORICAL ANALYSIS
    # ==========================================
    with tab2:
        if selected_cat:
            st.header(f"Analyzing: {selected_cat}")
            
            # --- Row 1: Split & Counts ---
            c1, c2 = st.columns(2)

            # Plot 1: Percentage Split (Normalized 100% Bar)
            with c1:
                st.subheader("1. Proportional Split")
                # Create a normalized crosstab for percentages
                cross = pd.crosstab(df['Personality'], df[selected_cat], normalize='index') * 100
                cross = cross.reset_index().melt(id_vars='Personality', var_name=selected_cat, value_name='Percentage')
                
                fig_stack = px.bar(
                    cross, x="Percentage", y="Personality", color=selected_cat,
                    orientation='h', text_auto='.1f',
                    title=f"How {selected_cat} splits by Personality"
                )
                st.plotly_chart(fig_stack, use_container_width=True)

                # Interpretation logic
                # Find the most common response for Introverts
                intro_row = cross[cross['Personality'] == 'Introvert']
                if not intro_row.empty:
                    top_resp_intro = intro_row.loc[intro_row['Percentage'].idxmax()][selected_cat]
                    top_val_intro = intro_row['Percentage'].max()
                    st.info(f" {top_val_intro:.1f}% of **Introverts** selected '{top_resp_intro}' for this category.")

            # Plot 2: Raw Counts Grouped
            with c2:
                st.subheader("2. Raw Counts")
                fig_group = px.histogram(
                    df, x=selected_cat, color="Personality", 
                    barmode="group", text_auto=True,
                    title=f"Count of People by {selected_cat}"
                )
                st.plotly_chart(fig_group, use_container_width=True)
                
                st.info(" This chart compares the absolute number of people. Use this to check if one personality type dominates the dataset for a specific answer.")

            st.divider()

            # --- Row 2: Global View & Hierarchy ---
            c3, c4 = st.columns(2)

            # Plot 3: Global Donut Chart
            with c3:
                st.subheader("3. Overall Distribution")
                # Global counts regardless of personality
                global_counts = df[selected_cat].value_counts().reset_index()
                global_counts.columns = [selected_cat, 'Count']
                
                fig_pie = px.pie(
                    global_counts, values='Count', names=selected_cat,
                    title=f"Global Breakdown of {selected_cat}",
                    hole=0.4 # Donut style
                )
                st.plotly_chart(fig_pie, use_container_width=True)

                # Interpretation
                top_global = global_counts.iloc[0]
                st.info(f" Ignoring personality types, the most common answer overall is **{top_global[selected_cat]}** ({top_global['Count']} people).")

            # Plot 4: Sunburst Chart (Hierarchy)
            with c4:
                st.subheader("4. Hierarchy (Sunburst)")
                # Hierarchy: Personality -> Category
                fig_sun = px.sunburst(
                    df, path=['Personality', selected_cat], 
                    title=f"Hierarchy: Personality ➝ {selected_cat}"
                )
                st.plotly_chart(fig_sun, use_container_width=True)
                
                st.info(f" This chart shows the 'part-to-whole' relationship. The inner ring is Personality; the outer ring shows how their choices are distributed.")

        else:
            st.warning("No categorical columns found.")

    # ==========================================
    #       TAB 3: CORRELATION HEATMAP
    # ==========================================
    with tab3:
        st.header("Feature Correlations")
        st.markdown("This map shows how features relate to each other. **Red** = Positive relationship (move together). **Blue** = Negative relationship (move opposite).")

        # Calculate Correlation
        corr_matrix = df[numeric_cols].corr()

        # Plot Heatmap
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto", 
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # Automated Interpretation of Strongest Correlation
        st.subheader("🔍 Top Relationships")
        
        # Unstack matrix, remove self-correlations (1.0), and sort
        corr_unstacked = corr_matrix.abs().unstack()
        corr_unstacked = corr_unstacked[corr_unstacked < 1.0].sort_values(ascending=False)
        
        if not corr_unstacked.empty:
            top_pair = corr_unstacked.index[0]
            val = corr_matrix.loc[top_pair]
            st.info(f" The strongest relationship in the data is between **{top_pair[0]}** and **{top_pair[1]}** (Correlation: {val:.2f}).")
            
            if val > 0:
                st.write(f"This is a **Positive Correlation**: As {top_pair[0]} increases, {top_pair[1]} tends to increase.")
            else:
                st.write(f"This is a **Negative Correlation**: As {top_pair[0]} increases, {top_pair[1]} tends to decrease.")

if __name__ == "__main__":
    st.set_page_config(page_title="Interpreted EDA", layout="wide")
    show_eda()