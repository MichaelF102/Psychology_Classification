import pandas as pd
import streamlit as st
import io

# Page setup
st.set_page_config(page_title="Data Inspection Tool", layout="wide")
st.title("📊 Data Inspection Dashboard")

# 1. Load the Dataset
# We use cache so the file doesn't reload every time you interact with the app
@st.cache_data
def load_data():
    try:
        return pd.read_csv("evi.csv")
    except FileNotFoundError:
        st.error("File 'evi.csv' not found. Please ensure the file is in the same directory.")
        return None

df = load_data()

if df is not None:
    # 2. Preview the First 5 Rows
    st.subheader("1. Preview Data")
    st.caption("Displays the top 5 rows to understand structure.")
    st.dataframe(df.head())

# 3. Check Data Info (Converted to DataFrame)
    st.subheader("2. Data Info")
    
    # Construct a DataFrame with the info details
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Non-Null Count': df.count().values,
        'Dtype': df.dtypes.astype(str).values
    })
    
    # Display as an interactive table
    # hide_index=True makes it look cleaner by removing the 0,1,2 row numbers
    st.dataframe(info_df, use_container_width=True, hide_index=True)

    # 4. Check Dataset Dimensions
    st.subheader("3. Dataset Dimensions")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])

    # 5. Statistical Summary
    st.subheader("4. Statistical Summary")
    st.caption("Count, mean, std, min, max, and quartiles for numerical columns.")
    st.dataframe(df.describe())

    # 6. Categorical Data Inspection
    st.subheader("5. Categorical Data Inspection")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("**Unique Values in Object Columns:**")
        st.dataframe(df.select_dtypes(include='object').nunique(), use_container_width=True)

    with c2:
        st.write("**Personality Distribution:**")
        # Display as a dataframe
        st.dataframe(df['Personality'].value_counts(), use_container_width=True)
        # Optional: Add a simple chart for better visualization
        st.bar_chart(df['Personality'].value_counts())

    # 7. Check for Duplicates
    st.subheader("6. Data Quality Checks")
    dup_count = df.duplicated().sum()
    if dup_count == 0:
        st.success(f"Duplicate Rows: {dup_count}")
    else:
        st.warning(f"Duplicate Rows: {dup_count}")

    # 8. Check for Missing Values
    st.write("**Missing Values per Column:**")
    missing_data = df.isnull().sum()
    st.dataframe(missing_data)
    
    # Visualizing missing data (Filtering only columns with missing values)
    if missing_data.sum() > 0:
        st.bar_chart(missing_data[missing_data > 0])