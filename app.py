import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# =============================
# HARD CODED API KEY
# =============================

GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# =============================
# STREAMLIT UI
# =============================

st.set_page_config(page_title="Talking Rabbitt", page_icon="🐰")

st.title("🐰 Talking Rabbitt")
st.subheader("Conversational Intelligence for Your Data")

st.markdown("Upload a dataset and ask questions about it.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])

    question = st.text_input(
        "Example: Which region had the highest revenue?"
    )

    if question:

        prompt = f"""
You are a data analyst.

Dataset columns:
{list(df.columns)}

Sample data:
{df.head(20).to_string()}

User question:
{question}

Answer clearly in short business language.
"""

        response = model.generate_content(prompt)

        answer = response.text

        st.subheader("AI Insight")
        st.success(answer)

        st.subheader("Visualization")

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:

            x_col = categorical_cols[0]
            y_col = numeric_cols[0]

            fig, ax = plt.subplots()

            df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} by {x_col}")

            st.pyplot(fig)

        else:
            st.warning("Dataset needs both numeric and categorical columns.")
