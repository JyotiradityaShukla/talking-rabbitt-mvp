import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import google.generativeai as genai

# =============================
# CONFIG
# =============================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

st.set_page_config(page_title="Talking Rabbitt", page_icon="🐰")

st.title("🐰 Talking Rabbitt")
st.subheader("Conversational Intelligence for Your Data")

st.markdown("Upload a dataset and ask questions about it.")

# =============================
# FILE UPLOAD
# =============================

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.write("Rows:", df.shape[0], "Columns:", df.shape[1])

    st.subheader("Ask a Question")

    question = st.text_input(
        "Example: Which region had the highest revenue?"
    )

    if question:

        # =============================
        # LLM ANALYSIS
        # =============================

        prompt = f"""
You are a data analyst.

Dataset columns:
{list(df.columns)}

Sample data:
{df.head(20).to_string()}

User question:
{question}

1. Answer the question clearly.
2. Suggest what chart would best represent the answer.
3. Respond in short business-friendly language.
"""

        response = model.generate_content(prompt)

        answer = response.text

        st.subheader("AI Insight")

        st.success(answer)

        # =============================
        # AUTOMATIC VISUALIZATION
        # =============================

        st.subheader("Visualization")

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        if len(numeric_cols) > 0 and len(categorical_cols) > 0:

            x_col = categorical_cols[0]
            y_col = numeric_cols[0]

            chart_type = st.selectbox(
                "Select Chart Type",
                ["Bar Chart", "Line Chart"]
            )

            fig, ax = plt.subplots()

            if chart_type == "Bar Chart":
                df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)

            elif chart_type == "Line Chart":
                df.groupby(x_col)[y_col].sum().plot(kind="line", ax=ax)

            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"{y_col} by {x_col}")

            st.pyplot(fig)

        else:
            st.warning("Dataset needs both numeric and categorical columns for visualization.")