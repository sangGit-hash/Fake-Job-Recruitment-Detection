import streamlit as st
import re
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import confusion_matrix

# Load the saved model
with open(r'C:\Users\Ganash\Documents\fake job detection ml\Fake-Job-Predictor-main\Code\fake_job_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

nb_classifier = model_data['nb_classifier']
clf_log = model_data['clf_log']
clf_num = model_data['clf_num']
count_vectorizer = model_data['count_vectorizer']

# Example dictionary: {location: (fake_count, real_count)}
location_stats = {
    "new york": (30, 120),
    "san francisco": (10, 90),
    "remote": (20, 80),
    "los angeles": (25, 100)
}

def calculate_ratio(location):
    location = location.strip().lower()
    if location in location_stats:
        fake, real = location_stats[location]
        return fake / (fake + real) if (fake + real) > 0 else 0.5
    else:
        return 0.5  # default if location not found

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, 
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    ax.set_title(f"Confusion Matrix: {model_name}")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

st.title('🕵️ Fake Job Posting Detection App')

tab1, tab2 = st.tabs(["🔍 Predict Fake Job", "📊 EDA"])

# ---------------------------------
# Tab 1: Prediction
# ---------------------------------
with tab1:
    st.header("Enter Job Posting Details")

    location = st.text_input("Job Location (e.g., New York, Remote, etc.)")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")

    telecommuting = st.selectbox("Telecommuting (Remote Job)", [0, 1])

    DIGIT_BLOCK_THRESHOLD = 25  # >=25 digits across description+requirements => block

def has_alpha(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]", str(text or "")))

def total_digits(text: str) -> int:
    return len(re.findall(r"\d", str(text or "")))

# ---- Replace your current button block with this ----
if st.button("🚀 Predict"):
    desc = str(description or "")
    reqs = str(requirements or "")
    loc  = str(location or "")

    # 1) empty check
    if (not desc.strip()) and (not reqs.strip()):
        st.warning("Please provide Job Description or Job Requirements before predicting.")
    else:
        # 2) block if description is numeric-only (no letters at all)
        if not has_alpha(desc) and desc.strip() != "":
            st.warning("Job Description appears numeric-only. Provide a meaningful textual description.")
        else:
            # 3) block if total digits across description+requirements exceed threshold
            digits_count = total_digits(desc) + total_digits(reqs)
            if digits_count >= DIGIT_BLOCK_THRESHOLD:
                st.warning(f"Input contains {digits_count} digits across Description+Requirements (≥ {DIGIT_BLOCK_THRESHOLD}); please remove phone numbers/IDs and provide textual details.")
            else:
                # --- existing prediction logic unchanged below ---
                combined_text = f"{description} {requirements}"
                character_count = len(combined_text)
                ratio = calculate_ratio(location)

                text_vector = count_vectorizer.transform([combined_text])
                numerical_input = np.array([[telecommuting, ratio, character_count]])

                pred_log = clf_log.predict(text_vector)
                pred_num = clf_num.predict(numerical_input)
                final_pred = 0 if (pred_log[0] == 0 and pred_num[0] == 0) else 1

                if final_pred == 0:
                    st.success("✅ This is likely a **genuine** job posting.")
                else:
                    st.error("⚠️ This is likely a **fraudulent** job posting.")
    # if st.button("🚀 Predict"):
    # # if both description and requirements are empty or whitespace -> ask user to provide input
    #     if (not str(description).strip()) and (not str(requirements).strip()):
    #         st.warning("Please provide Job Description or Job Requirements before predicting.")
    #     else:
    #         combined_text = f"{description} {requirements}"
    #         character_count = len(combined_text)
    #         ratio = calculate_ratio(location)

    #         text_vector = count_vectorizer.transform([combined_text])
    #         numerical_input = np.array([[telecommuting, ratio, character_count]])

    #         pred_log = clf_log.predict(text_vector)
    #         pred_num = clf_num.predict(numerical_input)
    #         final_pred = 0 if (pred_log[0] == 0 and pred_num[0] == 0) else 1

    #     if final_pred == 0:
    #         st.success("✅ This is likely a **genuine** job posting.")
    #     else:
    #         st.error("⚠️ This is likely a **fraudulent** job posting.")

    # if st.button("🚀 Predict"):
    #     combined_text = f"{description} {requirements}"
    #     character_count = len(combined_text)
    #     ratio = calculate_ratio(location)

    #     text_vector = count_vectorizer.transform([combined_text])
    #     numerical_input = np.array([[telecommuting, ratio, character_count]])

    #     pred_log = clf_log.predict(text_vector)
    #     pred_num = clf_num.predict(numerical_input)
    #     final_pred = 0 if (pred_log[0] == 0 and pred_num[0] == 0) else 1

    #     if final_pred == 0:
    #         st.success("✅ This is likely a **genuine** job posting.")
    #     else:
    #         st.error("⚠️ This is likely a **fraudulent** job posting.")

# ---------------------------------
# Tab 2: EDA
# ---------------------------------
with tab2:
    st.header("Upload a Dataset for EDA")

    uploaded_file = st.file_uploader("Upload a cleaned CSV file (like `cleaned_jobs.csv`)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display columns to ensure correct column names
        st.write("📄 Columns of the uploaded file:")
        st.write(df.columns)

        st.write("📄 Preview of Uploaded Data:")
        st.dataframe(df.head())

        st.write("📊 Basic Statistics:")
        st.write(df.describe())

        if 'fraudulent' in df.columns:
            st.subheader("Fraudulent Job Posting Distribution")
            fig1, ax1 = plt.subplots()
            sns.countplot(data=df, x='fraudulent', ax=ax1)
            st.pyplot(fig1)

        if 'character_count' in df.columns:
            st.subheader("Character Count Distribution")
            fig2, ax2 = plt.subplots()
            sns.histplot(df['character_count'], kde=True, ax=ax2)
            st.pyplot(fig2)

        if 'ratio' in df.columns:
            st.subheader("Location Ratio Distribution")
            fig3, ax3 = plt.subplots()
            sns.histplot(df['ratio'], kde=True, ax=ax3)
            st.pyplot(fig3)

        # Job Postings Count by Location (with top 10 locations only)
        if 'location' in df.columns:
            st.subheader("Job Postings Count by Location")
            location_counts = df['location'].value_counts().head(10)  # Top 10 locations
            fig4, ax4 = plt.subplots(figsize=(10, 6))  # Adjust size
            sns.barplot(x=location_counts.index, y=location_counts.values, ax=ax4)
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha="right")  # Rotate labels
            st.pyplot(fig4)

        # Job Postings Count by Industry (with top 10 industries only)
        if 'industry' in df.columns:
            st.subheader("Job Postings Count by Industry")
            industry_counts = df['industry'].value_counts().head(10)  # Top 10 industries
            fig5, ax5 = plt.subplots(figsize=(10, 6))  # Adjust size
            sns.barplot(x=industry_counts.index, y=industry_counts.values, ax=ax5)
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha="right")  # Rotate labels
            st.pyplot(fig5)

        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            st.subheader("Correlation Heatmap")
            corr = numeric_df.corr()
            fig6, ax6 = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax6)
            st.pyplot(fig6)

        # WordCloud Generation (Example)
        # WordClouds for Real and Fake Job Descriptions
       # WordClouds for Real and Fake Job Descriptions (Larger + Lower)
        if 'description' in df.columns and 'fraudulent' in df.columns:
            st.markdown("---")
            st.subheader("📌 Word Clouds for Job Descriptions (Real vs Fake)")
            st.markdown("See what words frequently appear in real vs fake job descriptions:")

            real_desc = ' '.join(df[df['fraudulent'] == 0]['description'].dropna())
            fake_desc = ' '.join(df[df['fraudulent'] == 1]['description'].dropna())

            wordcloud_real = WordCloud(width=1200, height=600, max_words=100, background_color='white').generate(real_desc)
            wordcloud_fake = WordCloud(width=1200, height=600, max_words=100, background_color='black', colormap='Reds').generate(fake_desc)

            st.markdown("### ✅ Real Job Descriptions")
            st.image(wordcloud_real.to_array())

            st.markdown("### ⚠️ Fake Job Descriptions")
            st.image(wordcloud_fake.to_array())
        else:
            st.warning("Required columns ('description', 'fraudulent') not found in dataset.")

        if 'fraudulent' in df.columns and 'description' in df.columns and 'requirements' in df.columns:
            # Prepare the data for prediction
            desc = df['description'].fillna("").astype(str)
            reqs = df['requirements'].fillna("").astype(str)
            combined_texts = desc + " " + reqs
            text_vectors = count_vectorizer.transform(combined_texts)
            numerical_inputs = np.array([
             [
                 int(row.get('telecommuting', 0)),
                 calculate_ratio(str(row.get('location', '')).lower()),
                 len(str(row.get('description', '') if not pd.isna(row.get('description')) else '') +
                str(row.get('requirements', '') if not pd.isna(row.get('requirements')) else ''))
             ]
            for _, row in df.iterrows()
])

            # Get true labels
            y_true = df['fraudulent']

            # Predictions for each model
            pred_nb = nb_classifier.predict(text_vectors)
            pred_log = clf_log.predict(text_vectors)
            pred_num = clf_num.predict(numerical_inputs)
            final_pred = (pred_log == 1) | (pred_num == 1)  # Final prediction

            # Plot confusion matrices for each model
            st.markdown("### Confusion Matrices for Each Model")
            plot_confusion_matrix(y_true, pred_nb, "Naive Bayes")
            plot_confusion_matrix(y_true, pred_log, "Logistic Regression")
            plot_confusion_matrix(y_true, pred_num, "Numerical Model")
            plot_confusion_matrix(y_true, final_pred, "Combined Model")
        else:
            st.warning("Required columns ('fraudulent', 'description', 'requirements') not found in dataset.")