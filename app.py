import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import joblib
import os

app_dir = os.path.dirname(__file__)
model_path = "models/emotion_classifier_pipeline.pkl"
file_path = os.path.join(app_dir, model_path)

print("file_path", file_path)

pipe_lr = joblib.load(open(file_path, "rb"))

def predict_emotion(docx):
    return pipe_lr.predict([docx])

def get_prediction_proba(docx):
    return pipe_lr.predict_proba([docx])

emotions_dict = {
    "anger": "Raiva 😠",
    "disgust": "Aversão 😖",
    "fear": "Medo 😨",
    "happy": "Felicidade 😁",
    "joy": "Alegria 😊",
    "neutral": "Neutro 😐",
    "sad": "Triste 😢",
    "sadness": "Tristeza 😥",
    "shame": "Vergonha 😳",
    "surprise": "Surpresa 😲"
}

def main():
    st.title("Emotion Classifier")
    st.subheader("Emotion Classifier")

    with st.form(key='emotion_clf_form'):
        raw_text = st.text_area("Enter your Text:")
        submit_text = st.form_submit_button(label='Send')

    if submit_text:
        st.markdown("""---""")

        # Prediction
        prediction = predict_emotion(raw_text)
        probability = get_prediction_proba(raw_text)

        st.subheader("Original Text")
        st.write(raw_text)

        st.subheader("Prediction")
        emotion = emotions_dict[prediction[0]]
        st.write(emotion)
        st.write("Confidence: {} %".format(np.max(probability) * 100))

        # Visualization
        st.subheader("Prediction Probability")
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        st.write(proba_df)
        proba_df_clean = proba_df.T.reset_index()
        proba_df_clean.columns = ["Emotion", "Probability"]

        fig = alt.Chart(proba_df_clean).mark_bar().encode(
            x='Emotion',
            y='Probability',
            color='Emotion'
        )

        st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()