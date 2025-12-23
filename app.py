import streamlit as st
import numpy as np
import librosa
import joblib
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Voice Emotion AI",
    page_icon="🎤",
    layout="wide", # Wider layout for side-by-side analysis
)

# -------------------------
# Helper: Dynamic Color Mapping
# -------------------------
emotion_colors = {
    "angry": "#FF4B4B",
    "calm": "#1ABC9C",
    "disgust": "#A569BD",
    "fearful": "#5D6D7E",
    "happy": "#FACA2B",
    "neutral": "#BDC3C7",
    "sad": "#3498DB",
    "surprised": "#E67E22"
}

# -------------------------
# Load model safely
# -------------------------
@st.cache_resource # Cache the model so it doesn't reload on every click
def load_model():
    model_path = "models/emotion_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()
if model is None:
    st.error("❌ Model file not found.")
    st.stop()

# -------------------------
# Feature extraction
# -------------------------
def extract_features(audio_path, n_mfcc=40, duration=3, offset=0.5):
    y, sr = librosa.load(audio_path, duration=duration, offset=offset)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    features = np.mean(mfccs.T, axis=0)
    return features, y, sr

# -------------------------
# App Header
# -------------------------
st.title("🎤 Human Emotion Detection")
st.markdown("---")

# -------------------------
# Sidebar for Upload
# -------------------------
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload Audio (WAV/MP3)", type=["wav", "mp3"])
    st.info("The AI analyzes the first 3 seconds of the audio to predict the underlying emotion.")

# -------------------------
# Main Interface
# -------------------------
if uploaded_file:
    # Save temp file
    temp_file = "temp_audio.wav"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Processing
    with st.spinner("🧠 AI is listening..."):
        features, signal, sr = extract_features(temp_file)
        prediction = model.predict([features])[0]
        probs = model.predict_proba([features])[0]
        classes = model.classes_
        
        # Get color for predicted emotion
        result_color = emotion_colors.get(prediction, "#4CAF50")

    # Layout: Two Columns
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🎵 Audio Analysis")
        st.audio(uploaded_file)
        
        # Visualization: Waveform
        fig_wave, ax_wave = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(signal, sr=sr, ax=ax_wave, color=result_color)
        ax_wave.set_title("Audio Waveform")
        ax_wave.axis('off')
        st.pyplot(fig_wave)

        # Final Result Card
        st.markdown(f"""
            <div style="background-color:{result_color}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">Predicted: {prediction.upper()}</h2>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Emotion Probability")
        
        # Plotly Radar Chart
        fig = go.Figure(data=go.Scatterpolar(
            r=probs,
            theta=classes,
            fill='toself',
            line_color=result_color
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Breakdown Table
    with st.expander("See Detailed Probability Breakdown"):
        cols = st.columns(len(classes))
        for i, (emo, p) in enumerate(zip(classes, probs)):
            cols[i].metric(label=emo.title(), value=f"{p:.1%}")

else:
    st.warning("Please upload an audio file in the sidebar to begin.")