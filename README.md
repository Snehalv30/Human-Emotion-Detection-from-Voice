# 🎤 Human Emotion Detection From Voice

An AI-based mini project that detects human emotions from speech audio using Machine Learning and Audio Signal Processing. The system analyzes short voice clips and predicts emotions such as **Angry, Calm, Disgust, Fearful, Happy, Neutral, Sad, and Surprised** with probability scores and visual insights.

---

## 📌 Project Overview

Human emotions play a crucial role in communication. This project uses **audio feature extraction** and a **trained ML model** to identify the speaker's emotional state from voice recordings.

The application is built with:

* **Python** for backend & ML
* **Librosa** for audio feature extraction
* **Scikit-learn** for model training
* **Streamlit** for an interactive web interface

---

## ✨ Features

* 🎧 Upload WAV/MP3 audio files
* 🔍 Automatic emotion prediction
* 📊 Emotion probability breakdown
* 📈 Radar chart visualization of emotions
* 🌊 Audio waveform display
* ⚡ Fast and lightweight Streamlit UI
* 🧠 ML-based classification model

---

## 🧠 Emotions Supported

* Angry
* Calm
* Disgust
* Fearful
* Happy
* Neutral
* Sad
* Surprised

---

## 🗂️ Project Structure

```
Human_Emotion_Detection_From_Voice/
│
├── app.py                 # Streamlit application
├── train_model.py         # Model training script
├── models/                # Saved trained model
├── data/                  # Dataset (audio files)
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .venv/                 # Virtual environment (optional)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Human_Emotion_Detection_From_Voice.git
cd Human_Emotion_Detection_From_Voice
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Open your browser and go to:

```
http://localhost:8501
```

---

## 🧪 How It Works

1. User uploads an audio file (WAV/MP3)
2. Audio is preprocessed and trimmed
3. Features extracted:

   * MFCC
   * Chroma
   * Mel Spectrogram
4. ML model predicts emotion probabilities
5. Results are visualized in charts and tables

---

## 📊 Output

* Predicted dominant emotion
* Emotion confidence percentages
* Radar chart visualization
* Audio waveform preview

---

## 📦 Dataset Used

* RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

> Note: Dataset is used for educational purposes only.

---

## 🚀 Future Enhancements

* 🎙️ Live microphone emotion detection
* 🤖 Deep Learning (CNN/LSTM) model
* 🌐 Cloud deployment (Streamlit Cloud / HuggingFace)
* 📁 Emotion history & analytics
* 🧾 Downloadable emotion reports

---

## 🎓 Use Cases

* Academic mini/final year projects
* Human-computer interaction systems
* Mental health analysis (research)
* Call center sentiment analysis
* AI-based voice assistants

---

## 👨‍💻 Author

**Snehal Kedar**
AI & Machine Learning Enthusiast

---

## 📜 License

This project is for **educational and academic use only**.

---

⭐ If you like this project, feel free to star the repository!
