import streamlit as st
import librosa
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Music Genre Classifier", layout="centered")

st.title("🎵 Music Genre Classification App")
st.markdown("Upload a song and let AI predict the genre 🎧")

# 🔹 PATHS
DATASET_PATH = r"C:\Users\shiva\Downloads\Data\genres_original"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# 🔹 FEATURE EXTRACTION
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # 🔥 improved
        return np.mean(mfcc.T, axis=0)
    except:
        return None

# 🔹 LOAD OR TRAIN MODEL
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.success("⚡ Model loaded instantly!")
else:
    st.warning("⏳ Training model for first time...")

    features = []
    labels = []

    for genre in os.listdir(DATASET_PATH):
        genre_path = os.path.join(DATASET_PATH, genre)

        if not os.path.isdir(genre_path):
            continue

        for file in os.listdir(genre_path)[:20]:  # speed control
            file_path = os.path.join(genre_path, file)

            data = extract_features(file_path)

            if data is not None:
                features.append(data)
                labels.append(genre)

    st.write("✅ Feature extraction done!")

    X = np.array(features)
    y = np.array(labels)

    # 🔥 SCALING
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=200)  # 🔥 improved
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    st.success("✅ Model trained & saved!")
    st.write("🎯 Accuracy:", accuracy)

# 🔹 PREDICTION
def predict_genre(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    mfcc_scaled = scaler.transform([mfcc_scaled])  # 🔥 important

    prediction = model.predict(mfcc_scaled)
    return prediction[0]

# 🔹 UPLOAD UI
st.subheader("🎧 Upload Audio File")

uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Predict Genre 🎯"):
        with st.spinner("Analyzing audio..."):
            prediction = predict_genre("temp.wav")

        st.success(f"🎵 Predicted Genre: {prediction}")

        # 🎨 extra UI
        if prediction == "rock":
            st.info("🤘 Rock vibes detected!")
        elif prediction == "jazz":
            st.info("🎷 Smooth jazz feel!")
        elif prediction == "classical":
            st.info("🎻 Classical music!")