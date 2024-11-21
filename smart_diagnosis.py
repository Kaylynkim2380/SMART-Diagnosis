import streamlit as st
import os
import numpy as np
import librosa

# Import TensorFlow
import tensorflow as tf

# Load the model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('smart_diagnosis_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def smart_diagnosis(audio_file):
    try:
        audio, sr = librosa.load(audio_file, sr=16000)
        prediction = model.predict(audio)
        confidence = float(prediction[0][0])
        
        return {
            "diagnosis": "COPD" if confidence > 0.5 else "NOT COPD",
            "confidence": confidence * 100
        }
    except Exception as e:
        return {"error": str(e)}

st.title("COPD Detection System")
st.write("Upload a breathing sound recording to check for COPD indicators.")

audio_file = st.file_uploader("Upload Audio File", type=['wav'])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Analyze Audio"):
        with st.spinner("Analyzing..."):
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            
            result = smart_diagnosis(temp_path)
            
            os.remove(temp_path)
            
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.subheader("Results:")
                st.write(f"Diagnosis: {result['diagnosis']}")
                st.write(f"Confidence: {result['confidence']:.2f}%")
                st.progress(result['confidence'] / 100)
