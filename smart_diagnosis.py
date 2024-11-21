import streamlit as st
import tensorflow as tf
import numpy as np
import os

# Load the model
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('smart_diagnosis_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

def process_audio(audio_file):
    # Load and process audio using TensorFlow
    audio_binary = tf.io.read_file(audio_file)
    waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    return waveform

def smart_diagnosis(audio_file):
    try:
        # Process audio
        waveform = process_audio(audio_file)
        
        # Make prediction
        prediction = model.predict(waveform)
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
            # Save temporarily
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Process and predict
            result = smart_diagnosis(temp_path)
            
            # Clean up
            os.remove(temp_path)
            
            # Show results
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.subheader("Results:")
                st.write(f"Diagnosis: {result['diagnosis']}")
                st.write(f"Confidence: {result['confidence']:.2f}%")
                st.progress(result['confidence'] / 100)
