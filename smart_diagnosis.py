import streamlit as st
import numpy as np
import tensorflow as tf
import librosa

# Set page title
st.set_page_config(page_title="COPD Diagnosis", page_icon="🫁")

st.title("COPD Detection System")
st.write("Upload a breathing sound recording to check for COPD indicators.")

def process_audio(audio_file):
    try:
        # Load and preprocess audio
        audio, sr = librosa.load(audio_file, sr=16000)
        
        # Pad or truncate to fixed length (80000 samples)
        fixed_length = 80000
        if len(audio) > fixed_length:
            audio = audio[:fixed_length]
        else:
            audio = np.pad(audio, (0, fixed_length - len(audio)), 'constant')
        
        # Reshape for model
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        audio = tf.reshape(audio, (1, 80000, 1))
        
        return audio
    except Exception as e:
        st.error(f"Error processing audio: {str(e)}")
        return None

# File uploader
audio_file = st.file_uploader("Upload Audio File", type=['wav'])

if audio_file is not None:
    st.audio(audio_file)
    
    if st.button("Analyze Audio"):
        try:
            st.info("Processing audio...")
            
            # Load model
            model = tf.keras.models.load_model('smart_diagnosis_model.keras')
            
            # Process audio
            audio = process_audio(audio_file)
            
            if audio is not None:
                # Make prediction
                prediction = model.predict(audio, verbose=0)
                confidence = float(prediction[0][0]) * 100
                diagnosis = "COPD" if confidence > 45 else "NOT COPD"
                
                # Display results
                st.subheader("Results:")
                st.write(f"Diagnosis: {diagnosis}")
                st.write(f"Confidence: {confidence:.2f}%")
                st.progress(confidence / 100)
                
                # Add download button for results
                result = {
                    "diagnosis": diagnosis,
                    "confidence": confidence,
                    "filename": audio_file.name
                }
                st.download_button(
                    "Download Results",
                    data=str(result),
                    file_name="diagnosis_results.txt"
                )
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            st.write("Technical details:", str(e))

st.sidebar.markdown("""
### About
This is a COPD detection system that analyzes breathing sounds.

### Instructions
1. Upload a WAV file of breathing sounds
2. Click 'Analyze Audio'
3. View the results and download if needed
""")
