from flask import Flask, request, jsonify
import numpy as np
import torch
import tempfile
import os
import librosa
from moviepy.editor import VideoFileClip

from cnn import AudioCNN
from audio_funcs import get_mfcc_features
from video_funcs import process_video

import ffmpeg
from flask_cors import CORS

# Load your model

# Audio Models
engage_model = AudioCNN()
calm_model = AudioCNN()
excited_model = AudioCNN()
friendly_model = AudioCNN()

engaging_tone_model_state = torch.load("./models/EngagingTone_model", map_location=torch.device("cpu"))
calm_model_state = torch.load("./models/Calm_model", map_location=torch.device("cpu"))
excited_model_sate = torch.load("./models/Excited_model", map_location=torch.device("cpu"))
friendly_model_state = torch.load("./models/Friendly_model", map_location=torch.device("cpu"))

# Camera Models
eye_model = AudioCNN()
# authentic_model = AudioCNN()

eye_contact_model_state = torch.load("./models/Videos_EyeContact_model", map_location=torch.device("cpu"))
authentic_model_state = torch.load("./models/Videos_Authentic_model", map_location=torch.device("cpu"))

eye_model.load_state_dict(eye_contact_model_state)
engage_model.load_state_dict(engaging_tone_model_state)

engage_model.eval() 
eye_model.eval()
# model.eval()



app = Flask(__name__)
CORS(app) 



def extract_audio_from_video(video_path):
    try:
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        (
            ffmpeg
            .input(video_path)
            .output(temp_audio, acodec='pcm_s16le')
            .overwrite_output()  # This avoids the y/N prompt
            .run()
        )
        return temp_audio
    except Exception as e:
        print(f"Error with ffmpeg: {e}")
        return None


def process_audio(audio_path):
    """Process the audio using librosa to extract features."""
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return torch.tensor(mfcc).unsqueeze(0)  # Shape: (1, 13, Time)

@app.route("/predict", methods=["POST"])
def predict():
    video = request.files.get("file")
    if not video:
        return jsonify({"error": "No video file provided"}), 400

    # Save the video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video.read())
        temp_video.close()

        # Extract audio from the video
        temp_audio_path = extract_audio_from_video(temp_video.name)

        # Process the extracted audio to get features
        
        audio_features = get_mfcc_features(temp_audio_path)
        print(temp_video.name)
        video_features = process_video(temp_video.name)
        # if len(video_features) == 0:
        #     return

        video_features = video_features.reshape(video_features.shape[0], video_features.shape[1], -1)


        audio_tensor = torch.tensor(audio_features, dtype=torch.float32).unsqueeze(1)
        video_tensor = torch.tensor(video_features, dtype=torch.float32).unsqueeze(1)
        print(audio_tensor.shape)
        # Run the model to get a prediction
        # print(np.array(audio_tensor.shape)
        with torch.no_grad():
            # Audio
            engage_output = np.array(engage_model(audio_tensor)).mean()
            calm_output = np.array(calm_model(audio_tensor)).mean()
            excited_output = np.array(excited_model(audio_tensor)).mean()
            friendly_output = np.array(friendly_model(audio_tensor)).mean()

            # Visual
            eye_contact_output = np.array(eye_model(video_tensor)).mean()
            # authentic_output = np.array(eye_model(video_tensor)).mean()

        # eye_contact_output = 0
        # Clean up temporary files
        # os.unlink(temp_video.name)
        # os.unlink(temp_audio_path)  # Delete the temporary audio file



        print({"EngagedTone": float(engage_output), "Calmness" : float(calm_output), "Eagerness" : float(excited_output), "Friendliness" : float(friendly_output),"EyeContact": float(eye_contact_output)})
    return jsonify({"EngagedTone": float(engage_output), "Calmness" : float(calm_output), "Eagerness" : float(excited_output), "Friendliness" : float(friendly_output),"EyeContact": float(eye_contact_output)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
