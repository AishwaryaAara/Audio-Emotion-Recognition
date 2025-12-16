from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app) 

model = load_model('best_model.keras')

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
encoder = LabelEncoder()
encoder.fit(emotions)

def extract_mfcc(filename, max_pad_len=100):
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5, sr=44100)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc.T
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    audio_path = 'temp.wav'
    audio_file.save(audio_path)

    mfcc = extract_mfcc(audio_path)
    if mfcc is None:
        return jsonify({'error': 'Failed to process audio file'}), 400

    mfcc = np.expand_dims(mfcc, axis=0)   
    
    try:
        prediction = model.predict(mfcc)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_class])
        emotion = encoder.inverse_transform([predicted_class])[0]
        return jsonify({'emotion': emotion, 'confidence': confidence})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)