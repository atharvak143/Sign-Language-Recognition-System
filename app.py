from flask import Flask, jsonify, render_template, redirect, url_for
import subprocess
import speech_recognition as sr
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Run your Python script (run.py)
    subprocess.Popen(['python', 'run.py'])
    return redirect(url_for('index'))
@app.route('/detectgif', methods=['POST'])
def detectgif():
    return render_template('detect.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('aboutproject.html')

@app.route('/detectspeech', methods=['POST'])
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        print("Processing...")

    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        filename = text.lower() + ".mp4"
        video_path = f"/static/assets/{filename}"
        
        if not os.path.exists(f".{video_path}"):
            return None

        return jsonify({"text": text, "video": video_path})
    except Exception as e:
        print(e)
        return jsonify({"text": "Could not recognize speech", "video": "/static/assets/default.mp4"})
if __name__ == '__main__':
    app.run(debug=True)
