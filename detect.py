from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import os



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