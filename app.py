# app.py
from flask import Flask, request, jsonify
import subprocess
import os
import signal

app = Flask(__name__)

# Function to run Streamlit app
def run_streamlit():
    command = ["streamlit", "run", "streamlit_app.py"]
    process = subprocess.Popen(command)
    return process

# Route to start the Streamlit app
@app.route('/start_streamlit', methods=['GET'])
def start_streamlit():
    global streamlit_process
    if streamlit_process is None or streamlit_process.poll() is not None:
        streamlit_process = run_streamlit()
        return jsonify({"message": "Streamlit app started"}), 200
    else:
        return jsonify({"message": "Streamlit app already running"}), 200

# Route to stop the Streamlit app
@app.route('/stop_streamlit', methods=['GET'])
def stop_streamlit():
    global streamlit_process
    if streamlit_process is not None:
        os.kill(streamlit_process.pid, signal.SIGTERM)
        streamlit_process = None
        return jsonify({"message": "Streamlit app stopped"}), 200
    else:
        return jsonify({"message": "Streamlit app is not running"}), 200

# Route to get the status of the Streamlit app
@app.route('/status', methods=['GET'])
def status():
    if streamlit_process is None or streamlit_process.poll() is not None:
        return jsonify({"status": "Streamlit app is not running"}), 200
    else:
        return jsonify({"status": "Streamlit app is running"}), 200

if __name__ == "__main__":
    global streamlit_process
    streamlit_process = None
    app.run(port=5000)
