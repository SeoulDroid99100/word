# flask.py
from flask import Flask
from threading import Thread

flask = Flask(__name__)  # Flask instance named 'flask'

@flask.route('/')
def index():
    return "Flask server is running!"

def run():
    flask.run(host='0.0.0.0', port=8080)

def start_flask():
    thread = Thread(target=run)
    thread.daemon = True
    thread.start()
