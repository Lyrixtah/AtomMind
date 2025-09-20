import os
from config import LOG_PATH

class Logger:
    def __init__(self):
        os.makedirs(LOG_PATH, exist_ok=True)

    def log(self, message):
        print(message)
