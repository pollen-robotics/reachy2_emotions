#!/usr/bin/env python3
import json
import time

import requests


def test_interrupt():
    """
    Sends two emotion requests in quick succession to test interrupting
    an ongoing emotion playback. The first request starts one move,
    and after a short delay the second request is sent to interrupt it.

    Make sure your Flask server is running (default: http://localhost:5001)
    and that the emotion names used here match the ones allowed by the server.
    """
    emotion1 = "furious1"
    emotion2 = "loving1"
    url = "http://localhost:5001/play_emotion"
    headers = {"Content-Type": "application/json"}

    payload1 = {"input_text": "First emotion move", "thought_process": "Starting first move", "emotion_name": f"{emotion1}"}
    print("Sending first emotion request...")
    try:
        response1 = requests.post(url, json=payload1, headers=headers)
        print("Response 1:", response1.status_code, response1.json())
    except Exception as e:
        print("Error sending first request:", e)
        return

    # Wait briefly to let the first move start
    time.sleep(2)

    # Second emotion request (should interrupt the first; use another allowed emotion, e.g., "accueillant")
    payload2 = {
        "input_text": "Second emotion move",
        "thought_process": "Interrupting first move",
        "emotion_name": f"{emotion2}",
    }
    print("Sending second emotion request to interrupt the first...")
    try:
        response2 = requests.post(url, json=payload2, headers=headers)
        print("Response 2:", response2.status_code, response2.json())
    except Exception as e:
        print("Error sending second request:", e)


if __name__ == "__main__":
    test_interrupt()
