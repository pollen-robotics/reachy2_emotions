# 🤖 reachy2_emotions

Record, replay, and experiment with expressive emotions on Reachy2!
This package provides CLI tools and utilities to capture synchronized motion and audio, replay them with smooth transitions, and serve emotion playback over a web API.

---

## 🎬 Demo


<div align="center">
  <img src="docs/gifs/cheerful.gif" width="250"/>
  <img src="docs/gifs/disgusted.gif" width="250"/>
  <img src="docs/gifs/curious.gif" width="250"/>
</div>

---

## 🛠 Installation

For regular users:

```bash
pip install .[tools]
```

For development:
```bash
pip install -e .[dev,tools]
```

This enables live editing, linting, testing, and access to all CLI tools.


🖥 CLI Tools

After installation, two commands are available:
### emotion-record

Records Reachy’s joint motions and microphone audio into .json and .wav files.
```bash
emotion-record --ip 192.168.1.42 --filename amazed1 --audio-device "USB Audio Device"
```

Arguments:

    --ip: IP of Reachy (default: localhost)

    --filename: base name for output files

    --freq: recording frequency (default: 100Hz)

    --audio-device: name or ID of the audio input device

    --list-audio-devices: list available audio input devices

    --record-folder: optional override for output folder

### emotion-play

Replays recorded joint trajectories and synchronized audio, with smooth interpolation and idle animations at the end.

```bash
emotion-play --ip 192.168.1.42 --name amazed1
```

Arguments:

    --ip: IP of Reachy

    --name: name of the recording (without extension)

    --audio-device: optional audio output device

    --audio-offset: offset between motion and audio

    --record-folder: folder to load recordings from

    --server: launch a Flask server to accept emotion replay commands

    --flask-port: port for the server (default: 5001)

    --list: list available emotions

    --all-emotions: play all available recordings sequentially

🎛 Tools
### rank.py

Ranks all .wav files in a folder by duration.
```bash
python tools/rank.py
```

### verif.py

Checks that each .json file has a matching .wav, and vice versa.
```bash
python tools/verif.py
```

### trim_all.py

Trims the first N seconds from all .wav files (default: 1.6s).
Used to align audio playback with motion onset after a BIP cue.
```bash
python tools/trim_all.py
```

⚠️ This modifies files in-place.

🧪 Testing & Development

To install dev dependencies:
```bash
pip install -e .[dev,tools]
```

To auto-format code:
```bash
black . --line-length 128
isort .
```

📁 Folder Structure
```
reachy2_emotions/
├── data/                # Emotion recordings (.json + .wav)
├── reachy2_emotions/    # Core source code (record + replay logic)
├── tools/               # Utility scripts (verif, trim, rank, etc.)
├── tests/
├── README.md
├── pyproject.toml
└── LICENSE
```
🧬 Acknowledgements

Inspired by Claire’s early work on demo_events.

Developed by Pollen Robotics to explore expressive, communicative robots using Reachy2.

📢 Contributions

Contributions, ideas, and feedback are welcome!
Feel free to open issues or submit pull requests.
🧾 License

This project is licensed under the terms of the Apache 2.0 licence.