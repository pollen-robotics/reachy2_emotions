# reachy2_emotions
Record and replay emotions on Reachy2!

📦 Installing the Project

For regular users:
```
pip install .[tools]
```

For contributors / development:
```
pip install -e .[dev,tools]
```
This enables live editing and access to dev tools like black, pytest, and documentation generators.



### 🖥 Command Line Tools

After installation, you can use the following commands:

- `emotion-record` — to record a new emotion movement
- `emotion-play` — to replay an existing recording

#### Examples:

```bash
emotion-record --emotion surprised
emotion-play --file uncomfortable1
```

### Record/replay

**Strong** inspiration from Claire's work:
https://github.com/pollen-robotics/demo_events/tree/main



1. **To record a move**: you can execute the record_move.py script. 

It captures the joint positions of arms, grippers and head on a JSON file (/!\ no recording of the mobile base).

Stop the recording with CTRL+C.

There are some parameters that you can tune : 
- ip address (<code>--ip *"put_ip_address"*</code>)
- name of the new recording (<code>--name *"filename_you_want"*</code>)
- frequency of the data capture (<code>--freq *wanted_Hz_rate*</code>)

By default, if you execute the script without specifing them, the ip address is localhost, the filename is "recording_MMdd_hhmm" and the frequency is 100Hz. 

Once the recording is over, the new JSON file will be added to the recordings folder.


2. **To replay the move** : you can execute the replay_move.py script.

You can also tune some parameters : 
- ip address (<code>--ip *"put_ip_address"*</code>) 
- name of the recording you want to replay (<code>--name *"filename_you_want"*</code>)

By default, if you execute the script without specifing them, the ip address is localhost and the file replayed is the last recorded one. 

Note: Interpolation methods are used under the hood to avoid brutal movements at the start of an emotion.


## Tools
### `rank.py`

Ranks all `.wav` files in a folder by duration (descending). Helpful for detecting unusually long or short emotion recordings.

#### Usage:

```bash
python tools/rank.py
```

### `verif.py`

Checks that every `.json` file has a corresponding `.wav` file (and vice versa) in the specified folder. Useful to detect broken or incomplete emotion pairs in your dataset.

#### Usage:

```bash
python3 tools/verif.py
```

### `trim_all.py`

Trims the first seconds of each `.wav` file in a folder. Useful if there's silence or unwanted audio at the beginning of your emotion recordings.
-> Typically used to remove the first ~1.6 seconds of autio before the "BIP" that's used in the record.py to synchronize.

⚠️ This overwrites files in-place!

#### Usage:

```bash
python3 tools/trim_all.py
```
