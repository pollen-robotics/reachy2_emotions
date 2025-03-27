# reachy2_emotions

## Installation
```
pip3 install sounddevice soundfile
```

## Common commands
```
python3 replay_move.py  --audio-offset -2.0

python3 record_move.py
```

## Source code
The list and versions of the repos of the hackathon emotion project are on the emotion branch:
https://github.com/pollen-robotics/docker_reachy2_core/tree/emotions



### Record/replay

**Strong** inspiration from Claire's work:
https://github.com/pollen-robotics/demo_events/tree/main



1. **To record a move**: you can execute the record_move.py script. 

It captures the joint positions of arms, grippers and head on a JSON file (/!\ no recording of the mobile base). 

There are some parameters that you can tune : 
- ip address (<code>--ip *"put_ip_address"*</code>)
- filename of the new recording (<code>--filename *"filename_you_want"*</code>)
- frequency of the data capture (<code>--freq *wanted_Hz_rate*</code>)

By default, if you execute the script without specify them, the ip address is localhost, the filename is "recording_MMdd_hhmm" and the frequency is 100Hz. 

Once the recording is over, the new JSON file will be added to the recordings folder.


2. **To replay the move** : you can execute the replay_move.py script.

You can also tune some parameters : 
- ip address (<code>--ip *"put_ip_address"*</code>) 
- filename of the recording you want to replay (<code>--filename *"filename_you_want"*</code>)

By default, if you execute the script without specify them, the ip address is localhost and the file replayed is the last recorded one. 

> Be careful that Reachy needs to be turned on already. And don't worry, the first pose will be reached with a time proportional to the distance from the current pose (if the robot has a pose very different from the first pose of the recording, it will go slowly to this pose)


## Tools
### `verif.py`

Checks that every `.json` file has a corresponding `.wav` file (and vice versa) in the specified folder. Useful to detect broken or incomplete emotion pairs in your dataset.

#### Usage:

```bash
python3 tools/verif.py
