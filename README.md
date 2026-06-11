# FR3 Voice-Controlled Pick-and-Place

[![ROS 2 Humble](https://img.shields.io/badge/ROS_2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04-orange)](https://releases.ubuntu.com/22.04/)
[![MoveIt 2](https://img.shields.io/badge/MoveIt-2-blueviolet)](https://moveit.ros.org/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

A ROS 2 package (`fr3_delivery_sim`) that enables a **Franka Research 3 (FR3)** robotic arm to perform voice-commanded pick-and-place tasks on physical hardware. The system accepts natural-language spoken commands, detects coloured blocks through an ArUco-calibrated overhead camera, resolves a collision-free placement position, and executes the full trajectory through MoveIt 2 — all in real-time on the physical robot.

---

## Table of Contents

- [Project Background](#project-background)
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Workspace Calibration](#workspace-calibration)
- [Running the System](#running-the-system)
- [System Architecture](#system-architecture)
- [Node Reference](#node-reference)
- [Supported Voice Commands](#supported-voice-commands)
- [Topic Reference](#topic-reference)
- [Troubleshooting](#troubleshooting)

---

## Project Background

This repository is the real-hardware continuation of an earlier Gazebo simulation:

> **Simulation baseline:** [see-yuH/franka_fr3_pick_and_place](https://github.com/see-yuH/franka_fr3_pick_and_place)

The simulation established the core MoveIt 2 motion-planning pipeline and colour-based object detection using an overhead camera in Gazebo. Once that approach was validated, the system was ported to the physical FR3 with the following key changes:

- ArUco marker-based workspace calibration replaced the fixed Gazebo world frame
- A `faster-whisper` VAD pipeline was added for on-device, CPU-only speech recognition
- Physical corner coordinates measured via the Franka Control Interface (FCI) establish the camera-to-robot homography
- LEFT and RIGHT dynamic placement zones replaced fixed drop coordinates
- A TTS feedback-prevention system was added to stop the microphone from re-recognising the robot's own spoken responses
- `real_pick_and_place.py` replaced the simulation executor with a full Franka hardware interface, including gripper homing, Cartesian path planning, and a MoveIt 2 collision scene

---

## Overview

The complete pipeline runs as five cooperating ROS 2 nodes:

1. A voice node continuously listens on the microphone, performs VAD-gated transcription with `faster-whisper`, and publishes the raw text.
2. A parser node applies regex rules to the transcript and emits a structured command token.
3. A vision node detects ArUco markers to define the workspace, segments red, green, and blue blocks by HSV colour, and publishes their robot-frame coordinates and rotation angles.
4. A selector node receives the command, looks up the target block's pose, computes a collision-free drop position in the requested half of the workspace, and triggers the motion executor.
5. A motion node builds and executes pick-and-place trajectories through MoveIt 2's `MoveGroupInterface` on the physical FR3.

---

## Repository Structure

```
fr3_delivery_sim/
├── config/
│   ├── fr3_gripper_controller.yaml              # Gripper ros2_control parameters
│   └── moveit_simple_controller_manager.yaml    # Registers fr3_arm_controller with MoveIt 2
├── launch/
│   └── voice_system.launch.py                   # Launches all five nodes with tuned parameters
├── src/
│   ├── whisper_node.py                          # VAD-based ASR node
│   ├── smart_parser_node.py                     # Rule-based NLP parser
│   ├── aruco_vision_detector.py                 # Vision node — ArUco + HSV detection
│   ├── color_block_detector.py                  # Standalone colour-detection utility
│   ├── color_selector_node.py                   # Zone resolver, command queue, TTS
│   └── real_pick_and_place.py                   # MoveIt 2 motion executor
├── urdf/
│   └── fr3_vision_env.urdf.xacro                # FR3 URDF extended with overhead camera link
├── CMakeLists.txt
└── package.xml
```

---

## Prerequisites

Ensure the following are installed and configured before building.

| Requirement | Version / Notes |
|---|---|
| Operating System | Ubuntu 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| MoveIt 2 | Humble release |
| ros2_control | Humble release |
| libfranka | ≥ 0.13.0 (FR3-compatible) |
| franka_ros2 | Humble branch |
| Python | 3.10 or later |
| espeak | Any version available via apt |

**Hardware required:**
- Franka Research 3 arm
- USB RGB camera mounted overhead (kernel index `1` by default)
- Microphone on ALSA device `plughw:0,0` by default
- Four ArUco markers from dictionary `DICT_4X4_50` (IDs 0–3), printed at a known physical size
- Red, green, and blue cubes or blocks within the robot's workspace

> This package builds on top of the official [franka_ros2](https://github.com/frankarobotics/franka_ros2) workspace. **Complete that full setup first** before proceeding with the steps below.

---

## Installation

### 1. Set up the Franka ROS 2 workspace

Follow the official `franka_ros2` setup guide completely before continuing:

> https://github.com/frankarobotics/franka_ros2

This establishes the base workspace at `~/ros2_ws` with all Franka dependencies, URDF descriptions, and the `franka_fr3_moveit_config` package already built.

### 2. Build libfranka from source

The FR3 requires libfranka version 0.13.0 or later.

```bash
sudo apt install -y build-essential cmake git libpoco-dev libeigen3-dev libfmt-dev
git clone --recursive https://github.com/frankarobotics/libfranka --branch 0.13.3
cd libfranka && mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
sudo cmake --install .
```

### 3. Clone this package into the existing workspace

```bash
cd ~/ros2_ws/src
git clone https://github.com/sanjeetsathiyamoorthy/franka-fr3-voice-pick-and-place.git
```

### 4. Install ROS dependencies

```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 5. Install Python dependencies

**OpenCV with ArUco support:**
```bash
pip install opencv-python opencv-contrib-python numpy scipy
```

> `opencv-contrib-python` is required. The base `opencv-python` package does not include `cv2.aruco`.

**Whisper voice system — install in this order:**
```bash
pip install faster-whisper
pip install SpeechRecognition
pip install pyaudio
pip install ctranslate2==4.4.0
```

> If `pyaudio` fails to build, install the PortAudio headers first:
> ```bash
> sudo apt install -y portaudio19-dev
> ```

**TTS engine:**
```bash
sudo apt install -y espeak
```

**Supporting Python packages:**
```bash
pip install torch transformers
```

### 6. Build the workspace

```bash
cd ~/ros2_ws
colcon build --symlink-install
```

### 7. Source the workspace

```bash
source install/setup.bash
```

> Add this line to your `~/.bashrc` to avoid sourcing manually in every new terminal:
> ```bash
> echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
> ```

### 8. Pre-download the Whisper model

The first launch downloads `small.en` automatically. To do it in advance:

```bash
python3 -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8')"
```

---

## Workspace Calibration

The vision system uses four ArUco markers arranged as the corners of the robot's working rectangle. The mapping from camera pixels to robot base-frame coordinates is established through a one-time manual calibration using the Franka Control Interface (FCI).

### Procedure

**1. Print and place the markers.**
Print markers 0, 1, 2, and 3 from `DICT_4X4_50` at a consistent physical size. Lay them flat so their inner corners define the four corners of the workspace rectangle visible to the overhead camera.

**2. Enable FCI mode.**
Open the Franka Desk interface in a browser (`https://<ROBOT_IP>`), navigate to *Settings → FCI*, enable FCI mode, and unlock the robot joints.

**3. Guide the arm to each inner corner.**
Press the physical enable button on the robot to enter gravity-compensation mode. Manually guide the end-effector to each of the four inner corners of the ArUco rectangle. At each position, read the Cartesian pose:

```bash
ros2 topic echo /franka_robot_state_broadcaster/robot_state --once
```

Record the `O_T_EE` translation `(x, y)` at each corner.

**4. Update the hardcoded coordinates.**
Open `src/aruco_vision_detector.py` and replace `MARKER_INNER_CORNER_ROBOT` with your measured values:

```python
MARKER_INNER_CORNER_ROBOT = {
    # marker_id : np.array([robot_x, robot_y])  — metres, in FR3 base frame
    0: np.array([x0, y0]),
    1: np.array([x1, y1]),
    2: np.array([x2, y2]),
    3: np.array([x3, y3]),
}
```

Also set `CUBE_Z_HEIGHT` to the height of your block faces above the table surface (default `0.030` m).

**5. Verify.**
Run the vision node in isolation and observe the output:

```bash
ros2 run fr3_delivery_sim aruco_vision_detector.py
ros2 topic echo /red_pose
```

Place a red object at a known location and confirm the published `x, y` values match your expectation before running the full system.

### How the mapping works

`cv2.findHomography` is called every frame using the four ArUco inner-corner pixel positions paired against the four hardcoded robot-frame coordinates. The resulting 3×3 transform matrix converts any detected block centroid pixel to a robot-frame `(x, y)` position via `cv2.perspectiveTransform`. The homography is cached for up to 30 frames when markers are briefly occluded.

---

## Running the System

The system requires **two terminals**. Source the workspace in each before running.

### Terminal 1 — Bring up MoveIt 2 and the robot

```bash
ros2 launch franka_fr3_moveit_config moveit.launch.py robot_ip:=<ROBOT_IP>
```

Wait until RViz has fully loaded and the `fr3_arm` planning group is displayed before proceeding. The arm will perform **gripper homing** when `real_pick_and_place` starts in Terminal 2.

### Terminal 2 — Launch the voice and vision pipeline

```bash
ros2 launch fr3_delivery_sim voice_system.launch.py
```

This starts all five nodes simultaneously with production-tuned parameters. On startup, `whisper_node` will print:

```
Calibrating noise floor — keep quiet for 3s...
```

Stay quiet during this period. Once calibration is complete and `color_selector_node` has initialised, the robot will say *"Robot ready"* through the speakers. The system is then ready to accept voice commands.

---

## System Architecture

```
  🎙  Microphone (ALSA plughw:0,0)
        │
        ▼
  [ whisper_node.py ]
    faster-whisper small.en · CPU · INT8
    VAD + dynamic noise calibration
    TTS mute via /tts_cooldown
        │  /voice_raw
        ▼
  [ smart_parser_node.py ]
    Regex rule engine · zero GPU overhead
    Colours: RED GREEN BLUE
    Sides: LEFT / RIGHT
    Special: HOME · STOP · STATUS
        │  /pick_command
        ▼
  [ color_selector_node.py ] ◄─────────────── [ aruco_vision_detector.py ]
    Zone resolver                                 📷 Camera (index 1, 30 fps)
    Collision-free drop search (8 cm gap)         ArUco DICT_4X4_50 · IDs 0–3
    Command queue · espeak TTS                    HSV segmentation (R/G/B)
        │                                         Publishes: /red_pose /green_pose
        │  /voice_pick_target                                /blue_pose /workspace_bounds
        │  /place_target                                     /red_yaw   /green_yaw
        │  /grasp_yaw                                        /blue_yaw
        ▼
  [ real_pick_and_place.py ]
    MoveIt 2 MoveGroupInterface
    Planning group: fr3_arm · Frame: fr3_link0
    Cartesian descend/ascend · Gripper homing
    Collision scene (table + floor)
        │  MoveGroupAction · GetCartesianPath
        ▼
  franka_fr3_moveit_config  (MoveIt 2 + RViz)
        │  FollowJointTrajectory · franka_msgs
        ▼
  Franka Research 3  (<ROBOT_IP>)
```

---

## Node Reference

This section describes every Python script in `src/` — what it does, how it fits into the pipeline, and the key internal design decisions.

---

### `whisper_node.py`

**Role:** Microphone capture, voice activity detection, and on-device speech recognition.

This node is the entry point for all human interaction. It records audio from the ALSA device `plughw:0,0` at 44 100 Hz in 1-second chunks using `arecord`. Before starting the main listen loop, it performs a 3-second background noise measurement, computing the 95th-percentile RMS across calibration chunks. The speech detection threshold is then set to `noise_floor_p95 × 1.3`, which adapts automatically to different room acoustics and microphone gain levels.

Once calibrated, the node enters a VAD loop. Audio chunks are only appended to an utterance buffer when their RMS exceeds the threshold for at least 2 consecutive chunks (`MIN_SPEECH_CHUNKS`). After 3 consecutive silent chunks the buffer is considered a complete utterance. The concatenated audio is resampled from 44 100 Hz to the 16 000 Hz that Whisper requires, then passed to `faster-whisper` (`small.en`, INT8 quantised, CPU inference) with a hotword list covering all pick-and-place vocabulary.

The node applies a post-processing correction dictionary that maps common ASR mishearings to their intended words (e.g. `blew → blue`, `greed → green`, `pig → pick`). Transcripts that match a known TTS output phrase, contain too many repeated tokens (hallucination filter), or are shorter than 2 words are silently discarded.

To prevent the robot's own `espeak` speech from being re-recognised, the node subscribes to `/tts_cooldown` (`std_msgs/Float64`). When a cooldown message arrives, the microphone is muted for that many seconds. It also subscribes to `/robot_state` and pauses listening entirely while the robot is marked `BUSY`, resuming only when the state returns to `HOME`.

**Publishes:** `/voice_raw` (`std_msgs/String`)
**Subscribes:** `/tts_cooldown` (`std_msgs/Float64`), `/robot_state` (`std_msgs/String`)

---

### `smart_parser_node.py`

**Role:** Converts raw transcribed text into structured command tokens.

This node is intentionally lightweight — no ML inference, no GPU, no model loading. It applies a small set of compiled regex patterns to the lowercased transcript from `/voice_raw`.

Colour detection scans for `red`, `green`, or `blue` as whole words. Side detection scans for phrases such as `"on the left"`, `"to the right"`, `"left side"`, and a range of equivalent wordings. If both a colour and a side are found, the output is `PICK_<COLOUR>_<SIDE>` (e.g. `PICK_RED_LEFT`). If only a colour is found, the output is `PICK_<COLOUR>`. The special commands `HOME`, `STOP`, and `STATUS` are matched by their own pattern sets and take priority over colour/side detection.

The node logs a warning for any transcript it cannot match, making it straightforward to identify unsupported phrasings during testing.

**Publishes:** `/pick_command` (`std_msgs/String`)
**Subscribes:** `/voice_raw` (`std_msgs/String`)

---

### `aruco_vision_detector.py`

**Role:** Workspace definition, block detection, pose estimation, and yaw measurement.

This node is the sole source of object pose data for the entire pipeline. It opens the camera at index `1`, runs at 30 fps via a ROS 2 timer, and processes each frame in two stages.

In the first stage it calls `cv2.ArucoDetector.detectMarkers` on a greyscale frame. If all four required markers (IDs 0–3 from `DICT_4X4_50`) are visible, it extracts the pre-configured inner corner pixel of each marker, pairs those pixels with the hardcoded robot-frame `(x, y)` coordinates in `MARKER_INNER_CORNER_ROBOT`, and calls `cv2.findHomography` to compute the 3×3 perspective transform `H`. The homography is cached for up to `H_MAX_AGE = 30` frames when markers are briefly occluded. Dynamic workspace bounds `[xmin, xmax, ymin, ymax]` derived from the corner coordinates are published every frame to `/workspace_bounds`.

In the second stage the frame is converted to HSV and passed through three separate colour segmenters — one per colour. Red uses two HSV ranges (it wraps around the 0/180 boundary). Each segmenter applies morphological opening and dilation to clean the mask, finds contours, rejects those below `MIN_CONTOUR_AREA = 400` pixels, and computes the centroid of the largest remaining contour. Block orientation is measured using `cv2.minAreaRect`, giving a rotation angle that is published as a yaw value in radians.

Each detected block's pixel centroid is transformed to robot coordinates via `cv2.perspectiveTransform(H)` and published as a `geometry_msgs/Point` (z is fixed at `CUBE_Z_HEIGHT`). An interactive HSV tuning window with OpenCV trackbars (`HSV Tuning`, `Green HSV`, `Blue HSV`) is opened at startup, allowing threshold adjustment under live lighting conditions without modifying code.

**Publishes:** `/red_pose`, `/green_pose`, `/blue_pose` (`geometry_msgs/Point`); `/red_yaw`, `/green_yaw`, `/blue_yaw` (`std_msgs/Float64`); `/workspace_bounds` (`std_msgs/Float64MultiArray`); `/detected_block/pixel` (legacy, red only)
**Subscribes:** nothing

---

### `color_block_detector.py`

**Role:** Standalone development and calibration utility.

This script is a self-contained version of the colour detection logic, independent of the main pipeline. It uses the same ArUco homography approach and the same HSV ranges as `aruco_vision_detector.py`, but runs as a single-purpose node with a hardcoded workspace definition. It was used during development to tune HSV thresholds, verify the homography, and test colour segmentation in isolation before integrating it into the full system. It is not launched by `voice_system.launch.py` and does not need to be running during normal operation.

**Publishes:** `/red_pose`, `/green_pose`, `/blue_pose` (`geometry_msgs/Point`)
**Subscribes:** nothing

---

### `color_selector_node.py`

**Role:** Command interpretation, placement planning, command queue management, and TTS feedback.

This node acts as the coordinator between the parsed voice command and the motion executor. When a command arrives on `/pick_command` it is added to a `deque` queue. Commands are processed one at a time; the node marks itself `_busy = True` while a motion is in progress and a 30-second timer (`QUEUE_DELAY_SEC`) releases the lock after the expected motion duration.

For a `PICK_<COLOUR>_<SIDE>` command, the node checks that the workspace bounds have been received and that the requested block's pose is available and lies within those bounds. The workspace is divided at the Y midpoint: `LEFT` targets the half with larger Y values, `RIGHT` the half with smaller Y values. A grid search scans the target half in 2 cm steps, testing each candidate position against all known block positions with an 8 cm clearance radius and a 4 cm edge margin, and selects the first conflict-free spot.

Once a valid drop position is found, the node speaks a short confirmation through `espeak` (e.g. *"Picking red, placing on the left"*) and publishes a calculated cooldown duration to `/tts_cooldown` so the microphone is muted for the duration of that speech. It then publishes the resolved drop coordinate to `/place_target`, the gripper yaw to `/grasp_yaw`, and — after a 150 ms delay to ensure the place target has been received — the pick coordinate to `/voice_pick_target`.

`HOME` clears the queue and publishes to `/go_home`. `STOP` clears the queue and releases the busy lock. `STATUS` speaks the current queue length and busy state.

**Publishes:** `/voice_pick_target`, `/place_target` (`geometry_msgs/Point`); `/go_home` (`std_msgs/Bool`); `/grasp_yaw`, `/tts_cooldown` (`std_msgs/Float64`); `/selector_status` (`std_msgs/String`)
**Subscribes:** `/pick_command`, `/red_pose`, `/green_pose`, `/blue_pose`, `/red_yaw`, `/green_yaw`, `/blue_yaw`, `/workspace_bounds`

---

### `real_pick_and_place.py`

**Role:** Full-stack MoveIt 2 motion executor for the physical FR3.

This is the most complex node in the system. On startup it declares and reads a comprehensive set of ROS 2 parameters (velocity and acceleration scaling, grasp height, gripper force, planning frame, and more), waits for the MoveIt 2 `/move_action` server, the arm's `FollowJointTrajectory` action server, and all three Franka gripper action servers (`grasp`, `move`, `homing`) to become available before proceeding. It then publishes the collision scene — a table box and a floor box — to `/planning_scene` using `franka_fr3_moveit_config`'s planning frame `fr3_link0`.

The node supports four execution modes, all triggered by different topic subscriptions:

- **Full pick-and-place** (`/voice_pick_target` + `/place_target`): the standard voice pipeline mode. Localises the block, moves to a pre-grasp pose above it, performs a Cartesian descend, closes the gripper, lifts, travels to the drop position, opens the gripper, and returns home.
- **Pick-only** (`/pick_only_target`): picks a block and holds it without placing.
- **Place-only** (`/place_only_target`): places a currently held block at a specified position.
- **Go-home** (`/go_home`): moves the arm to the home joint configuration `[0.0, −0.785, 0.0, −2.356, 0.0, 1.571, 0.785]`.

Joint-space moves are sent as `MoveGroupAction` goals with velocity and acceleration scales of `0.15` by default. Cartesian descend and ascend segments are computed with `GetCartesianPath` at a step resolution of 5 mm and executed via `FollowJointTrajectory` after re-timing to a safe Cartesian speed of 0.03 m/s. The gripper yaw received on `/grasp_yaw` is used to rotate the end-effector to align with the block's detected tilt angle before closing.

After every motion the node publishes `HOME` or `BUSY` to `/robot_state`, which `whisper_node` uses to pause listening during execution.

**Publishes:** `/robot_state` (`std_msgs/String`)
**Subscribes:** `/voice_pick_target`, `/pick_only_target`, `/place_only_target`, `/zone_pick_target`, `/place_target` (`geometry_msgs/Point`); `/go_home` (`std_msgs/Bool`); `/grasp_yaw` (`std_msgs/Float64`); `/joint_states` (`sensor_msgs/JointState`)

---

## Supported Voice Commands

The parser matches colour and side independently, so most natural phrasings work. A selection of examples is shown below.

| Spoken phrase | Published command | Action |
|---|---|---|
| `"pick up the red block"` | `PICK_RED` | Pick red (no placement) |
| `"grab red and put it on the left"` | `PICK_RED_LEFT` | Pick red → drop in left zone |
| `"move the blue one to the right"` | `PICK_BLUE_RIGHT` | Pick blue → drop in right zone |
| `"take the green cube to the left side"` | `PICK_GREEN_LEFT` | Pick green → drop in left zone |
| `"go home"` / `"return home"` / `"reset"` | `HOME` | Return arm to home position |
| `"stop"` / `"abort"` / `"cancel"` | `STOP` | Clear queue, halt |
| `"status"` / `"what's the state"` | `STATUS` | Speak queue and busy/idle state |

The LEFT zone covers the half of the workspace with larger Y values in the robot base frame; the RIGHT zone covers the half with smaller Y values. The boundary is the Y midpoint of the ArUco-defined workspace rectangle.

---

## Topic Reference

| Topic | Type | Publisher | Subscribers |
|---|---|---|---|
| `/voice_raw` | `std_msgs/String` | `whisper_node` | `smart_parser_node` |
| `/pick_command` | `std_msgs/String` | `smart_parser_node` | `color_selector_node` |
| `/red_pose` | `geometry_msgs/Point` | `aruco_vision_detector` | `color_selector_node` |
| `/green_pose` | `geometry_msgs/Point` | `aruco_vision_detector` | `color_selector_node` |
| `/blue_pose` | `geometry_msgs/Point` | `aruco_vision_detector` | `color_selector_node` |
| `/red_yaw` | `std_msgs/Float64` | `aruco_vision_detector` | `color_selector_node` |
| `/green_yaw` | `std_msgs/Float64` | `aruco_vision_detector` | `color_selector_node` |
| `/blue_yaw` | `std_msgs/Float64` | `aruco_vision_detector` | `color_selector_node` |
| `/workspace_bounds` | `std_msgs/Float64MultiArray` | `aruco_vision_detector` | `color_selector_node` |
| `/voice_pick_target` | `geometry_msgs/Point` | `color_selector_node` | `real_pick_and_place` |
| `/place_target` | `geometry_msgs/Point` | `color_selector_node` | `real_pick_and_place` |
| `/grasp_yaw` | `std_msgs/Float64` | `color_selector_node` | `real_pick_and_place` |
| `/go_home` | `std_msgs/Bool` | `color_selector_node` | `real_pick_and_place` |
| `/tts_cooldown` | `std_msgs/Float64` | `color_selector_node` | `whisper_node` |
| `/robot_state` | `std_msgs/String` | `real_pick_and_place` | `whisper_node` |
| `/selector_status` | `std_msgs/String` | `color_selector_node` | *(monitoring)* |
| `/detected_block/pixel` | `geometry_msgs/Point` | `aruco_vision_detector` | *(legacy, red only)* |

---

## Troubleshooting

**The arm does not move after a voice command is parsed correctly.**
Confirm that Terminal 1 is running and that MoveIt 2 has fully loaded — `real_pick_and_place` waits for `/move_action` to become available before completing its own startup. Run `ros2 node list` and verify `move_group` appears. Also check that FCI mode is enabled in Franka Desk.

**Voice commands are never recognised or are constantly triggering.**
The noise floor calibration at startup sets the speech threshold for your environment. If the room is loud, the threshold may be set too high. Try adjusting `NOISE_MULTIPLIER` in `whisper_node.py` (default `1.3`) — lower values make the system more sensitive, higher values reduce false triggers.

**The vision node reports `"No ArUco markers detected"` persistently.**
Verify that all four markers are in the camera's field of view and are well-lit. Check that the camera index is correct by running `ls /dev/video*` and updating `CAMERA_INDEX` in `aruco_vision_detector.py` if necessary. Ensure `opencv-contrib-python` is installed — the base `opencv-python` does not include `cv2.aruco`.

**The robot picks the wrong block or misses.**
This usually indicates the HSV thresholds need tuning for your lighting. Use the live trackbar windows (`HSV Tuning`, `Green HSV`, `Blue HSV`) that open when `aruco_vision_detector` starts to adjust the ranges while watching the mask output windows in real time.

**The gripper fails to grasp and the error mentions `franka_msgs`.**
Confirm that `franka_ros2` was built correctly and that `franka_msgs` is on the `PYTHONPATH`. Run `ros2 interface list | grep franka` to verify the action interfaces are available.

**MoveIt 2 consistently fails to find a valid plan.**
The requested pose may be near a kinematic singularity or outside the arm's reachable workspace. Check that the block coordinates published by the vision node fall within `BLOCK_X_MIN/MAX` and `BLOCK_Y_MIN/MAX` in `real_pick_and_place.py`. Increase `planning_time` in the launch file if plans are timing out rather than failing outright.

**`espeak` is not found at runtime.**
Install it with `sudo apt install -y espeak`. The `color_selector_node` catches the `FileNotFoundError` gracefully, so TTS will silently fail rather than crashing the node — but the `/tts_cooldown` mute signal will still be published correctly.

---


<img width="1600" height="999" alt="image" src="https://github.com/user-attachments/assets/c37aab69-6059-433d-b680-76082a960fb1" />



## Authors

**Sanjeet Sathiyamoorthy**
[github.com/sanjeetsathiyamoorthy](https://github.com/sanjeetsathiyamoorthy)

**Adithya Raj**
[github.com/see-yuH](https://github.com/see-yuH)
