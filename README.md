# signvoice

Real-time two-way conversation helper between a signer and a speaker, running locally:

- **Talk mode** — American Sign Language (ASL) from the webcam → text → spoken voice.
  Recognizes 250 isolated signs (PopSign vocabulary). An optional LLM turns the signs into a natural sentence.
- **Listen mode** — microphone → text on screen, transcribed locally with Whisper.

Everything is controlled with **hand gestures**: point your index finger at a button and hold it there
for about 1 second. You don't need the keyboard or the mouse, but both still work.

---

## Requirements

- **Python 3.12** (tested with 3.12.10; MediaPipe may not have wheels for newer versions yet)
- A webcam and a microphone
- About 200 MB of disk space for the models. They are downloaded automatically on the first run.
- Windows, macOS or Linux. The voice uses the system's built-in text-to-speech:
  - Windows: works out of the box.
  - macOS: works out of the box (`say`).
  - Linux: install `espeak-ng` and the PortAudio library (see step 3).

---

## Step-by-step setup

### 1. Get the code

Clone the repository, or copy the project folder, and open a terminal inside it:

```bash
git clone <repository-url> signvoice
cd signvoice
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

> If PowerShell refuses to run the activation script, allow scripts for your user once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
> and run the activation command again.

**Windows (cmd):**

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Once activated, the prompt starts with `(.venv)`. Run every following command in this activated terminal.

### 3. Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This installs MediaPipe, OpenCV, ONNX Runtime, NumPy, faster-whisper and sounddevice.

**Linux only:** also install the system packages for audio and voice:

```bash
sudo apt install libportaudio2 espeak-ng
```

### 4. Run it

```bash
python signvoice.py
```

The first run downloads the sign-recognition models into `models/` (about 40 MB). The first time you switch
to listen mode, it also downloads the Whisper speech model (about 140 MB). Later runs work offline,
except for the optional LLM.

---

## How to use

### Talk mode (signs → voice)

1. Sign in front of the camera. Each recognized sign appears as a chip at the bottom of the screen.
2. The sentence preview updates after every new sign.
3. To speak the sentence, **lower your hands** for 1.5 s, or point at **Speak**.
4. To remove a wrong sign, point at the **×** on its chip.

### Listen mode (voice → text)

Point at **Mode** to switch. Whatever the other person says is transcribed and shown in large text.
Sign recognition is paused in this mode.

### History

The sidebar on the right keeps the latest utterances from both modes:

- **You** — sentences you signed.
- **Them** — speech that was heard.

### Buttons

Point at a button and hold until the ring around the cursor fills. **Quit** needs a 2 s hold, to avoid
closing the program by accident.

| Button | Action | Key |
|---|---|---|
| Mode | Switch between talk and listen mode | `M` |
| Speak | Speak the current sentence now | `Space` |
| Undo | Remove the last sign | `Backspace` |
| Clear | Clear all signs (in listen mode: clear the caption) | `C` |
| LLM on/off | Toggle the LLM sentence reconstruction | `L` |
| Fullscreen / Window | Toggle fullscreen | `F` (`Esc` leaves fullscreen) |
| Quit | Close the program | `Q` |

---

## Optional: LLM sentence reconstruction

Without an LLM, the program speaks the signs literally (e.g. "Ice cream thank you."). With an LLM, they become
a natural sentence, and the classifier's alternatives are used to fix likely mistakes.

Any OpenAI-compatible API works. Set these environment variables **before** running the program, in the same
terminal.

**OpenRouter (default endpoint), PowerShell:**

```powershell
$env:LLM_API_KEY = "your-key"
$env:LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free" 
python signvoice.py
```

**macOS / Linux:**

```bash
export LLM_API_KEY="your-key"
export LLM_MODEL="nvidia/nemotron-3.5-super-vl:free" 
python signvoice.py
```
---

## Useful options

Run `python signvoice.py --help` for the full list.

| Option | Default | Description |
|---|---|---|
| `--camera N` | `0` | Webcam index (try `1` if the wrong camera opens) |
| `--fullscreen` | off | Start in fullscreen |
| `--mode talk\|listen` | `talk` | Initial mode |
| `--stt-model NAME` | `base` | Whisper model: `tiny` (fastest), `base`, `small` (more accurate), `medium`... |
| `--stt-lang CODE` | `en` | Spoken language in listen mode (`en`, `pt`, `fr`...). Empty = auto-detect. |
| `--voice-lang CODE` | `en` | Language of the spoken voice |
| `--voice NAME` | (first of the language) | Exact voice name; see `--list-voices` |
| `--voice-rate N` | `0` | Voice speed on Windows (`-10` to `10`) |
| `--no-llm` | off | Never call the LLM |
| `--dwell S` | `1.0` | Seconds to hold over a button before it fires |
| `--silence S` | `1.5` | Seconds with hands down before the sentence is spoken |
| `--no-face` | off | Disable face tracking (faster, slightly less accurate) |
| `--ros-camera` | off | Use the ROS/rosbridge camera instead of a local webcam |
| `--track-person` | off | Track the person's vertical position with the head and torso lift |
| `--track-base` | off | Use the base to keep the person centered horizontally |
| `--base-command-topic TOPIC` | `/face_tracking_vel` | ROS `Twist` topic for automatic base tracking; use `/key_vel` with the current TIAGo mux |
| `--nav-server` | off | Start the Windows phone navigation server |
| `--nav-port PORT` | `8080` | HTTP port used by the phone navigation server |
| `--nav-command-topic TOPIC` | `/phone_vel` | ROS `Twist` topic for manual phone navigation |
| `--infer-every S` | `0.5` | Minimum time in seconds between sign inferences |
| `--no-band` | off | Accept only stable/high-confidence signs; ignore lower-confidence band results |
| `--track-deadzone VALUE` | `0.18` | Vertical face-position margin ignored by head tracking |
| `--swap-hands` | off | Swap left/right hands if recognition is poor |
| `--verbose` | off | Print the top-3 guesses of every inference |

---

## Troubleshooting

- **"Could not open webcam 0"**: another app is using the camera, or the camera has another index.
  Try `--camera 1`.
- **No voice, or the wrong language**: run `python signvoice.py --list-voices` and pick one with
  `--voice "Exact Name"`. On Windows, extra voices are installed in
  *Settings → Time & language → Speech*.
- **Listen mode transcribes poorly**: use a larger model, e.g. `--stt-model small`, and set the right
  language with `--stt-lang`.
- **The gesture cursor is hard to control**: keep your hand at chest height with only the index finger
  extended. The cursor appears only while you are pointing.
- **The models failed to download**: check your internet connection and run the program again. A partial
  download is discarded automatically.

---

## Recommended TIAGo configuration

The following Windows command uses the ROS camera, vertical head and torso
tracking, horizontal base centering, the phone navigation server, fullscreen
display, slower inference to reduce CPU usage, and stable/high-confidence
signs only:

```powershell
python signvoice.py `
  --ros-camera `
  --track-person `
  --track-base `
  --base-command-topic /key_vel `
  --nav-server `
  --fullscreen `
  --infer-every 0.8 `
  --no-band
```

Parameter details:

- `--ros-camera`: receives the camera stream from ROS through rosbridge.
- `--track-person`: tracks the face vertically with the head and torso lift.
- `--track-base`: makes small angular corrections with the mobile base to
  center the person horizontally.
- `--base-command-topic /key_vel`: publishes automatic base corrections on
  `/key_vel`, which is already accepted by the TIAGo `twist_mux`.
- `--nav-server`: starts the phone navigation server on the Windows PC.
- `--fullscreen`: starts the camera window in fullscreen mode.
- `--infer-every 0.8`: waits at least 0.8 seconds between sign inferences,
  reducing processing load while keeping the camera loop active.
- `--no-band`: accepts only stable signs above the higher confidence threshold;
  lower-confidence `band` results are ignored.

The phone opens the navigation page at `http://<PC-IP>:8080`. Use
`NAVIGATION: ON` for manual navigation through `/phone_vel`, or
`NAVIGATION: OFF` to disable the manual topic and let the automatic base
tracking use `/key_vel`.

## Credits

- Sign recognition model and pipeline contract: [signchat](https://github.com/nlevites/signchat) (MIT),
  trained on the Kaggle *Isolated Sign Language Recognition* (PopSign) dataset.
- Landmarks: [MediaPipe](https://developers.google.com/mediapipe).
- Speech-to-text: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (OpenAI Whisper models).
