#!/usr/bin/env python3
"""
signvoice — ASL (250 isolated signs) -> text -> speech, running locally.

Pipeline (same contract as signchat, github.com/nlevites/signchat, MIT):
  webcam (OpenCV)
    -> MediaPipe Tasks: FaceLandmarker + HandLandmarker + PoseLandmarker
    -> frame (543, 3) in the Kaggle ISLR layout: [Face 468][LHand 21][Pose 33][RHand 21], NaN = not detected
    -> window of up to 48 frames -> asl-signs.onnx (built-in preprocessing) -> 250 logits
    -> admit (stable for 2 inferences) -> sign buffer
    -> end of turn (hands lowered or pointing at the "Speak" button) -> sentence (optional LLM) -> system voice

Listen mode ("Mode" button): microphone -> faster-whisper (local) -> on-screen text.

Usage:  python signvoice.py          (downloads the models on first run)
        python signvoice.py --help
        python signvoice.py --server ws://127.0.0.1:8080/ws/translator   (TIAGo robot: phone remote)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = HERE / "models"

_SIGNCHAT_RAW = (
    "https://raw.githubusercontent.com/nlevites/signchat/main/"
    "signchat-workbench/public/models/asl-signs"
)
_MP = "https://storage.googleapis.com/mediapipe-models"
MODEL_URLS = {
    "asl-signs.onnx": f"{_SIGNCHAT_RAW}/asl-signs.onnx",
    "sign_to_prediction_index_map.json": f"{_SIGNCHAT_RAW}/sign_to_prediction_index_map.json",
    "face_landmarker.task": f"{_MP}/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": f"{_MP}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": f"{_MP}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}

# ---- frame layout (Kaggle ISLR) ----------------------------------------------
N_FACE, N_LHAND, N_POSE, N_RHAND = 468, 21, 33, 21
FACE_OFF = 0
LHAND_OFF = N_FACE                      # 468
POSE_OFF = N_FACE + N_LHAND             # 489
RHAND_OFF = N_FACE + N_LHAND + N_POSE   # 522
N_TOTAL = 543

# compound labels -> English reading (copy of signchat's DICTIONARY_LABELS)
DICTIONARY_LABELS = {
    "WEUS": "we/us", "HESHEIT": "he/she/it", "MINEMY": "my/mine",
    "HAVETO": "have to", "CALLONPHONE": "call on phone", "FRENCHFRIES": "french fries",
    "GLASSWINDOW": "window", "ICECREAM": "ice cream", "TV": "TV",
    "THANKYOU": "thank you", "SHHH": "shush", "OWIE": "owie", "YUCKY": "yucky",
}
# "speakable" version without LLM (picks a single form)
SPOKEN_LABELS = {**DICTIONARY_LABELS, "WEUS": "we", "HESHEIT": "he", "MINEMY": "my"}


def dictionary_label(label: str) -> str:
    return DICTIONARY_LABELS.get(label.upper(), label.lower())


# =============================================================================
# 1. Model download
# =============================================================================
def ensure_models(models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = models_dir / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        print(f"[models] downloading {name} ...", flush=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)
        except Exception as e:  # noqa: BLE001
            tmp.unlink(missing_ok=True)
            raise SystemExit(
                f"Failed to download {name} from {url}: {e}\n"
                f"Download it manually and put it in {models_dir}"
            )
    # sanity check: the ONNX is ~22 MB; a Git LFS pointer would be a few bytes
    if (models_dir / "asl-signs.onnx").stat().st_size < 1_000_000:
        raise SystemExit("asl-signs.onnx looks invalid (< 1 MB). Download it again.")


# =============================================================================
# 2. Frame assembly (543, 3)
# =============================================================================
@dataclass
class HandObs:
    landmarks: list          # 21 objects with .x .y .z
    side: str                # "Left" | "Right" (MediaPipe label)
    score: float


def _write(frame: np.ndarray, off: int, count: int, lms) -> None:
    if not lms:
        return
    n = min(count, len(lms))
    frame[off:off + n] = np.array([[p.x, p.y, p.z] for p in lms[:n]], dtype=np.float32)


def assemble_frame(face, hands: list[HandObs], pose, swap_hands: bool = False) -> np.ndarray:
    """Returns (543, 3) float32; NaN where nothing was detected (do NOT use 0)."""
    frame = np.full((N_TOTAL, 3), np.nan, dtype=np.float32)
    _write(frame, FACE_OFF, N_FACE, face)          # FaceLandmarker gives 478 (with iris) -> use 468
    best: dict[str, HandObs] = {}
    for h in hands:
        side = h.side
        if swap_hands:
            side = {"Left": "Right", "Right": "Left"}.get(side, side)
        if side in ("Left", "Right") and (side not in best or h.score > best[side].score):
            best[side] = HandObs(h.landmarks, side, h.score)
    if "Left" in best:
        _write(frame, LHAND_OFF, N_LHAND, best["Left"].landmarks)
    if "Right" in best:
        _write(frame, RHAND_OFF, N_RHAND, best["Right"].landmarks)
    _write(frame, POSE_OFF, N_POSE, pose)
    return frame


class Detector:
    """Three MediaPipe Tasks API landmarkers in VIDEO mode (same as signchat)."""

    def __init__(self, models_dir: Path, use_face: bool = True):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self.mp = mp
        mode = vision.RunningMode.VIDEO
        self.face = None
        if use_face:
            self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(models_dir / "face_landmarker.task")),
                running_mode=mode, num_faces=1))
        self.hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "hand_landmarker.task")),
            running_mode=mode, num_hands=2))
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(models_dir / "pose_landmarker_lite.task")),
            running_mode=mode, num_poses=1))

    def process(self, rgb: np.ndarray, ts_ms: int):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        hr = self.hand.detect_for_video(img, ts_ms)
        hands = []
        for lms, cats in zip(hr.hand_landmarks or [], hr.handedness or []):
            if cats:
                hands.append(HandObs(lms, cats[0].category_name, float(cats[0].score)))
        pose = None
        pr = self.pose.detect_for_video(img, ts_ms)
        if pr.pose_landmarks:
            pose = pr.pose_landmarks[0]
        face = None
        if self.face is not None:
            fr = self.face.detect_for_video(img, ts_ms)
            if fr.face_landmarks:
                face = fr.face_landmarks[0]
        return face, hands, pose


# =============================================================================
# 3. ONNX classifier
# =============================================================================
class SignClassifier:
    def __init__(self, onnx_path: Path, labels_path: Path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        idx = json.loads(Path(labels_path).read_text())
        self.labels = [None] * len(idx)
        for label, i in idx.items():
            self.labels[i] = label

    def predict(self, window: np.ndarray) -> np.ndarray:
        """window: (T, 543, 3) float32 -> probabilities (250,)."""
        logits = self.sess.run(None, {self.input_name: window.astype(np.float32)})[0]
        z = logits - logits.max()
        e = np.exp(z)
        return e / e.sum()

    def topk(self, probs: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(probs)[::-1][:k]
        return [(self.labels[i], float(probs[i])) for i in order]


# =============================================================================
# 4. Admit: when a guess becomes a "recognized sign" (signchat logic)
# =============================================================================
@dataclass
class SignToken:
    label: str
    score: float
    via: str                            # "stable" | "band"
    topk: list[tuple[str, float]]


class Admitter:
    """
    - stable: top1 >= top1_thr and equal to the previous inference's top1 (2 ticks in a row)
    - band:   top2_thr <= top1 < top1_thr, top2 >= top2_thr and top1 != last accepted
              (only makes sense with the LLM, which picks among the alternatives)
    Extra: does not repeat the same sign until the hands are lowered.
    """

    def __init__(self, top1_thr=0.3, top2_thr=0.1, use_band=True):
        self.top1_thr, self.top2_thr, self.use_band = top1_thr, top2_thr, use_band
        self.prev_top: str | None = None
        self.last_label: str | None = None
        self.allow_repeat = True

    def boundary(self) -> None:
        """Call when the hands disappear: a new sign starts."""
        self.prev_top = None
        self.allow_repeat = True

    def step(self, top: list[tuple[str, float]]) -> SignToken | None:
        (l1, s1), (l2, s2) = top[0], top[1]
        via = None
        if s1 >= self.top1_thr and l1 == self.prev_top:
            via = "stable"
        elif (self.use_band and self.top2_thr <= s1 < self.top1_thr
              and s2 >= self.top2_thr and l1 != self.last_label):
            via = "band"
        self.prev_top = l1
        if via is None:
            return None
        if l1 == self.last_label and not self.allow_repeat:
            return None
        self.last_label = l1
        self.allow_repeat = False
        return SignToken(l1, s1, via, top)


# =============================================================================
# 5. Signs -> sentence
# =============================================================================
def compose_simple(tokens: list[SignToken]) -> str:
    words = [SPOKEN_LABELS.get(t.label.upper(), t.label.lower()) for t in tokens]
    s = " ".join(words).strip()
    return (s[:1].upper() + s[1:] + ".") if s else ""


LLM_SYSTEM = (
    "You reconstruct what a Deaf signer just said in casual English from a noisy "
    "classifier's top-K output. The CURRENT signs are the source of truth: the sentence "
    "must be built from them, one word/idea per sign, in order (use a listed alternative "
    "only when the top-1 makes no sense with the other current signs). Previous sentences "
    "are given only to help disambiguate; never repeat or paraphrase a previous sentence "
    "and never introduce words or topics that are not in the current signs, apart from "
    "small grammar words. Keep it short and natural. Return JSON only with keys "
    '"sentence" (string) and "usedSigns" (array of strings).'
)


def build_llm_user_prompt(tokens: list[SignToken], recent_dialog: list[str]) -> str:
    lines = []
    for i, t in enumerate(tokens, 1):
        alts = ", ".join(f"{dictionary_label(l)} ({s:.2f})" for l, s in t.topk[1:])
        lines.append(f"{i}. {dictionary_label(t.label)} ({t.score:.2f}) | alternatives: {alts}")
    vocab = sorted({dictionary_label(l) for t in tokens for l, _ in t.topk})
    dialog = "\n".join(recent_dialog[-2:]) or "(none)"
    return (f"Previous sentences (context only, do NOT repeat):\n{dialog}\n"
            f"Top-K classifier output for the CURRENT sign turn:\n" + "\n".join(lines) +
            f"\nWord translations: {', '.join(vocab)}\n\nReturn JSON only.")


class LLMClient:
    """Minimal client for any OpenAI-compatible API (OpenRouter, OpenAI, Ollama...)."""

    def __init__(self):
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        self.model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
        is_local = "localhost" in self.base_url or "127.0.0.1" in self.base_url
        self.available = bool(self.api_key) or is_local

    def _call(self, messages: list[dict], max_tokens: int, no_reasoning: bool) -> tuple[str | None, str]:
        payload = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": messages}
        if no_reasoning and "openrouter.ai" in self.base_url:
            payload["reasoning"] = {"enabled": False}
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}") from None
        if "error" in data:
            raise RuntimeError(str(data["error"])[:300])
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        return content, str(choice.get("finish_reason"))

    def reconstruct(self, tokens: list[SignToken], recent_dialog: list[str]) -> str:
        messages = [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": build_llm_user_prompt(tokens, recent_dialog)},
        ]
        # 1st attempt: reasoning off (fast). If it comes back empty (a reasoning model that
        # ignored the request or spent its tokens thinking), retry with more tokens.
        content, finish = self._call(messages, 600, no_reasoning=True)
        if not content:
            print(f"[llm] empty response (finish_reason={finish}); retrying with more tokens...",
                  flush=True)
            content, finish = self._call(messages, 4000, no_reasoning=False)
        if not content:
            raise RuntimeError(f"model returned empty text (finish_reason={finish}); "
                               "pick another model in LLM_MODEL")
        return parse_llm_sentence(content)


def parse_llm_sentence(content: str) -> str:
    m = re.search(r"\{.*\}", content, re.S)
    if m:
        try:
            return str(json.loads(m.group(0))["sentence"]).strip()
        except (ValueError, KeyError):
            pass
    return content.strip().strip('"')


# =============================================================================
# 6. Voice (native system TTS, no dependencies)
# =============================================================================
_WIN_SPEAK = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$s.Rate = [int]$env:SIGNVOICE_RATE; "
    "if ($env:SIGNVOICE_VOICE) { $s.SelectVoice($env:SIGNVOICE_VOICE) } "
    "else { $v = $s.GetInstalledVoices() | Where-Object { $_.Enabled -and "
    "$_.VoiceInfo.Culture.Name -like ($env:SIGNVOICE_LANG + '*') } | Select-Object -First 1; "
    "if ($v) { $s.SelectVoice($v.VoiceInfo.Name) } "
    "else { [Console]::Error.WriteLine('[voice] no ' + $env:SIGNVOICE_LANG + "
    "' voice installed; using the default voice. Run: python signvoice.py --list-voices') } }; "
    "$s.Speak($env:SIGNVOICE_TEXT)"
)
_WIN_LIST = (
    "Add-Type -AssemblyName System.Speech; "
    "(New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | "
    "ForEach-Object { $_.VoiceInfo.Name + '  |  ' + $_.VoiceInfo.Culture.Name }"
)
VOICE_LANG = "en"      # voice language (culture prefix: en, en-US, en-GB...)
VOICE_NAME = ""        # exact voice name; empty = first voice of the language


def list_voices() -> None:
    if platform.system() == "Windows":
        subprocess.run(["powershell", "-NoProfile", "-Command", _WIN_LIST], check=False)
    elif platform.system() == "Darwin":
        subprocess.run(["say", "-v", "?"], check=False)
    else:
        print("On Linux: espeak-ng --voices")


def speak(text: str, voice_rate: int = 0) -> None:
    if not text:
        return
    system = platform.system()
    try:
        if system == "Windows":
            env = {**os.environ, "SIGNVOICE_TEXT": text, "SIGNVOICE_RATE": str(int(voice_rate)),
                   "SIGNVOICE_LANG": VOICE_LANG, "SIGNVOICE_VOICE": VOICE_NAME}
            subprocess.run(["powershell", "-NoProfile", "-Command", _WIN_SPEAK],
                           env=env, check=False,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        elif system == "Darwin":
            subprocess.run(["say", "-v", VOICE_NAME or "Samantha", text], check=False)
        else:
            for cmd in (["espeak-ng", text], ["espeak", text], ["spd-say", "-w", text]):
                if shutil.which(cmd[0]):
                    subprocess.run(cmd, check=False)
                    break
            else:
                print("[voice] install espeak-ng to hear the speech:", text)
    except Exception as e:  # noqa: BLE001
        print("[voice] error:", e)


class TurnWorker(threading.Thread):
    """Closes each turn (sentence + voice) outside the video loop, so the camera never stalls."""

    def __init__(self, llm: LLMClient | None, voice_rate: int):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue()
        self.llm, self.voice_rate = llm, voice_rate
        self.use_llm = bool(llm and llm.available)
        self.last_sentence = ""
        self.busy = False
        self.dialog: list[str] = []
        # utterance history from both modes, for the sidebar: (who, text)
        self.history: deque[tuple[str, str]] = deque(maxlen=50)

    def remember(self, who: str, text: str) -> None:
        """who: 'you' (signed sentence) | 'them' (speech heard in listen mode)."""
        self.history.append((who, text))
        self.dialog.append(f"{'You' if who == 'you' else 'They'} said: {text}")

    def submit(self, tokens: list[SignToken], sentence: str = "") -> None:
        """sentence: an already composed sentence (e.g. from the preview), to avoid calling the LLM again."""
        self.q.put((list(tokens), sentence))

    def compose(self, tokens: list[SignToken]) -> str:
        sentence = ""
        if self.use_llm and self.llm:
            try:
                sentence = self.llm.reconstruct(tokens, self.dialog)
            except Exception as e:  # noqa: BLE001
                print("[llm] failed, using the simple sentence:", e)
        return sentence or compose_simple(tokens)

    def run(self):
        while True:
            tokens, sentence = self.q.get()
            self.busy = True
            sentence = sentence or self.compose(tokens)
            self.last_sentence = sentence
            self.remember("you", sentence)
            print(f"[sentence] {' '.join(t.label for t in tokens)}  ->  {sentence}", flush=True)
            speak(sentence, self.voice_rate)
            self.busy = False


class PreviewWorker(threading.Thread):
    """Recomposes the sentence whenever a sign is added/removed (display only, not spoken).
    Serves only the latest request; until the LLM answers, the simple sentence is shown."""

    def __init__(self, turn: TurnWorker):
        super().__init__(daemon=True)
        self.turn = turn
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.pending: list[SignToken] = []
        self.want: tuple[str, ...] = ()       # signs of the latest request
        self.done: tuple[str, ...] = ()       # signs of the last composed sentence
        self.sentence = ""

    def request(self, tokens: list[SignToken]) -> None:
        key = tuple(t.label for t in tokens)
        if key == self.want:
            return
        with self.lock:
            self.want, self.pending = key, list(tokens)
        self.event.set()

    def text(self, tokens: list[SignToken]) -> str:
        if not tokens:
            return ""
        return self.ready(tokens) or compose_simple(tokens)

    def ready(self, tokens: list[SignToken]) -> str:
        """Sentence already composed for exactly these signs, or ''."""
        with self.lock:
            return self.sentence if self.done == tuple(t.label for t in tokens) else ""

    def run(self):
        while True:
            self.event.wait()
            self.event.clear()
            with self.lock:
                key, tokens = self.want, self.pending
            sentence = self.turn.compose(tokens) if tokens else ""
            with self.lock:
                if key == self.want:          # discard if the signs changed meanwhile
                    self.done, self.sentence = key, sentence


# =============================================================================
# 6b. Listen mode: microphone -> text (faster-whisper, local)
# =============================================================================
class Listener(threading.Thread):
    """Captures the microphone only while active, splits each utterance by energy
    (starts above the background noise, ends after `end_silence` s of silence)
    and transcribes it with Whisper (faster-whisper / CTranslate2, CPU int8)."""

    SR = 16000
    BLOCK = 1600                      # 0.1 s per block

    def __init__(self, model_name: str, lang: str, models_dir: Path, on_text=None,
                 is_muted=None, end_silence: float = 0.8, max_utt: float = 15.0):
        super().__init__(daemon=True)
        self.model_name, self.lang, self.models_dir = model_name, lang or None, models_dir
        self.on_text = on_text
        self.is_muted = is_muted or (lambda: False)
        self.end_blocks = int(end_silence * self.SR / self.BLOCK)
        self.max_blocks = int(max_utt * self.SR / self.BLOCK)
        self.active = threading.Event()
        self.q: queue.Queue = queue.Queue()
        self.lines: deque[str] = deque(maxlen=20)
        self.status = ""
        self.model = None

    def set_active(self, on: bool) -> None:
        if on:
            self.status = self.status or "loading speech model..."
            self.active.set()
        else:
            self.active.clear()

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        self.status = f"loading speech model ({self.model_name})..."
        print(f"[listen] loading Whisper '{self.model_name}' "
              "(downloaded on first use)...", flush=True)
        self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8",
                                  download_root=str(self.models_dir / "whisper"))

    def _transcribe(self, audio: np.ndarray) -> str:
        self.status = "transcribing..."
        segments, _ = self.model.transcribe(audio, language=self.lang, beam_size=1,
                                            vad_filter=True, condition_on_previous_text=False)
        # drop segments Whisper itself considers non-speech (avoids "Thank you." on silence)
        return " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.6).strip()

    def run(self):
        import sounddevice as sd

        while True:
            self.active.wait()
            try:
                if self.model is None:
                    self._load()
                with sd.InputStream(samplerate=self.SR, channels=1, dtype="float32",
                                    blocksize=self.BLOCK,
                                    callback=lambda data, *_: self.q.put(data[:, 0].copy())):
                    self._listen_loop()
            except Exception as e:  # noqa: BLE001
                self.status = f"speech error: {e}"
                print("[listen] error:", e, flush=True)
                self.active.clear()

    def _listen_loop(self) -> None:
        noise, speech, silent = 0.005, [], 0
        pre: deque[np.ndarray] = deque(maxlen=3)        # 0.3 s before the speech starts
        while self.active.is_set():
            self.status = "speaking detected..." if speech else "listening..."
            try:
                block = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.is_muted():
                speech, silent = [], 0
                pre.clear()
                continue
            rms = float(np.sqrt(np.mean(block ** 2)))
            loud = rms > max(0.01, 3 * noise)
            if not speech:
                if loud:
                    speech, silent = [*pre, block], 0
                else:
                    noise = 0.95 * noise + 0.05 * rms      # track the background noise
                    pre.append(block)
                continue
            speech.append(block)
            silent = 0 if loud else silent + 1
            if silent >= self.end_blocks or len(speech) >= self.max_blocks:
                text = self._transcribe(np.concatenate(speech))
                speech, silent = [], 0
                pre.clear()
                if text:
                    self.lines.append(text)
                    print(f"[heard] {text}", flush=True)
                    if self.on_text:
                        self.on_text(text)
        with self.q.mutex:
            self.q.queue.clear()
        self.status = ""


# =============================================================================
# 6c. Optional remote control (TIAGo robot): phone <-> tiago_remote server <-> here
# =============================================================================
class RemoteLink(threading.Thread):
    """Only used with --server. Sends the conversation state to the tiago_remote server
    (shown on the phone) and receives the phone's buttons, which run the same actions
    as the on-screen buttons. Reconnects automatically. Needs: pip install websocket-client"""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        try:
            import websocket
        except ImportError:
            raise SystemExit("--server needs: pip install websocket-client")
        self.websocket = websocket
        self.url = url
        self.ws = None
        self.lock = threading.Lock()
        self.inbox: queue.Queue = queue.Queue()
        self.last = ""
        self.last_t = 0.0

    def run(self):
        fails = 0
        while True:
            try:
                ws = self.websocket.create_connection(self.url, timeout=5)
                fails = 0
                ws.settimeout(None)
                with self.lock:
                    self.ws, self.last = ws, ""          # resend the full state after reconnecting
                print(f"[remote] connected to {self.url}", flush=True)
                while True:
                    msg = ws.recv()
                    if not msg:
                        break
                    m = json.loads(msg)
                    if m.get("type") == "action":
                        self.inbox.put(m)
            except Exception as e:  # noqa: BLE001
                if self.ws is not None:
                    print(f"[remote] disconnected ({e}); retrying...", flush=True)
                else:
                    fails += 1
                    if fails == 1 or fails % 15 == 0:      # say why, without flooding the terminal
                        print(f"[remote] cannot reach {self.url} ({e}); retrying...", flush=True)
            with self.lock:
                self.ws = None
            time.sleep(1.0)

    def actions(self) -> list[dict]:
        out = []
        while not self.inbox.empty():
            out.append(self.inbox.get())
        return out

    def publish(self, state: dict) -> None:
        """Sends the state when it changes (and at least every 2 s)."""
        data = json.dumps({"type": "state", **state})
        now = time.monotonic()
        with self.lock:
            if self.ws is None or (data == self.last and now - self.last_t < 2.0):
                return
            try:
                self.ws.send(data)
                self.last, self.last_t = data, now
            except Exception:  # noqa: BLE001
                pass


# =============================================================================
# 7. Interface (OpenCV) and gesture control
# =============================================================================
# camera region (normalized x0, y0, x1, y1, already mirrored) the index finger sweeps
# to cover the whole screen: corners are reachable without the hand leaving the frame
POINT_BOX = (0.15, 0.08, 0.85, 0.78)


@dataclass
class Target:
    """Clickable/pointable area: a command button or the x of a buffered sign."""
    action: str                              # mode | speak | undo | clear | llm | fullscreen | quit | del
    rect: tuple[int, int, int, int]          # hit area (x1, y1, x2, y2) on screen
    label: str = ""
    arg: int = -1                            # sign index (action "del")
    box: tuple[int, int, int, int] | None = None   # chip rectangle (action "del")

    @property
    def key(self) -> tuple[str, int]:
        return self.action, self.arg

    def hit(self, x: int, y: int) -> bool:
        x1, y1, x2, y2 = self.rect
        return x1 <= x <= x2 and y1 <= y <= y2


def pointing_tip(hands: list[HandObs], fw: int, fh: int) -> tuple[float, float] | None:
    """Index fingertip (normalized) if a hand is pointing:
    index extended and middle, ring and pinky curled."""
    for h in hands:
        lm = h.landmarks
        wx, wy = lm[0].x * fw, lm[0].y * fh
        d = lambda i: np.hypot(lm[i].x * fw - wx, lm[i].y * fh - wy)   # noqa: E731
        if d(8) > 1.15 * d(6) and all(d(tip) < d(pip) for tip, pip in ((12, 10), (16, 14), (20, 18))):
            return lm[8].x, lm[8].y
    return None


def to_screen(tip: tuple[float, float], mirror: bool, size: tuple[int, int]) -> tuple[int, int]:
    W, H = size
    bx0, by0, bx1, by1 = POINT_BOX
    u = (1 - tip[0]) if mirror else tip[0]
    u = min(max((u - bx0) / (bx1 - bx0), 0.0), 1.0)
    v = min(max((tip[1] - by0) / (by1 - by0), 0.0), 1.0)
    return int(u * (W - 1)), int(v * (H - 1))


class DwellPointer:
    """Gesture 'click': hold the cursor over a target for `dwell` seconds.
    After firing, it only fires again once the cursor leaves the target."""

    def __init__(self, dwell: float = 1.0, quit_dwell: float = 2.0):
        self.dwell, self.quit_dwell = dwell, quit_dwell
        self.key: tuple[str, int] | None = None
        self.t0 = 0.0
        self.armed = True

    def update(self, pos, targets: list[Target], now: float):
        """Returns (target under the cursor, target fired now, progress 0..1)."""
        hit = next((t for t in targets if pos and t.hit(*pos)), None)
        key = hit.key if hit else None
        if key != self.key:
            self.key, self.t0, self.armed = key, now, True
        if hit is None or not self.armed:
            return hit, None, 0.0
        need = self.quit_dwell if hit.action == "quit" else self.dwell
        progress = min(1.0, (now - self.t0) / need)
        if progress >= 1.0:
            self.armed = False
            return hit, hit, 1.0
        return hit, None, progress


def ui_scale(W: int, H: int) -> float:
    return max(0.6, min(H / 480, W / 640, 3.0))     # scale by the tighter side


def window_size(cv2, win: str, fallback: tuple[int, int]) -> tuple[int, int]:
    """Current size of the window's image area (width, height)."""
    try:
        _, _, w, h = cv2.getWindowImageRect(win)
        if w > 0 and h > 0:
            return w, h
    except cv2.error:
        pass
    return fallback


def letterbox(cv2, frame: np.ndarray, out_w: int, out_h: int):
    """Fits the frame into an out_w x out_h canvas without distortion (black bars in the gaps).
    Returns (canvas, (x0, y0, video_width, video_height))."""
    fh, fw = frame.shape[:2]
    scale = min(out_w / fw, out_h / fh)
    vw, vh = max(1, int(fw * scale)), max(1, int(fh * scale))
    x0, y0 = (out_w - vw) // 2, (out_h - vh) // 2
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    canvas[y0:y0 + vh, x0:x0 + vw] = cv2.resize(frame, (vw, vh), interpolation=interp)
    return canvas, (x0, y0, vw, vh)


def _fit_text(cv2, text: str, max_w: int, scale: float, thick: int) -> str:
    """Truncates the text (with …) to fit in max_w pixels."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if cv2.getTextSize(text, font, scale, thick)[0][0] <= max_w:
        return text
    while text and cv2.getTextSize(text + "...", font, scale, thick)[0][0] > max_w:
        text = text[:-1]
    return text + "..."


def sidebar_x(W: int, H: int) -> int:
    """Left edge of the history sidebar (takes the full height on the right)."""
    return W - max(int(180 * ui_scale(W, H)), int(W * 0.25))


def layout_targets(cv2, W: int, H: int, tokens: list[SignToken], llm_on: bool,
                   fullscreen: bool, listening: bool = False) -> list[Target]:
    """Positions of the buttons (bottom bar) and of the sign chips (panel above it, left of
    the sidebar). In listen mode only the relevant buttons remain and there are no chips."""
    s = ui_scale(W, H)
    px = lambda v: max(1, int(v * s))               # noqa: E731
    font = cv2.FONT_HERSHEY_SIMPLEX
    targets = []
    area_w = sidebar_x(W, H)

    screen = ("fullscreen", "Window" if fullscreen else "Fullscreen")
    if listening:
        buttons = [("mode", "Mode: listen"), ("clear", "Clear"), screen, ("quit", "Quit")]
        tokens = []
    else:
        buttons = [("mode", "Mode: talk"), ("speak", "Speak"), ("undo", "Undo"),
                   ("clear", "Clear"), ("llm", f"LLM {'on' if llm_on else 'off'}"),
                   screen, ("quit", "Quit")]
    # buttons in a full-width bottom bar (below the text and the history)
    margin, bh, gap = px(10), px(40), px(8)
    cols = len(buttons)
    if (W - 2 * margin - (cols - 1) * gap) // cols < px(75):        # narrow: 2 rows
        cols = (cols + 1) // 2
    rows = (len(buttons) + cols - 1) // cols
    bar_top = H - (2 * margin + rows * bh + (rows - 1) * gap)
    bw = (W - 2 * margin - (cols - 1) * gap) // cols
    for i, (action, label) in enumerate(buttons):
        x = margin + (i % cols) * (bw + gap)
        y = bar_top + margin + (i // cols) * (bh + gap)
        targets.append(Target(action, (x, y, x + bw, y + bh), label))

    # chips: show the latest ones that fit; the x sits on each one's top-right corner
    sc, th = 0.65 * s, px(2)
    pad, ch, r, gap = px(10), px(34), px(11), px(18)
    widths = [cv2.getTextSize(t.label, font, sc, th)[0][0] + 2 * pad for t in tokens]
    avail = area_w - px(24) - px(40)                # px(40) reserved for the "..."
    first, used = len(tokens), 0
    while first > 0 and used + widths[first - 1] + gap <= avail:
        first -= 1
        used += widths[first] + gap
    x = px(12) + (px(40) if first > 0 else 0)
    y1 = bar_top - px(106)
    hr = int(r * 1.8)                               # hit area larger than the drawn x
    for i in range(first, len(tokens)):
        x2 = x + widths[i]
        targets.append(Target("del", (x2 - hr, y1 - hr, x2 + hr, y1 + hr), tokens[i].label, i,
                              box=(x, y1, x2, y1 + ch)))
        x = x2 + gap
    return targets


def _wrap(cv2, text: str, max_w: int, scale: float, thick: int) -> list[str]:
    """Wraps the text into lines of at most max_w pixels (a word too long is truncated)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and cv2.getTextSize(cand, font, scale, thick)[0][0] > max_w:
            lines.append(cur)
            cand = word
        cur = cand
    if cur:
        lines.append(cur)
    return [_fit_text(cv2, line, max_w, scale, thick) for line in lines]


def draw_history(cv2, canvas, history: list[tuple[str, str]], box, s: float) -> None:
    """Sidebar: the most recent utterances (at the bottom) that fit in the box, from both modes."""
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    px = lambda v: max(1, int(v * s))               # noqa: E731
    sc, th, lh = 0.5 * s, px(1), px(20)             # scale, thickness, line height
    pad = px(10)
    cv2.putText(canvas, "History", (x1 + pad, y1 + px(24)), font, 0.6 * s,
                (255, 255, 255), px(2), cv2.LINE_AA)
    top, y = y1 + px(40), y2 - pad
    for who, text in reversed(history):
        name, color = ("You", (0, 255, 180)) if who == "you" else ("Them", (255, 180, 80))
        lines = _wrap(cv2, text, x2 - x1 - 3 * pad, sc, th)
        need = lh * (len(lines) + 1)
        if y - need < top:
            break
        for i, line in enumerate(lines):
            cv2.putText(canvas, line, (x1 + 2 * pad, y - (len(lines) - 1 - i) * lh), font, sc,
                        (230, 230, 230), th, cv2.LINE_AA)
        cv2.putText(canvas, name, (x1 + pad, y - len(lines) * lh), font, sc, color,
                    px(2), cv2.LINE_AA)
        y -= need + px(8)


def draw_ui(cv2, canvas, video_rect, hands, mirror, targets, worker, sentence,
            hover=None, progress=0.0, cursor=None, listener=None):
    """Draws the interface on the final canvas. Everything scales with the window size.
    listener: the Listener when listen mode is active (shows the transcript)."""
    H, W = canvas.shape[:2]
    x0, y0, vw, vh = video_rect
    s = ui_scale(W, H)
    font = cv2.FONT_HERSHEY_SIMPLEX
    px = lambda v: max(1, int(v * s))               # noqa: E731
    buttons = [t for t in targets if t.action != "del"]
    chips = [t for t in targets if t.action == "del"]

    # hand points, placed inside the video area
    for hobs in hands:
        for p in hobs.landmarks:
            x = (1 - p.x) if mirror else p.x
            cv2.circle(canvas, (x0 + int(x * vw), y0 + int(p.y * vh)), px(3), (0, 255, 180), -1)

    side_x = sidebar_x(W, H)
    left, max_w = px(12), side_x - px(24)
    # top of the bottom button bar; the text panel and the history sit above it
    bar_top = min((t.rect[1] for t in buttons), default=H + px(10)) - px(10)

    # bottom panel texts: (text, y above the bar top, size, thickness, color)
    if listener is not None:
        # listen mode: the last heard utterance, wrapped (earlier ones are in the history)
        heard = listener.lines[-1] if listener.lines else ""
        lines = _wrap(cv2, heard, max_w, 0.8 * s, px(2))[-6:] or [""]
        rows = [(line, 44 + 34 * (len(lines) - 1 - i), 0.8, 2, (255, 255, 255))
                for i, line in enumerate(lines)]
        rows.append((listener.status, 14, 0.5, 1, (0, 255, 180)))
        panel_h = px(84 + 34 * (len(lines) - 1))
    else:
        sent = ("(speaking...) " if worker.busy else "") + sentence
        rows = [
            (f"sentence: {sent}", 44, 0.7, 2, (255, 255, 255)),
            ("point and hold on a button or x | lower hands to speak",
             14, 0.5, 1, (180, 180, 180)),
        ]
        panel_h = px(122)

    # semi-transparent panels and buttons
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, bar_top - panel_h), (side_x, bar_top), (20, 20, 20), -1)
    cv2.rectangle(overlay, (side_x, 0), (W, bar_top), (20, 20, 20), -1)
    cv2.rectangle(overlay, (0, bar_top), (W, H), (20, 20, 20), -1)
    for t in buttons:
        color = (90, 70, 20) if t.key == hover else (40, 40, 40)
        cv2.rectangle(overlay, t.rect[:2], t.rect[2:], color, -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    # command buttons
    for t in buttons:
        x1, y1, x2, y2 = t.rect
        if t.action == "quit":
            edge = (60, 60, 230)
        else:
            edge = (0, 255, 180) if t.key == hover else (120, 120, 120)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), edge, px(2))
        # centered text; shrink the font if the button is narrow
        sc, th = 0.6 * s, px(1)
        tw, tht = cv2.getTextSize(t.label, font, sc, th)[0]
        if tw > (x2 - x1) - px(8):
            sc *= ((x2 - x1) - px(8)) / tw
            tw, tht = cv2.getTextSize(t.label, font, sc, th)[0]
        cv2.putText(canvas, t.label, ((x1 + x2 - tw) // 2, (y1 + y2 + tht) // 2), font, sc,
                    (255, 255, 255), th, cv2.LINE_AA)

    draw_history(cv2, canvas, list(worker.history), (side_x, 0, W, bar_top), s)

    # sign chips, each with an x on top
    if chips and chips[0].arg > 0:
        cv2.putText(canvas, "...", (left, bar_top - px(82)), font, 0.65 * s,
                    (0, 255, 180), px(2), cv2.LINE_AA)
    r = px(11)
    for t in chips:
        x1, y1, x2, y2 = t.box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 60, 60), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 180), px(2))
        cv2.putText(canvas, t.label, (x1 + px(10), y2 - px(10)), font, 0.65 * s,
                    (255, 255, 255), px(2), cv2.LINE_AA)
        hot = t.key == hover
        cv2.circle(canvas, (x2, y1), int(r * 1.3) if hot else r,
                   (80, 80, 255) if hot else (40, 40, 200), -1, cv2.LINE_AA)
        d = int(r * 0.45)
        for a, b in (((x2 - d, y1 - d), (x2 + d, y1 + d)), ((x2 - d, y1 + d), (x2 + d, y1 - d))):
            cv2.line(canvas, a, b, (255, 255, 255), px(2), cv2.LINE_AA)

    for text, from_bottom, size, thick, color in rows:
        sc, th = size * s, px(thick)
        cv2.putText(canvas, _fit_text(cv2, text, max_w, sc, th), (left, bar_top - px(from_bottom)),
                    font, sc, color, th, cv2.LINE_AA)

    # gesture cursor, with a progress ring while holding over a target
    if cursor:
        cv2.circle(canvas, cursor, px(9), (255, 255, 255), px(2), cv2.LINE_AA)
        if progress > 0:
            cv2.ellipse(canvas, cursor, (px(17), px(17)), -90, 0, int(360 * progress),
                        (0, 255, 180), px(4), cv2.LINE_AA)


# =============================================================================
# 8. Main loop
# =============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="ASL -> speech in real time (PopSign 250 vocabulary)")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    ap.add_argument("--window", type=int, default=48, help="frames per window (max 48)")
    ap.add_argument("--min-frames", type=int, default=8, help="minimum frames before inferring")
    ap.add_argument("--infer-every", type=float, default=0.5, help="seconds between inferences")
    ap.add_argument("--top1", type=float, default=0.30, help="confidence threshold (stable)")
    ap.add_argument("--top2", type=float, default=0.10, help="lower threshold (band)")
    ap.add_argument("--hand-gap", type=float, default=0.4,
                    help="s without hands = end of a sign (clears the window)")
    ap.add_argument("--silence", type=float, default=1.5,
                    help="s without hands = end of the sentence (speaks the buffer)")
    ap.add_argument("--dwell", type=float, default=1.0,
                    help="s pointing at a button/x until it fires")
    ap.add_argument("--quit-dwell", type=float, default=2.0,
                    help="s pointing at 'Quit' until it closes")
    ap.add_argument("--swap-hands", action="store_true",
                    help="swap left/right if recognition is poor")
    ap.add_argument("--no-face", action="store_true",
                    help="disable the FaceLandmarker (faster, may lose accuracy)")
    ap.add_argument("--no-mirror", action="store_true", help="do not mirror the preview")
    ap.add_argument("--fullscreen", action="store_true", help="start in fullscreen")
    ap.add_argument("--no-llm", action="store_true", help="never call the LLM")
    ap.add_argument("--no-band", action="store_true",
                    help="only accept stable signs, even with the LLM (less noise)")
    ap.add_argument("--verbose", action="store_true",
                    help="print the top-3 of each inference and which hands were detected")
    ap.add_argument("--voice-rate", type=int, default=0, help="voice speed on Windows (-10..10)")
    ap.add_argument("--voice-lang", default="en", help="voice language (en, en-US, en-GB...)")
    ap.add_argument("--voice", default="", help="exact voice name (see --list-voices)")
    ap.add_argument("--list-voices", action="store_true", help="list the installed voices and exit")
    ap.add_argument("--mode", choices=["talk", "listen"], default="talk",
                    help="initial mode: talk (signs -> speech) or listen (microphone -> text)")
    ap.add_argument("--stt-model", default="base",
                    help="Whisper model for listen mode (tiny, base, small, medium...)")
    ap.add_argument("--stt-lang", default="en",
                    help="spoken language in listen mode (en, pt, fr...; empty = auto-detect)")
    ap.add_argument("--server", default="",
                    help="TIAGo robot: tiago_remote WebSocket, e.g. ws://127.0.0.1:8080/ws/translator "
                         "(buttons also on the phone; pointing gestures off unless --gestures)")
    ap.add_argument("--gestures", action="store_true",
                    help="keep the pointing gestures on when --server is used")
    args = ap.parse_args()

    global VOICE_LANG, VOICE_NAME
    VOICE_LANG, VOICE_NAME = args.voice_lang, args.voice
    if args.list_voices:
        list_voices()
        return

    import cv2

    ensure_models(args.models_dir)
    clf = SignClassifier(args.models_dir / "asl-signs.onnx",
                         args.models_dir / "sign_to_prediction_index_map.json")
    det = Detector(args.models_dir, use_face=not args.no_face)

    llm = None if args.no_llm else LLMClient()
    worker = TurnWorker(llm, args.voice_rate)
    worker.start()
    preview = PreviewWorker(worker)
    preview.start()
    # whatever is heard goes into the history, as context for the talk-mode LLM
    listener = Listener(args.stt_model, args.stt_lang, args.models_dir,
                        on_text=lambda text: worker.remember("them", text),
                        is_muted=lambda: worker.busy)    # do not transcribe our own voice
    listener.start()
    link = RemoteLink(args.server) if args.server else None
    if link:
        link.start()
    # with the phone remote, pointing is off by default: ASL signs like YOU/ME/THERE point too
    gestures = not args.server or args.gestures
    listening = args.mode == "listen"
    listener.set_active(listening)
    admitter = Admitter(args.top1, args.top2, use_band=worker.use_llm and not args.no_band)
    if worker.use_llm:
        print(f"[llm] on: {llm.model} at {llm.base_url}", flush=True)
    else:
        print("[llm] off (LLM_API_KEY not found in this terminal)", flush=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam {args.camera}. Try --camera 1.")

    win = "signvoice"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 720)
    fullscreen = args.fullscreen

    def set_fullscreen(on: bool) -> None:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)

    set_fullscreen(fullscreen)
    clicks: list[tuple[int, int]] = []
    cv2.setMouseCallback(
        win, lambda ev, x, y, *_: clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONUP else None)

    ring: deque[np.ndarray] = deque(maxlen=min(args.window, 48))
    tokens: list[SignToken] = []
    top: list[tuple[str, float]] = []
    targets: list[Target] = []          # targets of the last drawn frame
    canvas_size = (960, 720)
    pointer = DwellPointer(args.dwell, args.quit_dwell)
    cursor: tuple[int, int] | None = None
    last_hand_t = time.monotonic()
    last_infer = 0.0
    last_ts = 0

    def run_action(t: Target) -> bool:
        """Runs a command (gesture, mouse or keyboard). Returns False to quit."""
        nonlocal tokens, fullscreen, listening
        if t.action == "quit":
            return False
        if t.action == "mode":
            listening = not listening
            listener.set_active(listening)
        elif t.action == "clear" and listening:
            listener.lines.clear()
        elif t.action == "speak" and tokens:
            worker.submit(tokens, preview.ready(tokens))
            tokens = []
        elif t.action == "undo" and tokens:
            tokens.pop()
        elif t.action == "clear":
            tokens = []
        elif t.action == "del" and 0 <= t.arg < len(tokens):
            tokens.pop(t.arg)
        elif t.action == "fullscreen":
            fullscreen = not fullscreen
            set_fullscreen(fullscreen)
        elif t.action == "llm":
            if llm and llm.available:
                worker.use_llm = not worker.use_llm
                admitter.use_band = worker.use_llm and not args.no_band
            else:
                print("[llm] set LLM_API_KEY (or a local LLM_BASE_URL) to enable it.")
        return True

    keymap = {ord(" "): "speak", 8: "undo", ord("c"): "clear", ord("l"): "llm",
              ord("f"): "fullscreen", ord("q"): "quit", ord("m"): "mode"}
    print("Ready. Sign; lower your hands to speak the sentence. "
          "Point your index finger at the buttons for commands.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam frame failed.")
            break
        now = time.monotonic()
        ts = max(last_ts + 1, int(now * 1000))   # MediaPipe requires increasing timestamps
        last_ts = ts
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # not mirrored: same as signchat
        face, hands, pose = det.process(rgb, ts)
        mirror = not args.no_mirror

        # gesture cursor: pointing index finger (smoothed)
        tip = pointing_tip(hands, frame.shape[1], frame.shape[0]) if gestures else None
        if tip:
            pos = to_screen(tip, mirror, canvas_size)
            cursor = pos if cursor is None else (int(0.6 * cursor[0] + 0.4 * pos[0]),
                                                 int(0.6 * cursor[1] + 0.4 * pos[1]))
        else:
            cursor = None
        hover, fired, progress = pointer.update(cursor, targets, now)
        actions = [fired] if fired else []

        if hands:
            last_hand_t = now
            # a hand pointing at a button/x is not a sign; there are no signs in listen mode
            if hover is None and not listening:
                ring.append(assemble_frame(face, hands, pose, args.swap_hands))
                if len(ring) >= args.min_frames and now - last_infer >= args.infer_every:
                    last_infer = now
                    top = clf.topk(clf.predict(np.stack(ring)), 3)
                    if args.verbose:
                        sides = "+".join(sorted(h.side for h in hands))
                        print(f"  [top3] hands={sides:<10} face={'y' if face else 'n'} "
                              + "  ".join(f"{l}:{s:.2f}" for l, s in top), flush=True)
                    tok = admitter.step(top)
                    if tok:
                        tokens.append(tok)
                        print(f"[sign] {tok.label} ({tok.score:.2f}, {tok.via})", flush=True)
        else:
            gap = now - last_hand_t
            if gap > args.hand_gap and ring:
                ring.clear()
                admitter.boundary()
                top = []
            if gap > args.silence and tokens and not listening:
                worker.submit(tokens, preview.ready(tokens))
                tokens = []

        preview.request(tokens)             # no-op if the signs did not change
        # while there are signs, show the preview; after speaking, the last spoken sentence
        sentence = preview.text(tokens) if tokens else worker.last_sentence
        display = cv2.flip(frame, 1) if mirror else frame
        win_w, win_h = window_size(cv2, win, (display.shape[1], display.shape[0]))
        canvas_size = (win_w, win_h)
        canvas, video_rect = letterbox(cv2, display, win_w, win_h)
        targets = layout_targets(cv2, win_w, win_h, tokens, worker.use_llm, fullscreen, listening)
        draw_ui(cv2, canvas, video_rect, hands, mirror, targets, worker, sentence,
                hover.key if hover else None, progress, cursor,
                listener if listening else None)
        cv2.imshow(win, canvas)

        # mouse and keyboard still work as an alternative to gestures
        while clicks:
            x, y = clicks.pop(0)
            hit = next((t for t in targets if t.hit(x, y)), None)
            if hit:
                actions.append(hit)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:                       # ESC leaves fullscreen (or quits outside it)
            actions.append(Target("fullscreen" if fullscreen else "quit", (0, 0, 0, 0)))
        elif key in keymap:
            actions.append(Target(keymap[key], (0, 0, 0, 0)))

        if fired:                           # the pointing gesture must not become a sign
            ring.clear()
            admitter.boundary()
            top = []
        if link:
            # phone buttons -> the same actions as the on-screen buttons
            for m in link.actions():
                a, v = m.get("action"), m.get("value")
                if (a == "mode" and v is not None and (v == "listen") == listening) or \
                   (a == "llm" and v is not None and bool(v) == worker.use_llm) or \
                   (a == "fullscreen" and v is not None and bool(v) == fullscreen):
                    continue                    # already in the requested state
                if a in ("mode", "speak", "undo", "clear", "del", "llm", "fullscreen"):
                    actions.append(Target(a, (0, 0, 0, 0), arg=int(m.get("arg", -1))))
            link.publish({
                "mode": "listen" if listening else "talk",
                "tokens": [dictionary_label(t.label).upper() for t in tokens],
                "sentence": sentence, "pending": bool(tokens),
                "heard": listener.lines[-1] if listener.lines else "",
                "listen_status": listener.status,
                "history": [list(h) for h in list(worker.history)[-30:]],
                "llm": worker.use_llm, "llm_available": bool(llm and llm.available),
                "fullscreen": fullscreen, "speaking": worker.busy,
            })
        if not all([run_action(t) for t in actions]):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
