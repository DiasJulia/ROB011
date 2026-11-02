#!/usr/bin/env python3
"""
signvoice_robot — runs the signvoice ASL recogniser for the TIAGo robot.

Put this file next to signvoice.py (it reuses its MediaPipe detector, the 250-sign classifier,
the LLM sentence reconstruction and the voice). signvoice.py itself is not modified.

    webcam (on the robot's head)
      -> MediaPipe (face, hands, pose) -> 250-sign classifier -> sentence (optional LLM) -> voice
      <-> WebSocket to the tiago_remote server:
            sends   webcam frames (only while the phone shows the Camera tab),
                    face box (for the head calibration), signs + sentence (captions)
            receives "active" (recognise only in Translate mode with someone in view)

Differences from signvoice.py:
  * no window and no "point at a button" gestures (controls are on the phone;
    ASL signs that point - YOU, ME, THERE... - would otherwise trigger buttons)
  * signs are only recognised when the server says so

Usage (in the signvoice virtual environment, Python 3.12):
    pip install websocket-client
    python signvoice_robot.py                         # server on this PC (ws://localhost:8080)
    python signvoice_robot.py --server ws://192.168.1.20:8080/ws/recognizer --camera 1
    python signvoice_robot.py --show                  # debug window with the face box
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

import signvoice as sv

try:
    import websocket          # pip install websocket-client
except ImportError:
    raise SystemExit("Missing dependency: pip install websocket-client")


# --------------------------------------------------------------------------- server link
class Link(threading.Thread):
    """Keeps a WebSocket to the tiago_remote server open (reconnects automatically)."""

    def __init__(self, url: str, hello: dict):
        super().__init__(daemon=True)
        self.url, self.hello = url, hello
        self.ws = None
        self.lock = threading.Lock()
        self.active = False      # recognise signs now?
        self.stream = False      # send webcam frames now?
        self.connected = False

    def run(self):
        while True:
            try:
                ws = websocket.create_connection(self.url, timeout=5)
                ws.settimeout(None)
                with self.lock:
                    self.ws = ws
                self.connected = True
                print(f"[robot] connected to {self.url}", flush=True)
                self.send_json(self.hello)
                while True:
                    msg = ws.recv()
                    if not msg:
                        break
                    m = json.loads(msg)
                    if m.get("type") == "config":
                        if bool(m.get("active")) != self.active:
                            print(f"[robot] translation {'ON' if m.get('active') else 'off'}", flush=True)
                        self.active, self.stream = bool(m.get("active")), bool(m.get("stream"))
            except Exception as e:  # noqa: BLE001
                if self.connected:
                    print(f"[robot] disconnected ({e}); retrying...", flush=True)
            self.connected = self.active = self.stream = False
            with self.lock:
                self.ws = None
            time.sleep(1.0)

    def _send(self, data, opcode):
        with self.lock:
            if self.ws is None:
                return
            try:
                self.ws.send(data, opcode=opcode)
            except Exception:  # noqa: BLE001
                pass

    def send_json(self, obj: dict) -> None:
        self._send(json.dumps(obj), websocket.ABNF.OPCODE_TEXT)

    def send_bytes(self, data: bytes) -> None:
        self._send(data, websocket.ABNF.OPCODE_BINARY)


# --------------------------------------------------------------------------- face box
def face_box(face, pose) -> dict:
    """Normalised face box (u, v = centre; w, h = size) from MediaPipe landmarks."""
    pts = None
    if face:
        pts = np.array([[p.x, p.y] for p in face])
    elif pose:
        pts = np.array([[p.x, p.y] for p in pose[:11]])     # nose, eyes, ears, mouth
        c = pts.mean(axis=0)
        pts = c + (pts - c) * 1.6                           # those points cover only part of the face
    if pts is None:
        return {"type": "face", "visible": False}
    (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
    return {"type": "face", "visible": True, "u": float((x0 + x1) / 2), "v": float((y0 + y1) / 2),
            "w": float(x1 - x0), "h": float(y1 - y0)}


def label(tok) -> str:
    return sv.dictionary_label(tok.label).upper()


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="signvoice for the TIAGo robot (webcam -> signs -> server)")
    ap.add_argument("--server", default="ws://localhost:8080/ws/recognizer", help="tiago_remote WebSocket")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--hfov", type=float, default=60.0,
                    help="webcam horizontal field of view in degrees (used to aim the head)")
    ap.add_argument("--rotate180", action="store_true", help="webcam mounted upside down")
    ap.add_argument("--stream-fps", type=float, default=10.0, help="frames/s sent to the phone")
    ap.add_argument("--jpeg-quality", type=int, default=70)
    ap.add_argument("--show", action="store_true", help="debug window (q to quit)")
    ap.add_argument("--mute", action="store_true", help="do not speak the sentences")
    ap.add_argument("--models-dir", type=Path, default=sv.DEFAULT_MODELS_DIR)
    ap.add_argument("--window", type=int, default=48)
    ap.add_argument("--min-frames", type=int, default=8)
    ap.add_argument("--infer-every", type=float, default=0.5)
    ap.add_argument("--top1", type=float, default=0.30)
    ap.add_argument("--top2", type=float, default=0.10)
    ap.add_argument("--hand-gap", type=float, default=0.4)
    ap.add_argument("--silence", type=float, default=1.5, help="hands down for S seconds -> sentence is final")
    ap.add_argument("--swap-hands", action="store_true")
    ap.add_argument("--no-face", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-band", action="store_true")
    ap.add_argument("--voice-rate", type=int, default=0)
    ap.add_argument("--voice-lang", default="en")
    ap.add_argument("--voice", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import cv2

    sv.VOICE_LANG, sv.VOICE_NAME = args.voice_lang, args.voice
    if args.mute:
        sv.speak = lambda *a, **k: None          # TurnWorker looks it up at call time

    sv.ensure_models(args.models_dir)
    clf = sv.SignClassifier(args.models_dir / "asl-signs.onnx",
                            args.models_dir / "sign_to_prediction_index_map.json")
    det = sv.Detector(args.models_dir, use_face=not args.no_face)
    llm = None if args.no_llm else sv.LLMClient()
    worker = sv.TurnWorker(llm, args.voice_rate)
    worker.start()
    preview = sv.PreviewWorker(worker)
    preview.start()
    admitter = sv.Admitter(args.top1, args.top2, use_band=worker.use_llm and not args.no_band)
    print(f"[llm] {'on: ' + llm.model if worker.use_llm else 'off (no LLM_API_KEY)'}", flush=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not open webcam {args.camera}. Try --camera 1.")
    H, W = frame.shape[:2]

    link = Link(args.server, {"type": "hello", "width": W, "height": H, "hfov": args.hfov})
    link.start()

    ring: deque[np.ndarray] = deque(maxlen=min(args.window, 48))
    tokens: list = []
    last_glosses: list[str] = []
    last_hand_t = time.monotonic()
    last_infer = last_face = last_frame = 0.0
    last_ts = 0
    inactive_since = None
    last_sent = None
    print("Ready. Waiting for Translate mode on the phone.", flush=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Webcam frame failed.")
            break
        if args.rotate180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        now = time.monotonic()
        ts = max(last_ts + 1, int(now * 1000))
        last_ts = ts
        face, hands, pose = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)

        # face box -> head calibration (15 Hz)
        fb = face_box(face, pose)
        if now - last_face >= 1 / 15:
            last_face = now
            link.send_json(fb)

        # webcam frames -> phone Camera tab (only when asked)
        if link.stream and now - last_frame >= 1 / args.stream_fps:
            last_frame = now
            ok_jpg, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
            if ok_jpg:
                link.send_bytes(jpg.tobytes())

        # sign recognition (same logic as signvoice.py, without the pointing buttons)
        if link.active:
            inactive_since = None
            if hands:
                last_hand_t = now
                ring.append(sv.assemble_frame(face, hands, pose, args.swap_hands))
                if len(ring) >= args.min_frames and now - last_infer >= args.infer_every:
                    last_infer = now
                    top = clf.topk(clf.predict(np.stack(ring)), 3)
                    if args.verbose:
                        print("  [top3] " + "  ".join(f"{l}:{s:.2f}" for l, s in top), flush=True)
                    tok = admitter.step(top)
                    if tok:
                        tokens.append(tok)
                        print(f"[sign] {tok.label} ({tok.score:.2f}, {tok.via})", flush=True)
            else:
                gap = now - last_hand_t
                if gap > args.hand_gap and ring:
                    ring.clear()
                    admitter.boundary()
                if gap > args.silence and tokens:
                    last_glosses = [label(t) for t in tokens]
                    worker.submit(tokens, preview.ready(tokens))     # final sentence + voice
                    tokens = []
        else:
            if ring:
                ring.clear()
                admitter.boundary()
            if inactive_since is None:
                inactive_since = now
            elif tokens and now - inactive_since > 3.0:
                tokens = []                                           # unfinished sentence dropped

        # captions -> phone + robot screen (only when they change)
        preview.request(tokens)
        caption = preview.text(tokens) if tokens else worker.last_sentence
        glosses = [label(t) for t in tokens] if tokens else last_glosses
        sent = (caption, tuple(glosses))
        if sent != last_sent and link.connected:
            last_sent = sent
            link.send_json({"type": "sign", "caption": caption, "glosses": glosses})

        if args.show:
            view = frame.copy()
            if fb["visible"]:
                x0, y0 = int((fb["u"] - fb["w"] / 2) * W), int((fb["v"] - fb["h"] / 2) * H)
                x1, y1 = int((fb["u"] + fb["w"] / 2) * W), int((fb["v"] + fb["h"] / 2) * H)
                cv2.rectangle(view, (x0, y0), (x1, y1), (26, 122, 255), 2)
            state = ("server OK" if link.connected else "no server") + (" | TRANSLATING" if link.active else "")
            cv2.putText(view, state, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (26, 122, 255), 2)
            cv2.putText(view, caption[:60], (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow("signvoice_robot", view)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
