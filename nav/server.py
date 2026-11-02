#!/usr/bin/env python3
"""Minimal phone joystick server for TIAGo navigation via rosbridge."""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import websocket


ROOT = Path(__file__).resolve().parent
ZERO = {"linear": {"x": 0.0, "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": 0.0}}


class RosbridgeTwist:
    def __init__(self, url: str, topic: str, timeout: float,
                 max_linear: float, max_angular: float):
        self.topic = topic
        self.timeout = timeout
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.lock = threading.Lock()
        self.twist = dict(ZERO)
        self.last_command = 0.0
        self.enabled = True
        self.socket = websocket.create_connection(url, timeout=10)
        self._advertise()
        self.thread = threading.Thread(target=self._watchdog, daemon=True)
        self.thread.start()

    def _advertise(self) -> None:
        self.socket.send(json.dumps({
            "op": "advertise", "topic": self.topic, "type": "geometry_msgs/Twist"}))

    def set_enabled(self, enabled: bool) -> None:
        with self.lock:
            if self.enabled == enabled:
                return
            was_enabled = self.enabled
            if was_enabled and not enabled:
                self._publish(ZERO)
            self.enabled = enabled
            self.twist = dict(ZERO)
            self.last_command = 0.0
        if enabled:
            self._advertise()
        else:
            self.socket.send(json.dumps({"op": "unadvertise", "topic": self.topic}))

    @staticmethod
    def _clamp(value: object, limit: float) -> float:
        return max(-limit, min(limit, float(value)))

    def command(self, linear: object, lateral: object, angular: object = 0.0) -> None:
        with self.lock:
            if not self.enabled:
                return
            self.twist = {
                "linear": {"x": self._clamp(linear, self.max_linear),
                           "y": self._clamp(lateral, self.max_linear), "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0,
                            "z": self._clamp(angular, self.max_angular)},
            }
            self.last_command = time.monotonic()
            message = self.twist
        self._publish(message)

    def stop(self) -> None:
        with self.lock:
            self.twist = dict(ZERO)
            self.last_command = 0.0
        self._publish(ZERO)

    def _publish(self, twist: dict) -> None:
        if not self.enabled:
            return
        self.socket.send(json.dumps({
            "op": "publish", "topic": self.topic, "msg": twist}))

    def _watchdog(self) -> None:
        while True:
            time.sleep(0.1)
            with self.lock:
                expired = self.twist != ZERO and (
                    time.monotonic() - self.last_command > self.timeout)
            if expired:
                self.stop()

    def close(self) -> None:
        self.stop()
        self.socket.send(json.dumps({"op": "unadvertise", "topic": self.topic}))
        self.socket.close()


class Handler(BaseHTTPRequestHandler):
    server_version = "TIAGoNav/1.0"

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self) -> None:
        if self.path not in ("/cmd", "/stop", "/mode"):
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            bridge: RosbridgeTwist = self.server.bridge  # type: ignore[attr-defined]
            if self.path == "/mode":
                bridge.set_enabled(bool(payload.get("enabled", False)))
            elif self.path == "/stop":
                bridge.stop()
            else:
                bridge.command(payload.get("linear", 0.0), payload.get("lateral", 0.0),
                               payload.get("angular", 0.0))
            self._json(200, {"ok": True})
        except (ValueError, TypeError, KeyError, OSError) as exc:
            self._json(400, {"error": str(exc)})

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        data = (ROOT / "web" / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: object) -> None:
        print("[nav] " + (fmt % args), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TIAGo phone navigation server")
    parser.add_argument("--rosbridge-url", default="ws://10.68.0.1:9090")
    parser.add_argument("--topic", default="/phone_vel")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-linear", type=float, default=0.25)
    parser.add_argument("--max-angular", type=float, default=0.5)
    parser.add_argument("--timeout", type=float, default=0.5)
    args = parser.parse_args()

    bridge = RosbridgeTwist(args.rosbridge_url, args.topic, args.timeout,
                            args.max_linear, args.max_angular)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.bridge = bridge  # type: ignore[attr-defined]
    print(f"[nav] phone server: http://0.0.0.0:{args.port}", flush=True)
    print(f"[nav] rosbridge: {args.rosbridge_url} -> {args.topic}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        bridge.close()


if __name__ == "__main__":
    main()
