"""Live: python scripts/run_live.py [--src 0|rtsp://...|file] [--detector yunet --model x.onnx]"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robust_rppg.live import run_camera
ap = argparse.ArgumentParser()
ap.add_argument("--src", default="0"); ap.add_argument("--detector", default="auto"); ap.add_argument("--model", default=None)
ap.add_argument("--preset", default="lean", choices=["lean", "full"])
ap.add_argument("--width", type=int, default=320); ap.add_argument("--height", type=int, default=240)
a = ap.parse_args()
run_camera(int(a.src) if a.src.isdigit() else a.src, a.detector, a.model, a.width, a.height, preset=a.preset)
