"""
Real-time keyword spotting demo.

Captures audio from the microphone, extracts MFCC features,
runs inference, and prints detected keywords in real-time.

Usage:
    python realtime_demo.py --backend onnx --threshold 0.7 --stride 100

Press Ctrl+C to stop.
"""

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from audio_capture import AudioCapture, SAMPLE_RATE
from realtime_mfcc import RealtimeMFCC, CLIP_LENGTH, TIME_FRAMES
from inference_engine import create_engine
from keyword_detector import KeywordDetector, NO_KEYWORD, KEYWORDS


def main():
    parser = argparse.ArgumentParser(
        description="Real-time Keyword Spotting Demo"
    )
    parser.add_argument("--backend", type=str, default="onnx",
                        choices=["onnx", "trt"],
                        help="Inference backend: 'onnx' (CPU) or 'trt' (GPU)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model file (default: auto-detect based on backend)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Confidence threshold for keyword detection (default: 0.7)")
    parser.add_argument("--cooldown", type=float, default=500.0,
                        help="Cooldown between detections in ms (default: 500)")
    parser.add_argument("--stride", type=float, default=100.0,
                        help="Inference stride in ms (default: 100)")
    parser.add_argument("--device", type=int, default=None,
                        help="sounddevice input device index (default: system default)")
    args = parser.parse_args()

    # Resolve model path
    if args.model is None:
        if args.backend == "onnx":
            args.model = os.path.join(os.path.dirname(__file__), "models", "kws_fp32.onnx")
        else:
            args.model = os.path.join(os.path.dirname(__file__), "models", "kws_trt_fp16.engine")

    if not os.path.exists(args.model):
        print(f"ERROR: Model file not found: {args.model}")
        print("Run earlier modules to generate the model file.")
        sys.exit(1)

    # Initialize components
    print(f"Loading {args.backend.upper()} model: {args.model}")
    engine = create_engine(args.backend, args.model)

    print(f"Initializing MFCC extractor (n_mfcc=40, time_frames={TIME_FRAMES})")
    mfcc_extractor = RealtimeMFCC(sample_rate=SAMPLE_RATE, device='cpu')

    print(f"Initializing keyword detector (threshold={args.threshold}, cooldown={args.cooldown}ms)")
    detector = KeywordDetector(threshold=args.threshold, cooldown_ms=args.cooldown)

    stride_samples = int(SAMPLE_RATE * args.stride / 1000)

    print(f"\nKeywords: {', '.join(KEYWORDS)}")
    print(f"Stride: {args.stride}ms ({stride_samples} samples)")
    print(f"Listening... (Ctrl+C to stop)\n")

    # Start audio capture
    capture = AudioCapture(sample_rate=SAMPLE_RATE, device=args.device)

    try:
        capture.start()
        time.sleep(0.2)  # Let buffer fill initially

        last_inference_time = time.perf_counter()

        while True:
            now = time.perf_counter()
            elapsed_ms = (now - last_inference_time) * 1000

            if elapsed_ms < args.stride:
                time.sleep(max(0, (args.stride - elapsed_ms) / 1000 - 0.005))
                continue

            last_inference_time = time.perf_counter()

            # Get last 1 second of audio
            audio = capture.get_audio(CLIP_LENGTH)

            # Extract MFCC features
            mfcc = mfcc_extractor.extract(audio)

            # Run inference
            logits, latency_ms = engine.predict(mfcc)

            # Detect keyword
            keyword, confidence = detector.detect(logits)

            # Print result
            if keyword != NO_KEYWORD:
                print(f"[{latency_ms:.2f}ms] \"{keyword}\" ({confidence:.2f})")
            else:
                print(f"[{latency_ms:.2f}ms] \"{NO_KEYWORD}\" ({confidence:.2f})")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        capture.stop()

        stats = engine.get_stats()
        print(f"\nInference Statistics:")
        print(f"  Total inferences: {stats['n']}")
        print(f"  Mean latency: {stats['mean_ms']:.2f} ms")
        print(f"  P50 latency:  {stats['p50_ms']:.2f} ms")
        print(f"  P95 latency:  {stats['p95_ms']:.2f} ms")
        print(f"  P99 latency:  {stats['p99_ms']:.2f} ms")

        engine.close()


if __name__ == "__main__":
    main()