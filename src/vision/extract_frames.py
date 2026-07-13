import cv2
import os
import argparse
from pathlib import Path


VALID_CLASSES = [
    "standing_strike",
    "clinch",
    "takedown_attempt",
    "guard",
    "half_guard",
    "mount",
    "back_control",
    "ground_and_pound"
]


def extract_frames(video_path, output_class, frames_per_second=1,
                   max_frames=500, start_second=0, end_second=None):
    """
    Extract frames from a video file at a given rate.

    Args:
        video_path: path to the video file
        output_class: which position class this video contains
        frames_per_second: how many frames to extract per second of video
        max_frames: maximum frames to extract from this video
        start_second: start extracting from this timestamp
        end_second: stop extracting at this timestamp (None = end of video)
    """

    if output_class not in VALID_CLASSES:
        print(f"Error: '{output_class}' is not a valid class.")
        print(f"Valid classes: {VALID_CLASSES}")
        return

    output_dir = Path(f"data/vision/frames/{output_class}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # count existing frames so we don't overwrite
    existing = len(list(output_dir.glob("*.jpg")))
    print(f"Existing frames in {output_class}: {existing}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    print(f"Video: {video_path}")
    print(f"FPS: {fps:.1f} | Duration: {duration:.1f}s | "
          f"Total frames: {total_frames}")

    # set start position
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps) if end_second else total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # calculate interval between extracted frames
    interval = int(fps / frames_per_second)
    if interval < 1:
        interval = 1

    frame_count = 0
    saved_count = 0
    frame_idx = start_frame

    print(f"Extracting 1 frame every {interval} frames "
          f"({frames_per_second} per second)...")

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret or frame_idx >= end_frame:
            break

        if frame_count % interval == 0:
            # resize to standard size for consistency
            frame_resized = cv2.resize(frame, (640, 480))

            filename = f"{output_class}_{existing + saved_count:05d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"  Saved {saved_count} frames...")

        frame_count += 1
        frame_idx += 1

    cap.release()
    print(f"Done. Saved {saved_count} frames to data/vision/frames/{output_class}/")
    print(f"Total frames in class now: {existing + saved_count}")
    return saved_count


def count_dataset():
    """Print current frame counts per class."""
    print("\n=== DATASET STATUS ===")
    total = 0
    for cls in VALID_CLASSES:
        path = Path(f"data/vision/frames/{cls}")
        count = len(list(path.glob("*.jpg"))) if path.exists() else 0
        bar = "█" * (count // 10) + "░" * max(0, 50 - count // 10)
        print(f"  {cls:25s} {count:4d} frames  {bar}")
        total += count
    print(f"\n  Total: {total} frames across {len(VALID_CLASSES)} classes")
    print(f"  Target: 500+ frames per class ({len(VALID_CLASSES) * 500} total)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from MMA video for position classification"
    )
    parser.add_argument("--video",   type=str, help="Path to video file")
    parser.add_argument("--class",   type=str, dest="cls",
                        help="Position class label")
    parser.add_argument("--fps",     type=float, default=1.0,
                        help="Frames per second to extract (default: 1)")
    parser.add_argument("--max",     type=int, default=500,
                        help="Max frames to extract (default: 500)")
    parser.add_argument("--start",   type=float, default=0,
                        help="Start timestamp in seconds")
    parser.add_argument("--end",     type=float, default=None,
                        help="End timestamp in seconds")
    parser.add_argument("--status",  action="store_true",
                        help="Show dataset status and exit")

    args = parser.parse_args()

    if args.status:
        count_dataset()
    elif args.video and args.cls:
        extract_frames(
            video_path=args.video,
            output_class=args.cls,
            frames_per_second=args.fps,
            max_frames=args.max,
            start_second=args.start,
            end_second=args.end
        )
    else:
        print("Usage:")
        print("  Extract frames: python src/vision/extract_frames.py "
              "--video path/to/video.mp4 --class guard --fps 1 --max 500")
        print("  Check status:   python src/vision/extract_frames.py --status")
        count_dataset()