import sys
from pathlib import Path
import cv2

video, out, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 6
out.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(video))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(n):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
    ok, f = cap.read()
    if ok:
        cv2.imwrite(str(out / f"still_{i+1:02d}.jpg"), f)
(out / "README.txt").write_text("Frames extracted from rgb.mp4; not an independent photo capture; no EXIF.\n")
