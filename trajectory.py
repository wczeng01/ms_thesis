import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from collections import defaultdict
import cv2  # type: ignore
import numpy as np  # type: ignore
from ultralytics import YOLO  # type: ignore

# Load model
model = YOLO('best3-3(v11m_50).pt')

# Open input video
video_path = "2ants.mp4"
cap = cv2.VideoCapture(video_path)

# Grab properties for the writer
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
out = cv2.VideoWriter('tracked_output.mp4', fourcc, fps, (width, height))

track_history = defaultdict(list)
i = 0

while cap.isOpened() and i < 150:
    success, frame = cap.read()
    if not success:
        break

    i += 1
    if i < 100:
        continue

    # Run tracking & plot
    result = model.track(frame, tracker='bytetrack.yaml', persist=True, conf=0.1)[0]
    if result.boxes and result.boxes.id is not None:
        boxes = result.boxes.xywh.cpu()
        ids   = result.boxes.id.int().cpu().tolist()
        frame = result.plot()

        # Draw trajectories
        for box, tid in zip(boxes, ids):
            x, y, w, h = box
            hist = track_history[tid]
            hist.append((int(x), int(y)))
            if len(hist) > 30:
                hist.pop(0)
            pts = np.array(hist, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(frame, [pts], False, (230,230,230), 2)

    # write the frame to video
    out.write(frame)

# clean up
cap.release()
out.release()
cv2.destroyAllWindows()