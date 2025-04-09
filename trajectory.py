import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from collections import defaultdict

import cv2 #type:ignore
import numpy as np #type:ignore
from ultralytics import YOLO #type:ignore

# Load the YOLO11 model
model = YOLO('best3-3(v11m_50).pt')

# Open the video file
video_path = "2ants_cropped.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

i = 0
# Loop through the video frames
while cap.isOpened() and i < 100:
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        i = i+1
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        result = model.track(frame, persist=True, conf=0.1)[0]

        # Get the boxes and track IDs
        if result.boxes and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()

            # Visualize the result on the frame
            frame = result.plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

        # Display the annotated frame
        #cv2.imshow("YOLO11 Tracking", frame)

        # Break the loop if 'q' is pressed
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.imwrite('trajectory.jpg', frame)
cv2.destroyAllWindows()