# run_tracking.py
import sys
from ultralytics import YOLO #type:ignore

def main(video_path, flag):
    try:
        if flag == 0:
            model = YOLO('best1-2.pt')
            results = model.track(source=video_path, show=False, show_labels=True, show_boxes=True,
                                  save=True, save_txt=True, stream=True, line_width=1, tracker='bytetrack.yaml')
        else:
            model = YOLO('best3-3(v11m_50).pt')
            results = model.track(source=video_path, show=False, show_labels=True, show_boxes=True,
                                  save=True, save_txt=True, stream=True, line_width=1, tracker='bytetrack.yaml', conf=0.1, persist=True)
            
        # Process and save results
        for result in results:
            result.save(filename="uploads/result.jpg")
        print("Tracking completed successfully.")
    except Exception as e:
        print(f"Error during tracking: {e}", file=sys.stderr)

if __name__ == "__main__":
    # Expect the video file path as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python run_tracking.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"Running tracking on: {video_path}")
    main(video_path)
