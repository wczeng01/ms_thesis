# run_tracking.py
import sys
from ultralytics import YOLO  #type:ignore

def main(video_path, flag, stop_event):
    try:
        model = YOLO('best1-2.pt') if flag == 0 else YOLO('best3-3(v11m_50).pt')
        results = model.track(source=video_path, show=False, show_labels=True, show_boxes=True, save=True,
                              save_txt=True, stream=False, line_width=1, tracker='bytetrack.yaml',
                              conf=0.1 if flag != 0 else None, persist=True if flag != 0 else None)
        
        # Process and save results. If results is an iterator or generator, you must periodically check the stop_event.
        for result in results:
            if stop_event.is_set():
                print("Stop event detected. Exiting tracking loop.")
                break
            # Process the result.
            result.save(filename="uploads/result.jpg")
        print("Tracking completed successfully.")
    except Exception as e:
        print(f"Error during tracking: {e}", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_tracking.py <video_path> <flag>")
        sys.exit(1)

    video_path = sys.argv[1]
    flag = int(sys.argv[2])
    # Here, since you’re running from the command line, you might create a dummy Event.
    from threading import Event
    stop_event = Event()
    print(f"Running tracking on: {video_path} with flag: {flag}")
    main(video_path, flag, stop_event)