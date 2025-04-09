import cv2 #type:ignore
import glob
import os
import shutil
from pathlib import Path

def duplicate_directory(source_dir, destination_dir):
    """
    Duplicates the source directory to destination directory.
    This can be used to back up the directory before processing.
    """
    try:
        shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
        print(f"Directory '{source_dir}' successfully copied to '{destination_dir}'")
    except FileExistsError:
        print(f"Error: Directory '{destination_dir}' already exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

def get_most_recent_folder(directory):
    """
    Finds the most recently created subdirectory in the given directory
    that starts with 'track'. Returns its path or None if not found.
    """
    # List only subdirectories that start with "track"
    folders = [
        os.path.join(directory, d)
        for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d)) and d.startswith("track")
    ]
    if not folders:
        return None
    most_recent_folder = max(folders, key=os.path.getctime)
    return most_recent_folder

def process_video(true_positive_ids_input, false_positive_ids_input, video_path, txt_dir, output_video, normalized=True):
    """
    Processes the tracking results by removing false positive bounding boxes.
    
    Parameters:
      - true_positive_ids_input (str): Comma-separated true positive IDs.
      - false_positive_ids_input (str): Comma-separated false positive IDs.
      - video_path (str): Path to the background video.
      - txt_dir (str): Directory containing YOLO label (.txt) files.
      - output_video (str): Filename for the updated tracking video.
      - normalized (bool): Whether coordinates in the txt files are normalized.
      
    Returns:
      - output_video (str): The name of the saved video with updated bounding boxes.
    """
    # --- Determine the set of false positive IDs ---
    if true_positive_ids_input.strip():
        # If the user provided true-positive IDs, calculate false positives as:
        # false_positive_ids = all IDs in the txt files - true_positive_ids.
        try:
            true_positive_ids = set(int(x.strip()) for x in true_positive_ids_input.split(',') if x.strip().isdigit())
        except Exception as e:
            print("Error parsing true positive IDs:", e)
            true_positive_ids = set()
        # Gather all track IDs from the txt files.
        txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
        all_track_ids = set()
        for txt_file in txt_files:
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 6:
                        continue
                    try:
                        track_id = int(parts[-1])
                        all_track_ids.add(track_id)
                    except ValueError:
                        continue
        false_positive_ids = all_track_ids - true_positive_ids
        print("All track IDs found:", all_track_ids)
        print("True positive IDs:", true_positive_ids)
        print("Computed false positive IDs:", false_positive_ids)
    elif false_positive_ids_input.strip():
        try:
            false_positive_ids = set(int(x.strip()) for x in false_positive_ids_input.split(',') if x.strip().isdigit())
        except Exception as e:
            print("Error parsing false positive IDs:", e)
            false_positive_ids = set()
    else:
        # If neither input is provided, use a default set.
        # false_positive_ids = {3, 8}
        print("No ID input provided.")
        return

    # --- Step 1: Remove false positive lines from each txt file ---
    # txt_files = sorted(glob.glob(os.path.join(txt_dir, "*.txt")))
    txt_files = sorted(Path(txt_dir).iterdir(), key=os.path.getmtime)
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                track_id = int(parts[-1])
            except ValueError:
                continue
            if track_id not in false_positive_ids:
                filtered_lines.append(line + "\n")
        with open(txt_file, 'w') as f:
            f.writelines(filtered_lines)
    print("False positive lines removed from txt files.")

    # --- Step 2: Re-render the tracking video with updated bounding boxes ---
    # If the output video already exists, delete it
    if os.path.exists(output_video):
        os.remove(output_video)
        print(f"Existing file {output_video} removed.")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    frame_index = 0
    total_txt = len(txt_files)

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_index < total_txt:
            txt_file = txt_files[frame_index]
            with open(txt_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])
                    track_id = int(parts[5])
                except ValueError:
                    continue

                if normalized:
                    x_center *= frame_width
                    y_center *= frame_height
                    box_width *= frame_width
                    box_height *= frame_height

                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Updated tracking video saved as {output_video}")
    return

if __name__ == '__main__':
    # For testing duplicate_directory:
    duplicate_directory("runs/detect/track1/labels", "runs/detect/track1/labels_backup")
    # For testing process_video, uncomment below:
    # process_video("", "", "30.mp4", "runs/detect/track1/labels", "updated_tracking_video.mp4", normalized=True)
