import cv2 #type:ignore
import glob
import os
import shutil
from pathlib import Path

def process_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if len(lines) == 2:
        parts1 = lines[0].split()
        parts2 = lines[1].split()

        if len(parts1) < 6:
            return None
        
        # return (track ID, x-center, y-center)
        ant1 = [int(parts1[5]), float(parts1[1]), float(parts1[2])]
        ant2 = [int(parts2[5]), float(parts2[1]), float(parts2[2])]
        return ant1, ant2
    else:
        return None
    
def process_video(video_path, txt_dir, output_video, normalized=True):
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

    # --- Step 1: Correct new track IDs for match old for each txt file ---
    # Specify the directory containing the txt files.
    txt_files = sorted(Path(txt_dir).iterdir(), key=os.path.getmtime)
    ant1_ids = []
    ant2_ids = []
    
    for txt_file in txt_files:
        result = process_file(txt_file)
        if result is not None:
            ant1, ant2 = result
            if len(ant1_ids) == 0:
                ant1_ids.append(ant1[0])
            if len(ant2_ids) == 0:
                ant2_ids.append(ant2[0])
            if ant1[0] not in ant1_ids and ant2[0] in ant2_ids:
                ant1_ids.append(ant1[0])
            elif ant1[0] in ant1_ids and ant2[0] not in ant2_ids:
                ant2_ids.append(ant2[0])
            elif ant1[0] in ant2_ids and ant2[0] not in ant1_ids:
                ant1_ids.append(ant2[0])
            elif ant1[0] not in ant2_ids and ant2[0] in ant1_ids:
                ant2_ids.append(ant1[0])

    # print(ant1_ids)
    # print(ant2_ids)

    # Now rewrite each txt with normalized IDs 1 or 2
    for txt_file in txt_files:
        # rewrite each line in each text file:
        # split the line (the last value is the track ID)
        # if the track ID is not 1 or 2, check if it's in ant1_ids or ant2_ids
        # if it's in ant1_ids, change the track ID to 1, if it's in ant2_ids, change it to 2
        # rewrite the new track ID (preserving all other information) back to the txt file
        # --- Step 1: Correct new track IDs for match old for each txt file --- 
        new_lines = []
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # skip malformed lines
                if len(parts) < 6:
                    continue

                # original track ID is in parts[5]
                orig_id = int(parts[5])
                if orig_id not in (1, 2):
                    if orig_id in ant1_ids:
                        parts[5] = '1'
                    elif orig_id in ant2_ids:
                        parts[5] = '2'
                    else:
                        # leave as-is (or you could choose to drop it)
                        print("uh oh spaghetti-o")
                        continue

                # reassemble and keep the newline
                new_lines.append(" ".join(parts) + "\n")

        # overwrite the txt file
        with open(txt_file, 'w') as f:
            f.writelines(new_lines)

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

    while frame_index < total_txt:
        ret, frame = cap.read()
        if not ret:
            break  # video ended early

        txt_file = txt_files[frame_index]
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])
            track_id = int(parts[5])

            if normalized:
                x_center *= frame_width
                y_center *= frame_height
                box_width  *= frame_width
                box_height *= frame_height

            x1 = int(x_center - box_width/2)
            y1 = int(y_center - box_height/2)
            x2 = int(x_center + box_width/2)
            y2 = int(y_center + box_height/2)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Updated tracking video saved as {output_video}")
    return

if __name__ == '__main__':
    process_video("2ants.mp4", "../runs/detect/track (0.5,0.6,0.8) 2.5 mins/labels", "updated_tracking_video.mp4", normalized=True)