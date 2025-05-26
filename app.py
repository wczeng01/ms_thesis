from flask import Flask, render_template, request, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename

# Tracking modules
from ultralytics import YOLO  # type: ignore
import false_positive
import test_track
import correct_tracks

BASE_DIR = os.path.dirname(__file__)
TRACKER_CFG = os.path.join(BASE_DIR, "bytetrack.yaml")

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
def init_upload_folder():
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
init_upload_folder()

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Default brood coordinates and threshold
default_brood_x = 0.535875

default_brood_y = 0.623067

proximity_threshold = 0.01

@app.route('/brood', methods=['GET'])
def brood_form():
    # Show upload form and any results
    return render_template('brood.html')

@app.route('/upload_videos', methods=['POST'])
def upload_videos():
    files = request.files.getlist('videos')
    if not files:
        flash('No videos uploaded.')
        return redirect(url_for('brood_form'))

    outcomes = []  # list of (video_name, winner)

    for f in files:
        if not (f and allowed_file(f.filename)):
            continue
        filename = secure_filename(f.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(save_path)

        # 1. Run YOLO tracking
        try:
            model = YOLO('best3-3(v11m_50).pt')
            model.track(
                source=save_path,
                show=False,
                stream=False,
                show_labels=True,
                show_boxes=True,
                save=True,
                save_txt=True,
                line_width=1,
                tracker=TRACKER_CFG,
                persist=True
            )
        except Exception as e:
            flash(f'YOLO tracking failed for {filename}: {e}')
            continue

        # 2. Locate the new track folder
        track_folder = false_positive.get_most_recent_folder('runs/detect')
        if not track_folder:
            flash(f'Could not locate track folder for {filename}.')
            continue
        txt_dir = os.path.join(track_folder, 'labels')

        # 3 & 4. Remap track IDs and re-render updated video
        updated_video = os.path.splitext(filename)[0] + '_updated.mp4'
        try:
            correct_tracks.process_video(
                video_path=save_path,
                txt_dir=txt_dir,
                output_video=updated_video,
                normalized=True
            )
        except Exception as e:
            flash(f'Error in ID correction/rendering for {filename}: {e}')
            continue

        # 5. Proximity analysis
        count1 = 0
        count2 = 0
        # iterate through all label files
        for entry in sorted(os.listdir(txt_dir)):
            if not entry.lower().endswith('.txt'):
                continue
            path_txt = os.path.join(txt_dir, entry)
            with open(path_txt, 'r') as tf:
                lines = [ln.strip() for ln in tf if ln.strip()]

            # default brood position
            bx, by = default_brood_x, default_brood_y
            # find brood detection if present
            for ln in lines:
                parts = ln.split()
                if len(parts) < 6:
                    continue
                if parts[0] == '1':
                    bx, by = float(parts[1]), float(parts[2])
                    break

            # check each ant
            for ln in lines:
                parts = ln.split()
                if len(parts) < 6 or parts[0] != '0':
                    continue
                ax, ay = float(parts[1]), float(parts[2])
                tid = int(parts[5])
                dist = test_track.distance((ax, ay), (bx, by))
                if dist <= proximity_threshold:
                    if tid == 1:
                        count1 += 1
                    elif tid == 2:
                        count2 += 1

        # 6. Determine winner
        if count1 > count2:
            winner = 'Ant 1'
        elif count2 > count1:
            winner = 'Ant 2'
        else:
            winner = 'Tie'
        outcomes.append((filename, winner))

    # 7. Render results back to page
    return render_template('brood.html', outcomes=outcomes)

if __name__ == '__main__':
    app.run(debug=True)
