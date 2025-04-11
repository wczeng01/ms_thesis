from flask import Flask, render_template, request, redirect, url_for, flash #type:ignore
import os
import subprocess
import signal
import false_positive
import test_track
from ultralytics import YOLO #type:ignore

app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

yolo_process = None
uploaded_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')

@app.route('/')
def index():
    # Home page: Upload a video.
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global yolo_process
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(request.url)
    
    if file:
        file.save(uploaded_video_path)
        flash('Video uploaded successfully. Now starting the tracking process...')
        
        try:
            yolo_process = subprocess.Popen(
                ['python', 'run_tracking.py', uploaded_video_path, "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            flash(f"Error running tracking process: {e}")
            return redirect(url_for('index'))
        
        flash("Tracking process completed. Now proceed to process false/true positive IDs.")
    return redirect(url_for('process_form'))

@app.route('/process_form', methods=['GET'])
def process_form():
    return render_template('process.html')

@app.route('/process', methods=['POST'])
def process_ids():
    fp_ids = request.form.get('false_positive_ids', '')
    tp_ids = request.form.get('true_positive_ids', '')
    duplicate_choice = request.form.get('duplicate_choice', 'no')
    
    # --- Optional: Duplicate the labels directory for backup if requested ---
    if duplicate_choice.lower() == 'yes':
        # Find the most recent tracking directory.
        most_recent = false_positive.get_most_recent_folder("runs/detect")
        if most_recent is None:
            flash("Could not find tracking directory for backup.")
            return redirect(url_for('index'))
        source_dir = os.path.join(most_recent, "labels")
        destination_dir = source_dir + "_backup"
        false_positive.duplicate_directory(source_dir, destination_dir)
        flash("Directory duplicated for backup.")
    
    # --- Find the most recent tracking directory for processing ---
    most_recent = false_positive.get_most_recent_folder("runs/detect")
    if most_recent is None:
        flash("Could not find a valid tracking directory.")
        return redirect(url_for('index'))
    txt_dir = os.path.join(most_recent, "labels")
    
    video_path = uploaded_video_path
    output_video = "updated_tracking_video.mp4"
    
    try:
        processed_video = false_positive.process_video(tp_ids, fp_ids, video_path, txt_dir, output_video, normalized=True)
        flash(f"Updated tracking video saved as {processed_video}")
    except Exception as e:
        flash(f"Error processing false/true positive IDs: {e}")
        print("Error processing false/true positive IDs:", e)
    
    return redirect(url_for('process_form'))

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global yolo_process
    if yolo_process:
        if yolo_process.poll() is None:
            try:
                yolo_process.send_signal(signal.SIGTERM)
                yolo_process.wait()
                yolo_process = None
                flash('Tracking process stopped successfully.')
                print("Tracking process stopped.")
            except Exception as e:
                flash(f"Error stopping the tracking process: {e}")
                print(f"Error stopping the tracking process: {e}")
        else:
            flash('Tracking process is already completed.')
            print("Tracking process had already completed.")
    else:
        flash('No active tracking process.')
    return redirect(url_for('process_form'))

@app.route('/stop_tracking2', methods=['POST'])
def stop_tracking2():
    global yolo_process
    if yolo_process:
        if yolo_process.poll() is None:
            try:
                yolo_process.send_signal(signal.SIGTERM)
                yolo_process.wait()
                yolo_process = None
                flash('Tracking process stopped successfully.')
                print("Tracking process stopped.")
            except Exception as e:
                flash(f"Error stopping the tracking process: {e}")
                print(f"Error stopping the tracking process: {e}")
        else:
            flash('Tracking process is already completed.')
            print("Tracking process had already completed.")
    else:
        flash('No active tracking process.')
    return redirect(url_for('process_form'))

@app.route('/skip_tracking', methods=['POST'])
def skip_tracking():
    if yolo_process:
        if yolo_process.poll() is not None:
            flash('Tracking in progress, unable to skip.')
    else:
        flash('Moving to processing.')
        return redirect(url_for('process_form'))
    
@app.route('/skip_tracking2', methods=['POST'])
def skip_tracking2():
    if yolo_process and yolo_process.poll() is None:
        flash('Tracking still in progress, unable to skip.')
        return redirect(url_for('index'))
    flash('Moving to brood page.')
    return redirect(url_for('brood_form'))
    
@app.route('/return_to_upload', methods=['POST'])
def return_to_upload():
    flash('Returning to Upload Page.')
    return redirect(url_for('index'))

@app.route('/brood_form', methods=['GET'])
def brood_form():
    return(render_template('brood.html'))

@app.route('/brood', methods=['POST'])
def brood():
    global yolo_process
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(request.url)
    
    if file:
        file.save(uploaded_video_path)
        flash('Video uploaded successfully. Now starting the brood tracking process...')
        
        try:
            yolo_process = subprocess.Popen(
                ['python', 'run_tracking.py', uploaded_video_path, "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except Exception as e:
            flash(f"Error running tracking process: {e}")
            return redirect(url_for('index'))
        
        flash("Tracking process completed. Now proceed to brood analysis.")
    return redirect(url_for('brood_form'))

@app.route('/brood2', methods=['POST'])
def brood2():
    most_recent = false_positive.get_most_recent_folder("../runs/detect")
    if most_recent is None:
        flash("Could not find tracking directory.")
        return redirect(url_for('index'))
    else:
        flash(most_recent)
    txt_dir = os.path.join(most_recent, "labels")

    # 0.535875 0.623067
    larva_pos = request.form.get('larva_pos', '')
    larva_x = larva_pos.split(",")[0]
    larva_y = larva_pos.split(",")[1]
    flash("Larva's x coordinate: ", larva_x)
    flash("Larva's y coordinate: ", larva_y)
    test_track.main(txt_dir, larva_x, larva_y)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part in the request.')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected.')
        return redirect(request.url)
    
    model = YOLO('best3-3(v11m_50).pt')
    results = model.predict(source=file, save=True, conf=0.1)

if __name__ == '__main__':
    app.run(debug=True)