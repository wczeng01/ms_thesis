from flask import Flask, render_template, request, redirect, url_for, flash
import os
import subprocess
import signal

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Track the YOLO process
yolo_process = None

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global yolo_process

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'video.mp4')
        file.save(filepath)
        print(f"File saved to {filepath}")

        # Start YOLO tracking by calling the external script with the video path
        yolo_process = subprocess.Popen(
            ['python', 'run_tracking.py', filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        flash('Video uploaded and tracking started!')
        return redirect(url_for('upload_form'))

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global yolo_process
    if yolo_process:
        # Check if the process is still running
        if yolo_process.poll() is None:  # None means it's still running
            try:
                yolo_process.send_signal(signal.SIGTERM)  # Send SIGTERM to stop the process
                yolo_process.wait()  # Wait for the process to terminate
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
        print("No active tracking process.")
    return redirect(url_for('upload_form'))

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
