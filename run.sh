!/bin/bash

export FLASK_APP=app.py
export FLASK_DEBUG=1  # Enable debug mode

# Run the Flask application on port 8080
flask run --host=0.0.0.0 --port=8080