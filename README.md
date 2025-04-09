# Ant Multi-Object Tracking Interface
Some information about the app

## Starting the app
1. Ensure that Python is installed
2. Run `./deploy.sh` from the app directory if first run, run `./run.sh` otherwise
3. The app should now be running on localhost:8080. 

## Understanding the app
* The deploy script configures a virtual environment in which
the user can run the app without installing various dependecies on their
machine.
* Video uploads are stored in ~/uploads, a directory that is created 
on deployment and deleted on termination. By default, Flask serves static files
from a static view that takes a path relative to the app/static directory.
* Results can be found in runs/detect/track_, where _ is an integer, the larger _
is the more recent the tracking run.
