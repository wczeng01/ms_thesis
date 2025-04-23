import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from ultralytics import YOLO #type:ignore
model = YOLO('best3-3(v11m_50).pt')
#model2 = YOLO('yolov8m.pt')

# 3/13/25:
# i think the difference is that model.track doesn't see the larva because it doesn't move?
# results = model.predict(source='2ants_cropped.mp4', show=False, stream=False, show_labels=True, show_boxes=True, save=True, save_txt=True, line_width=1, tracker='bytetrack.yaml', conf=0.1)

results = model.track(source='2ants.mp4', show=False, stream=False, show_labels=True, show_boxes=True, save=True, save_txt=False, line_width=1, tracker='botsort.yaml', persist=True)

# results = model.predict(source='2ants_cropped.jpg', save=True, line_width=2, conf=0.1)

#results = model.track(source='C0013.mp4', show=True, stream=False, show_labels=True, show_boxes=True, save=False, save_txt=False, line_width=1, tracker='bytetrack.yaml', conf=0)
#results = model.track(source='C0013.mp4', show=False, stream=False, show_labels=True, show_boxes=True, save=False, save_txt=False, save_conf=False, line_width=1, tracker='bytetrack.yaml')
# Process and save results
# for result in results:
#     result.save(filename="uploads/result.jpg")
# print("Tracking completed successfully.")

# 3/4/25:
# model.predict doesn't include IDs

# 3/18/25: python ant.py >nul 2>&1