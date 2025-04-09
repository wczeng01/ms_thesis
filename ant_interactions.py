# import os
# import cv2 # type: ignore
# import numpy as np # type: ignore
# import pandas as pd # type: ignore
# from glob import glob # type: ignore

# # Function to check if two bounding boxes overlap
# def check_overlap(box1, box2):
#     x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
#     x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
#     x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
#     x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

#     return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# # Function to filter bounding boxes in the middle region based on x-coordinates
# def filter_middle_region(data, x_min, x_max):
#     return np.array([box for box in data if x_min <= box[0] <= x_max])

# # Main function to process files and assign unique ant IDs
# def track_ant_interactions_with_ids(directory):
#     interaction_dict = {}  # Dictionary to track overlapping ant pairs
#     unique_ant_id = 0  # Counter for assigning unique ant IDs
#     ant_id_mapping = {}  # Map for tracking unique IDs of ants across frames

#     # Middle region cutoffs
#     x_min, x_max = 0.313, 0.704

#     # Get list of text files in the directory
#     files = sorted(glob(os.path.join(directory, "*.txt")))

#     for file_idx, file in enumerate(files):
#         data = pd.read_csv(file, sep='\s+', header=None).iloc[:, 1:].values

#         # Filter bounding boxes in the middle region only
#         filtered_data = np.array([box for box in data])
#         # filtered_data = filter_middle_region(data, x_min, x_max)

#         # Assign unique IDs to ants in the current frame
#         current_ant_ids = {}
#         for i, box in enumerate(filtered_data):
#             if i not in ant_id_mapping:
#                 ant_id_mapping[i] = unique_ant_id
#                 unique_ant_id += 1
#             current_ant_ids[i] = ant_id_mapping[i]

#         # Track overlapping pairs
#         for i in range(len(filtered_data)):
#             for j in range(i + 1, len(filtered_data)):
#                 if check_overlap(filtered_data[i, :4], filtered_data[j, :4]):
#                     ant_pair = tuple(sorted((current_ant_ids[i], current_ant_ids[j])))
#                     if ant_pair not in interaction_dict:
#                         interaction_dict[ant_pair] = 1
#                     else:
#                         interaction_dict[ant_pair] += 1

#     print(f"Total unique interactions tracked: {len(interaction_dict)}")
#     return interaction_dict

# # Directory containing text files
# directory = "runs/detect/full/labels"
# ant_interactions = track_ant_interactions_with_ids(directory)






import os
import numpy as np
import pandas as pd
from glob import glob

def check_overlap(box1, box2):
    """
    Check if two bounding boxes overlap.
    Box format: [x_center, y_center, width, height]
    """
    x1_min, y1_min = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    x1_max, y1_max = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    x2_min, y2_min = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    x2_max, y2_max = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2

    return not (x1_max < x2_min or x1_min > x2_max or y1_max < y2_min or y1_min > y2_max)

# def filter_middle_region(data, x_min, x_max):
#     """
#     Filter bounding boxes whose x-center lies between x_min and x_max.
#     Data columns (after dropping the first column):
#       0: x_center, 1: y_center, 2: width, 3: height, 4: ant ID
#     """
#     return np.array([box for box in data if x_min <= box[0] <= x_max])

def filter_middle_region(data, x_min, x_max):
    """
    Filter bounding boxes that lie completely within the region defined by x_min and x_max.
    Data columns (after dropping the first column):
      0: x_center, 1: y_center, 2: width, 3: height, 4: ant ID
    A box is considered completely within the region if:
      (x_center - width/2) >= x_min  and  (x_center + width/2) <= x_max.
    """
    return np.array([
        box for box in data 
        if (box[0] - box[2] / 2 > x_min) and (box[0] + box[2] / 2 < x_max)
    ])


def merge_intervals(frames, tolerance):
    """
    Given a sorted list of frame indices, merge indices that are within
    'tolerance' frames of each other into continuous segments.
    Returns a list of tuples: (start_frame, end_frame) for each segment.
    """
    if not frames:
        return []
    
    frames = sorted(frames)
    merged = []
    start = frames[0]
    end = frames[0]
    
    for f in frames[1:]:
        # If the gap is small enough, merge it into the current segment.
        if f - end <= tolerance:
            end = f
        else:
            merged.append((start, end))
            start = f
            end = f
    merged.append((start, end))
    return merged

def track_ant_interactions(directory, merge_tolerance=20, min_segment_length=10):
    """
    Process the text files (each representing one frame) to count unique ant interactions.
    
    Parameters:
      directory (str): Folder containing the .txt files.
      merge_tolerance (int): Maximum gap (in frames) between detections to consider them part of the same event.
      min_segment_length (int): Minimum number of frames for a continuous event to count as an interaction.
    
    Returns:
      total_interactions (int): The number of unique (merged) interactions detected.
    """
    # Middle region x-center cutoffs
    x_min, x_max = 0.313, 0.704

    # Get list of text files (one per frame), sorted in order.
    files = sorted(glob(os.path.join(directory, "*.txt")))
    
    # Dictionary to hold, for each ant pair, a list of frame indices where they overlapped.
    pair_frames = {}

    for frame_index, file in enumerate(files):
        # Read the data and drop the first column.
        # After dropping, columns are: [x_center, y_center, width, height, ant ID]
        data = pd.read_csv(file, sep='\s+', header=None).iloc[:, 1:].values
        
        # Filter out detections not in the middle region.
        filtered_data = filter_middle_region(data, x_min, x_max)
        n = len(filtered_data)
        
        # Check every pair for overlap.
        for i in range(n):
            for j in range(i + 1, n):
                box1 = filtered_data[i, :4]
                box2 = filtered_data[j, :4]
                if check_overlap(box1, box2):
                    # Use the provided ant IDs (convert to int for consistency)
                    ant1 = int(filtered_data[i, 4])
                    ant2 = int(filtered_data[j, 4])
                    pair = tuple(sorted((ant1, ant2)))
                    if pair not in pair_frames:
                        pair_frames[pair] = []
                    pair_frames[pair].append(frame_index)

    # Now, for each ant pair, merge detections that occur in close succession,
    # and count only those segments that last long enough.
    total_interactions = 0
    for pair, frames in pair_frames.items():
        intervals = merge_intervals(frames, merge_tolerance)
        # Count only intervals that are long enough.
        for (start, end) in intervals:
            if (end - start + 1) >= min_segment_length:
                total_interactions += 1

    print(f"Total unique interactions tracked: {total_interactions}")
    return total_interactions

# Directory containing the text files (each file corresponds to one frame)
directory = "runs/detect/full/labels"

# You may need to adjust merge_tolerance and min_segment_length to get closer to 30–40 total interactions.
ant_interactions_count = track_ant_interactions(directory, merge_tolerance=200, min_segment_length=270)

"""
merge_tolerance: If two overlapping detections for a given pair are separated by no more than this many frames, they are merged into one event.

min_segment_length: An event must span at least this many frames to count as an interaction.
200/180 - 60
200/270 - 34
200/300 - 29
"""