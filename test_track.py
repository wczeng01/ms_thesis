import os
import glob
import math
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
from scipy.signal import savgol_filter # type: ignore
from collections import defaultdict
from pathlib import Path

def handle_outliers(data, mad_factor=3.0):
    """
    Replace outliers in `data` with the median, where outliers
    are defined by a multiple of the median absolute deviation (MAD).
    """
    if len(data) < 3:
        return data  # Too short to compute a meaningful MAD

    data = np.array(data, dtype=float)
    median_val = np.median(data)
    abs_deviation = np.abs(data - median_val)
    mad = np.median(abs_deviation)

    if mad == 0:
        # If all points are identical, there's nothing to replace
        return data

    # Outlier threshold
    threshold = mad_factor * mad

    # Indices of outliers
    outlier_indices = abs_deviation > threshold
    # Replace outliers with the median
    data[outlier_indices] = median_val

    return data

def distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

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
    
def main(directory, x, y):
    # Specify the directory containing the txt files.
    txt_files = sorted(Path(directory).iterdir(), key=os.path.getmtime)
    l_x_center, l_y_center = x, y
    # Process only every nth file.
    interval = 60
    txt_files = txt_files[::interval]

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
    # Lists to store midpoints for each object across frames.
    ant1_list, ant2_list = [], []
    
    for txt_file in txt_files:
        result = process_file(txt_file)
        if result is not None:
            ant1, ant2 = result
            if ant1[0] in ant1_ids and ant2[0] in ant2_ids:
                ant1_list.append(ant1[1:])
                ant2_list.append(ant2[1:])
            else:
                ant1_list.append(ant2[1:])
                ant2_list.append(ant1[1:])
    
    # print(ant1_list[:10])
    # print(len(ant1_list))
    # print(ant2_list[:10])
    # print(len(ant2_list))

    # Compute distances for each frame.
    l = [l_x_center, l_y_center]
    dist_ant1_larva = np.array([distance(a1, l) for a1 in ant1_list])
    dist_ant2_larva = np.array([distance(a2, l) for a2 in ant2_list])
    dist_ant1_ant2  = np.array([distance(a1, a2) for a1, a2 in zip(ant1_list, ant2_list)])
    
    # 1) Outlier removal
    # dist_ant1_larva_no_outliers = handle_outliers(dist_ant1_larva)
    # dist_ant2_larva_no_outliers = handle_outliers(dist_ant2_larva)
    # dist_ant1_ant2_no_outliers  = handle_outliers(dist_ant1_ant2)

    # 2) Savitzky–Golay filtering
    # dist_ant1_larva_no_outliers = savgol_filter(dist_ant1_larva_no_outliers, 9, 2)
    # dist_ant2_larva_no_outliers = savgol_filter(dist_ant2_larva_no_outliers, 9, 2)
    # dist_ant1_ant2_no_outliers  = savgol_filter(dist_ant1_ant2_no_outliers,  9, 2)

    # Plot the distances.
    frames = list(range(1, len(ant1_list) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(frames, dist_ant1_larva, label="Distance: Ant1 - Larva")
    plt.plot(frames, dist_ant2_larva, label="Distance: Ant2 - Larva")
    plt.plot(frames, dist_ant1_ant2, label="Distance: Ant1 - Ant2")
    plt.xlabel("Frame")
    plt.ylabel("Scaled Distance")
    plt.title("Distance vs Frame")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()