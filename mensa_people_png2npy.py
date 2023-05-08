import os
import site
import numpy as np
try:
    import cv2
except ImportError:
	site.addsitedir('D:\\Program Files\\opencv-4.5.0-dldt-2021.1-vc16-avx2\\opencv\\build\\python')
import cv2

# path to the directory containing the subdirectories
directory_path = "D:\Downloads\mensa_extracted"

# loop through each subdirectory
for participant_dir_name in os.listdir(directory_path):
    participant_dir_path = os.path.join(directory_path, participant_dir_name)
    
    # create empty arrays for the frames
    rgb_frames = []
    depth_frames = []

    # loop through each file in the subdirectory
    for file_name in os.listdir(participant_dir_path):
        
        # check if the file is an rgb or depth frame
        if "rgb.png" in file_name:
            file_path = os.path.join(participant_dir_path, file_name)
            frame_number = int(file_name[:4])
            rgb_frame = cv2.imread(file_path)
            rgb_frame = cv2.resize(rgb_frame, (640, 480))
            rgb_frames.append((frame_number, rgb_frame))
        elif "depth.png" in file_name:
            file_path = os.path.join(participant_dir_path, file_name)
            frame_number = int(file_name[:4])
            depth_frame = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
            depth_frame = cv2.resize(depth_frame, (640, 480))
            depth_frames.append((frame_number, depth_frame))

    # sort the frames by frame number
    rgb_frames.sort(key=lambda x: x[0])
    depth_frames.sort(key=lambda x: x[0])

    # convert the frames to numpy arrays
    rgb_frames = np.array([frame[1] for frame in rgb_frames])
    depth_frames = np.array([frame[1] for frame in depth_frames])

    # save the numpy arrays as npy files
    np.save(os.path.join(directory_path, f"{participant_dir_name}_rgb.npy"), rgb_frames)
    np.save(os.path.join(directory_path, f"{participant_dir_name}_depth.npy"), depth_frames)
