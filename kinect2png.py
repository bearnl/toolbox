import os
import numpy as np
import cv2
from pathlib import Path

try:
    import segment_k4a
except FileNotFoundError:
    print('Azure Kinect SDK DLL not found')
except ImportError:
    print('Unable to import segment_k4a')

label_map = {
    '3': 'lady1', 
    '4': 'lady1', 
    '7': 'lady2', 
    '8': 'lady2', 
    '5': 'man1', 
    '6': 'man2', 
    '1': 'stephen', 
    '2': 'stephen', 
    '9': 'stephen'
}

def get_images_from_recording(playback):
    for cap in playback.get_capture_iter():
        colour = cap.get_colour_image()
        depth = cap.get_depth_image()
        if colour is None or depth is None:
            print('colour or depth image is missing. skip this capture')
            continue
        yield depth, colour

def generator(paths):
    for path in paths:
        # Extract the dataset name from the path
        dataset_name = os.path.basename(path)
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.mkv')]
        for file in files:
            filename = os.path.basename(file)
            label = None
            if 'leo-recording' in path:
                label = 'leo'
            elif 'mycapture' in path:
                label = filename.split('-')[2]
            elif 'stephen-stair-dk' in path:
                label = label_map[filename.split('-')[-1].replace('.mkv', '')]

            print('processing file', file, 'label', label)
            ctx = segment_k4a.K4a()
            playback = ctx.playback_open(file)
            img_iter = get_images_from_recording(playback)
            
            # Create a directory if not exists
            output_dir = Path('../inhouse/' + dataset_name)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for idx, (depth, color) in enumerate(img_iter):
                depth = cv2.resize(depth, (256, 256))
                color = cv2.resize(color, (256, 256))

                # Save images as "<dataset_name>_<label>_<seq>_<rgb|depth>.png"
                depth_filename = output_dir / f"{label}_{idx}_depth.png"
                color_filename = output_dir / f"{label}_{idx}_rgb.png"
                cv2.imwrite(str(depth_filename), depth.astype(np.uint16))
                cv2.imwrite(str(color_filename), color[..., :3])

generator([
    '/home/chenzz/projects/def-czarnuch/chenzz/rawdata/leo-recording',
    '/home/chenzz/projects/def-czarnuch/chenzz/rawdata/mycapture',
    '/home/chenzz/projects/def-czarnuch/chenzz/rawdata/stephen-stair-dk',
    ])
