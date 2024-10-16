# 3D object distance estimation from the BEV representations
This project shows the LiDAR-to-image-based method for metric distance estimation with 3D bounding box projections onto the image. We demonstrate that despite the general difficulty of the BEV representation in understanding features related to the height coordinate, it is possible to extract all parameters characterizing the bounding boxes of the objects, including their height and elevation. Finally, we applied the triangulation method to calculate the accurate distance to the objects and statistically proved that our methodology is one of the best in accuracy and robustness.

This is the extension code with implemented Extended Kalman filter-based tracker for the [Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds](https://github.com/maudzung/SFA3D)


## Highlights
1. [The code became part of the research paper](https://www.mdpi.com/1424-8220/23/4/2103) 
2. Fast training, fast inference
3. Anchor-free approach
4. Support distributed parallel training
5. Additionally EKF was implemented

## Demo Video
[![Watch the demo video](https://img.youtube.com/vi/hAoCTLpyiWw/0.jpg)](https://www.youtube.com/watch?v=hAoCTLpyiWw)


## Getting started
Follow the requirements, data preparation, and how-to-run steps that are well-described in [SFA3D](https://github.com/maudzung/SFA3D)

## EKF implementation

The main idea of the tracking mechanism is in multiple key features that include initialization, prediction, correction track management, track update algorithm, and track creation and deletion. We applied the Extended Kalman Filter method for the track prediction and correction of the frames. 
The tracker backbone was written in _tracker.py_. Refer to the **def draw_prediction** function inside _evaluation_utils.py_ for the EKF initialization and drawing results.
```
 # Tracker EKF
    tracker.Update(detection_centers)
    valid_tracks = []
    for track in tracker.tracks:
        if 0 <= track.track_id % 6 < len(track_colors):
            valid_tracks.append(track)
        else:
            print("Invalid track ID:", track.track_id)
    tracker.tracks = valid_tracks
    for track in tracker.tracks:
            track_centers.append(track.KF.predict())
```
## Results
Check out the results of the work

![result](./pic_track.png)

## Citation

```bash
@article{article,
author = {Usmankhujaev, Saidrasul and Baydadaev, Shokhrukh and Woo, Jang},
year = {2023},
month = {02},
pages = {},
title = {Accurate 3D to 2D Object Distance Estimation from the Mapped Point Cloud Data},
doi = {10.3390/s23042103}
}
