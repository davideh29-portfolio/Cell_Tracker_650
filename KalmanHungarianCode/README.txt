KalmanHungarianCode:

Folder contents:
├── tracker_main.py: Main method to execute Kalman-Hungarian tracker
├── CellTracks5_1.py: Functions for feature extraction from detections, and for visualization of forest graph
├── track_functions.py: Functions to initialize tracker, perform prediction and updation steps (with Hungarian algorithm) for Kalman Filter 
└── utility_functions_5.py: Helper functions to aid execution of main method

Code execution instructions for Kalman-Hungarian tracker:
1. Modify the path of data set to be tested in lines 10 and 11 of tracker_main.py
2. Tracker output images are stored in a folder named 'output'.
3. A forest graph of detections across frames will be displayed at the end of code execution.
4. A pickle file (called save.p) of tracker object will be saved in current directory, which can later be used for accuracy assessment.

Code execution for Accuracy measurement:
1. Move the save.p pickle file into the folder EdgeAcc.
2. Modify the path to save.p and ground truth being tested in lines 8 and 9 of edge_accuracy.py, and run code to view accuracy measures.
