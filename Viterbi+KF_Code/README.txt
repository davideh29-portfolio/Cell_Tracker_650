KalmanHungarianCode:

Folder contents:
├── tracker_main.py: Main method to execute Viterbi tracker
├── CellTracks5_1.py: Functions for feature extraction from detections, and for visualization of forest graph
├── track_functions.py: Functions to perform Kalman Filter updates and manage track probabilities/assignments to detections
└── utility_functions_5.py: Helper functions to aid execution of main method

Code execution instructions for Viterbi tracker:
1. Modify the paths for the data set to be tested in lines 10 and 11 of tracker_main.py
2. Tracker output images are stored in a folder named 'output'.
3. A forest graph of detections across frames will be displayed at the end of code execution.
4. A pickle file (called save.p) of tracker object will be saved in current directory, which can later be used for accuracy assessment.
