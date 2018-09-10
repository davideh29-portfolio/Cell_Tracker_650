# Import libraries
import utility_functions_5 as util
from CellTracks5_1 import CellTracks
import numpy as np
import pickle
from track_functions import *

# Set paths to data and to trained model
trial = 3
seg_path = "/home/davideh29/PycharmProjects/Cell_Tracker/normalized_frames/video_3/Binary/"
color_path = "/home/davideh29/PycharmProjects/Cell_Tracker/normalized_frames/video_3/Raw/"
out_path = "./output_" + format(trial) + "/"
save_path = "/home/davideh29/PycharmProjects/Cell_Tracker/EdgeAcc/"

# Set parameters
max_md = 3          # Max number of consecutive missed detections allowed
search_dist = 80    # Max distance moved per frame
min_area = 15       # Minimum area of contour to be considered cell
md_penalty = -500   # Penalty for missed detection
count_penalty = 500  # Penalty for high cell count
init_covariance = np.diag([5, 5, 200, 200])  # Initial state covariance
PI = 16*np.eye(2)  # Process covariance
R = 12*np.eye(2)  # Measurement covariance

# Load color images and segmentations
bin_mask, vid_img = util.load_testing_data(seg_path, color_path)

# Get image dimensions
image_height = bin_mask.shape[1]
image_width = bin_mask.shape[2]

# Generate graphical model of detections
cell_tracks = CellTracks(bin_mask=bin_mask, vid_img=vid_img, search_dist=search_dist, min_area=min_area)
num_frames = len(cell_tracks.img_nodes)
all_tracks = []

print "Running Viterbi..."
# Starting frame loop
for start_frame in range(0, num_frames-1):
    print "Starting at frame #" + format(start_frame)
    # Track creation loop
    while True:
        current_frame = start_frame + 1
        complete_tracks = []
        active_tracks = init_tracks(frame=start_frame, graph=cell_tracks, init_cov=init_covariance)
        if len(active_tracks) == 0:
            break
        while current_frame < num_frames:
            if not len(active_tracks):
                break
            # KF prediction step
            kalman_predict(tracks=active_tracks, frame_num=current_frame, max_md=max_md, PI=PI)
            # Calculate most likely track assignment for each current detection
            assignments, scores = assign_tracks(graph=cell_tracks, tracks=active_tracks, frame=current_frame,
                                                search_dist=search_dist, count_penalty=count_penalty)
            # Create tracks for missed detection case
            det_tracks, md_tracks, end_tracks = update_tracks(tracks=active_tracks, assignments=assignments, scores=scores,
                                                              nodes=cell_tracks.img_nodes[current_frame],
                                                              max_md=max_md, md_penalty=md_penalty)
            # Store tracks that ended early
            if len(end_tracks) > 0:
                complete_tracks += end_tracks

            # KF measurement step
            kalman_update(graph=cell_tracks, tracks=det_tracks, R=R)
            active_tracks = det_tracks + md_tracks
            current_frame += 1

        complete_tracks += active_tracks

        if len(complete_tracks):
            # Get most probable overall track (highest score/frame)
            max_score_track = -1
            max_prob = -float('Inf')
            for t in range(0, len(complete_tracks)):
                curr_track_score = complete_tracks[t][-1]['score'] / len(complete_tracks[t])
                # Get index of the most probable track
                if curr_track_score > max_prob:
                    max_score_track = t
                    max_prob = curr_track_score

            if max_score_track >= 0:
                most_prob_track = complete_tracks[max_score_track]
                # Add most probable tracks to track list and DAG
                add_edges(graph=cell_tracks, track=most_prob_track, track_ind=len(all_tracks)-1)
                all_tracks.append(most_prob_track)
                # cell_tracks.draw_tracks(all_tracks)

# Draw and save track output
cell_tracks.G.remove_node(-1)
cell_tracks.draw_tracks(all_tracks)
pickle.dump(cell_tracks, open(save_path + "track_" + format(trial) + ".p", "wb"))
cell_tracks.draw_graph(False)

print "Number of tracks: " + format(len(all_tracks))
