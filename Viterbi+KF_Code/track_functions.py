import numpy as np
from scipy.linalg import inv
import copy


# Initializes tracks for a given frame at nodes with no track already through them
def init_tracks(frame, graph, init_cov):
    tracks = []
    # Create a dictionary for each detection in the specified frame with no existing track
    for i in range(0, len(graph.img_nodes[frame])):
        node_ind = graph.img_nodes[frame][i]
        node = graph.G.node[node_ind]
        has_track = len(node['tracks']) > 0
        if not has_track:
            state = np.array([node['centroid'][0], node['centroid'][1], 0, 0])
            dict = {"frame": frame, "global_ind": node_ind, "state": state, "md": 0, "cov": init_cov, "score": 0}
            one_track = [dict]
            tracks.append(one_track)
    return tracks


# Creates new track frame dictionary and predicts next position of tracks based on state (pos, vel)
def kalman_predict(tracks, frame_num, max_md, PI):
    dt = 0.1
    A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    V = np.array([[dt, 0], [0, dt], [1, 0], [0, 1]])

    # For every track input, predict and append to the track list
    for i in range(0, len(tracks)):
        state = tracks[i][-1]['state']
        missed_det = tracks[i][-1]['md']
        cov = np.asarray(tracks[i][-1]['cov'])

        # Predict state only if number of missed detections is less than maximum
        if missed_det <= max_md:
            # Predict next state
            pred_state = np.dot(A, state)

            # Predict next covariance
            pred_cov = np.dot(A, np.dot(cov, A.T)) + np.dot(V, np.dot(PI, V.T))

            # Make a dictionary with prediction values
            dict = {"frame": frame_num, "global_ind": -1, "state": pred_state, "md": 0, "cov": pred_cov, "score": 0}

            # Append to track
            tracks[i].append(dict)
    return


# Assigns tracks to detections
def assign_tracks(graph, tracks, frame, search_dist, count_penalty):
    node_inds = graph.img_nodes[frame]
    max_prob_tracks = -1*np.ones(len(node_inds), dtype=int)
    max_prob_scores = 100*np.ones(len(node_inds))
    # For each detection in frame, assign a track
    for i in range(0, len(node_inds)):
        node = graph.G.node[node_inds[i]]
        track_probs = np.empty(len(tracks))
        in_dist_count = 0
        for t in range(0, len(tracks)):
            node_pos = node['centroid']
            track_pos = tracks[t][-1]['state'][0:2]
            track_cov = tracks[t][-1]['cov'][0:2, 0:2]
            prob_density = -float('Inf')
            # Calculate prob density for tracks within max search distance
            if np.linalg.norm(track_pos - node_pos) < search_dist:
                in_dist_count += 1
                prob_density = -.5*np.dot((track_pos - node_pos).T,
                                          np.dot(np.linalg.inv(track_cov), track_pos - node_pos))
                prob_density -= np.log(np.sqrt((2*np.pi)**2*np.linalg.det(track_cov)))
                if len(node['tracks']) > 0:
                    prob_density += count_penalty*np.log(1./(2.**len(node['tracks'])))
            track_probs[t] = prob_density
        if not in_dist_count == 0:
            # Only assign track if a track is found within search_dist
            max_prob_tracks[i] = np.argmax(track_probs)
            max_prob_scores[i] = np.max(track_probs)
    return max_prob_tracks, max_prob_scores


# Updates tracks with assigned detections, removes tracks without assigned detections
# and creates new tracks representing missed detections
def update_tracks(tracks, assignments, scores, nodes, max_md, md_penalty):
    # Loop through tracks
    det_tracks = []
    md_tracks = []
    end_tracks = []
    # Keep tracks with assigned detections
    for a in range(0, len(assignments)):
        t = assignments[a]
        if not t == -1:
            det_track = tracks[t][0:-1]
            det_track.append(copy.deepcopy(tracks[t][-1]))
            det_track[-1]['global_ind'] = nodes[a]
            det_track[-1]['score'] = det_track[-2]['score'] + scores[a]
            det_tracks.append(det_track)
    for t in range(0, len(tracks)):
        # Create missed-detection tracks
        if tracks[t][-1]['md'] < max_md:
            md_track = tracks[t][0:-1]
            md_track.append(copy.deepcopy(tracks[t][-1]))
            md_track[-1]['md'] = md_track[-2]['md'] + 1
            md_track[-1]['score'] = md_track[-2]['score'] + md_penalty
            md_tracks.append(md_track)
        else:
            md_track = tracks[t][0:-1]
            if len(md_track):
                end_tracks.append(md_track)
    return det_tracks, md_tracks, end_tracks


# Applies measurement step updates to tracks with assigned detections (not missed detections)
def kalman_update(graph, tracks, R):
    # Initialize params
    H = np.zeros([2, 4])
    H[0, 0] = 1
    H[1, 1] = 1

    # Loop through tracks
    for t in range(0, len(tracks)):
        node_ind = tracks[t][-1]['global_ind']
        track = tracks[t][-1]
        # Get detection centroid
        node_centroid = graph.G.node[node_ind]['centroid']
        # Apply measurement step update
        temp = np.dot(H, np.dot(track['cov'], H.T)) + R
        K = np.dot(track['cov'], np.dot(H.T, inv(temp)))
        innovation = node_centroid - np.dot(H, track['state'])
        track['state'] += np.dot(K, innovation)
        track['cov'] = np.dot((np.eye(4) - np.dot(K, H)), track['cov'])
    return


# Adds edges corresponding to optimal tracks to final DAG
def add_edges(graph, track, track_ind):
    # Loop through frames
    f = len(track)-1
    prev_node = -1
    while f >= 0:
        # Check if md > 0
        if track[f]['md'] > 1:
            f -= track[f]['md']
            continue
        else:
            curr_node = track[f]['global_ind']
            f -= 1
        if prev_node < 0:
            prev_node = curr_node
        else:
            graph.G.node[prev_node]['tracks'].append(track_ind)
            graph.G.add_edge(curr_node, prev_node)
            prev_node = curr_node
    if curr_node >= 0:
        graph.G.node[curr_node]['tracks'].append(track_ind)
    return
