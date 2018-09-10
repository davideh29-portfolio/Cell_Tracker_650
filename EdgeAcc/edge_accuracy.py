import networkx as nx
import pickle
import cv2

min_area = 20
num_frames = 20
# Set load paths
viterbi_output = "/Users/nivedha.sivakumar94/Desktop/LR/Final_Project/Tracker_KF-master/EdgeAcc/save.p"
ground_truth = "/Users/nivedha.sivakumar94/Desktop/LR/Final_Project/Tracker_KF-master/EdgeAcc/video_3.p"
# Load 2 graphs
# graph_ground_truth = CellTracks(video_img, bin_mask, weights)
graph_test = pickle.load(open(viterbi_output, "rb"))
graph_ground_truth = pickle.load(open(ground_truth, "rb"))
graph_ground_truth = graph_ground_truth[0]

# Trim both to num_frames
for frame in range(num_frames, len(graph_ground_truth.img_nodes)):
    for node in graph_ground_truth.img_nodes[frame]:
        graph_ground_truth.G.remove_node(node)

for frame in range(num_frames, len(graph_test.img_nodes)):
    for node in graph_test.img_nodes[frame]:
        graph_test.G.remove_node(node)

# Enforce min_area on ground-truth graph154
for frame in range(0, num_frames)[::-1]:
    for node_ind in range(0, len(graph_ground_truth.img_nodes[frame])):
        node = graph_ground_truth.img_nodes[frame][node_ind]
        contour = graph_ground_truth.contours[frame][node_ind]
        if cv2.contourArea(contour) < min_area:
            graph_ground_truth.G.remove_node(node)

# Get edge statistics...
# Exist in first graph but not second
graph_ground_truth.G = nx.convert_node_labels_to_integers(graph_ground_truth.G)
graph_test.G = nx.convert_node_labels_to_integers(graph_test.G)
gt_diff_vt = nx.difference(graph_ground_truth.G, graph_test.G)
vt_diff_gt = nx.difference(graph_test.G, graph_ground_truth.G)
# Exist in either but not both
symm_diff = nx.symmetric_difference(graph_ground_truth.G, graph_test.G).number_of_edges()
# Exist in both
intersect = nx.intersection(graph_ground_truth.G, graph_test.G).number_of_edges()

# Calculate error
print "Number of edges in truth: " + format(graph_ground_truth.G.number_of_edges())
print "Number of edges in test: " + format(graph_test.G.number_of_edges())
print "Edges in both: " + format(float(intersect))
print "Diff truth vs test: " + format(float(gt_diff_vt.number_of_edges()))
print "Diff test vs truth: " + format(float(vt_diff_gt.number_of_edges()))
print "Total different: " + format(float(symm_diff))
