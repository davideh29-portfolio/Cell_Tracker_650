import utility_feature_trainer_2 as util
import networkx as nx
import matplotlib.pyplot as plt
import time
import numpy as np
import cv2


class FeatureTrainer:

    def __init__(self, video_img, bin_mask):
        self.video_img = video_img
        self.bin_mask = bin_mask
        # Calculate vector of features for each image
        feats, self.contours, centroid_vect = util.gen_features(video_img, bin_mask)
        # Initialize nodes in graph
        self.G = nx.DiGraph()
        # Add a node for each detection
        self.img_nodes = []
        self.max_track_number = -1
        for i in range(0, len(feats)):
            curr_nodes = []
            for d in range(0, feats[i].shape[1]):
                # centroid = feats[i][0:2, d]
                feats_node = feats[i][:, d]
                centroid_node = centroid_vect[i][:, d]
                curr_nodes.append(self.G.number_of_nodes())
                self.G.add_node(self.G.number_of_nodes(), feats=feats_node, centroid=centroid_node, track_num = [])
            self.img_nodes.append(curr_nodes)
        print "hi"

    # Draws DAG representation of cell tracks
    # Inputs:
    #   with_labels - true if the graph should also draw cell #'s
    def draw_graph(self, with_labels):
        pos = {}
        node_count = 0
        for i in range(0, len(self.img_nodes)):
            for d in range(0, len(self.img_nodes[i])):
                pos[node_count] = ((i + 1), (d + 1))
                node_count += 1
        fig = plt.figure()
        fig.clear()
        nx.draw(self.G, pos, with_labels=with_labels)
        # plt.draw()
        plt.show(block=True)
        raw_input("Hit <Enter> To Continue...")
        plt.close(fig)
        time.sleep(.2)

    #   track_nodes - nx2 (frame, index) vector of nodes already in graph
    def label_display(self, frame, track_nodes, track_frames):

        # Get current frame with track labels
        img_num = cv2.cvtColor(self.bin_mask[frame].copy(), cv2.COLOR_GRAY2RGB)
        contours = self.contours[frame]
        cv2.drawContours(img_num, contours, -1, (255, 255, 255), -1)
        for det in range(0, len(self.img_nodes[frame])):
            contour = contours[det]
            label = det
            x, y, w, h = cv2.boundingRect(contour)
            # Labeled bounding box
            cv2.rectangle(img_num, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_num, "#" + format(label), (x + w + 10, y + h), 2, 0.6, (0, 255, 0))
            cv2.putText(img_num, "#" + format(label), (x - 10, y - 5), 2, 0.6, (0, 255, 0))
        img_num = cv2.copyMakeBorder(img_num, 10, 5, 5, 10, cv2.BORDER_CONSTANT,
                                     value=[255, 255, 255])

        # Get previous frame with track_nodes colored
        prev_frame_ind = frame
        prev_frame_tracks = []
        while not len(prev_frame_tracks):
            prev_frame_ind -= 1
            prev_frame_tracks = np.argwhere(np.array(track_frames) == prev_frame_ind)
            # prev_frame_tracks = [0 if ele != prev_frame_ind else 1 for ele in track_frames ]
        prev_frame_tracks = np.hstack(prev_frame_tracks)
        contours = self.contours[prev_frame_ind]
        img_track = cv2.cvtColor(self.bin_mask[prev_frame_ind].copy(), cv2.COLOR_GRAY2RGB)
        for detection_ind in range(0, len(self.img_nodes[prev_frame_ind])):
            detection = self.img_nodes[prev_frame_ind][detection_ind]
            if len(self.G.node[detection]['track_num']):
                cv2.drawContours(img_track, contours, detection_ind, (0, 255, 0), -1)
        for cont in prev_frame_tracks:
            contour = contours[track_nodes[cont]]
            x, y, w, h = cv2.boundingRect(contour)
            # Labeled bounding box
            cv2.rectangle(img_track, (x, y), (x + w, y + h), (230, 230, 250), 2)
            # cv2.drawContours(img_track, contours, track_nodes[cont], (255, 0, 0), -1)
        img_track = cv2.copyMakeBorder(img_track, 10, 5, 10, 5, cv2.BORDER_CONSTANT,
                                       value=[255, 255, 255])

        # Get original, normalized frame and previous frame
        curr_frame = cv2.copyMakeBorder(self.video_img[frame], 5, 10, 5, 10, cv2.BORDER_CONSTANT,
                                        value=[255, 255, 255])
        prev_frame = cv2.copyMakeBorder(self.video_img[prev_frame_ind], 5, 10, 10, 5, cv2.BORDER_CONSTANT,
                                        value=[255, 255, 255])

        # Append 4 images
        final_img = np.vstack([np.hstack([img_track, img_num]), np.hstack([prev_frame, curr_frame])])

        # Display image and command prompt
        cv2.namedWindow("Track Current Frame")
        cv2.moveWindow("Track Current Frame", 900, 20)
        cv2.imshow("Track Current Frame", final_img)
        print "Type command then press <ENTER>"
        print "Commands:"
        print "\t'#' - add node to path"
        print "\t'n' - skip to next frame\t'p' - go back to previous frame\t'e' - end track"
        print "\t'g' - display network graph\t's' - save graph\t'q' - quit"

        # Get next command
        var = -1
        prev_char = ' '
        while True:
            var = cv2.waitKey(33)
            if var != -1:
                if chr(var).isdigit() and prev_char.isdigit():
                    prev_char += chr(var)
                elif (var == 13 or chr(var) == 'f') and (prev_char == 'n' or prev_char == 'g' or prev_char == 's'
                                    or prev_char == 'e' or prev_char.isdigit() or prev_char == 'q' or prev_char == 'p'
                                                         or prev_char == 'd'):
                    break
                else:
                    prev_char = chr(var)
        time.sleep(.15)
        cv2.destroyAllWindows()
        time.sleep(.15)
        print "You entered: " + prev_char
        return prev_char
