import numpy as np
import cv2
from os import listdir, rename
from os.path import isfile, join


def convert_binary(color_path, img_path):
    # Get file names
    img_files = [f for f in listdir(color_path) if isfile(join(color_path, f))]
    # Loop through images
    for i in range(0, len(img_files)):
        # Load segmentation image
        img = cv2.imread(color_path + img_files[i], 0)
        img[img != 204] = 255
        img[img == 204] = 0
        cv2.imwrite(img_path + img_files[i], img_path + img)
    return


def convert_numerical(img_path):
    img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    for i in range(0, len(img_files)):
        b = img_files[i].replace(".", "_")
        a = [s for s in b.split("_") if s.isdigit()]
        rename(img_path + img_files[i], a[0])
    return


def load_testing_data(normalized_frames_path, binary_masks_path):
    # Get file names
    video_files = [int(f.split(".")[0]) for f in listdir(normalized_frames_path) if isfile(join(normalized_frames_path, f))]
    bin_mask_files = [int(f.split(".")[0]) for f in listdir(binary_masks_path) if isfile(join(binary_masks_path, f))]
    # Sort to make sure color images and masks line up
    video = 1
    video_files = np.sort(video_files)
    bin_mask_files = np.sort(bin_mask_files)
    # List of image red pixel values
    video_images = []
    bin_mask = []
    # Loop through images
    for i in range(0, len(video_files)):
        # Load color image
        video_images.append(cv2.imread(normalized_frames_path + format(video_files[i]) + ".tif", 1))
        # Load segmentation image
        bin_mask.append(cv2.imread(binary_masks_path + format(bin_mask_files[i]) + ".png", 0))
    # Convert to ndarray and return
    video_images = np.array(video_images)
    bin_mask = np.array(bin_mask)
    return video_images, bin_mask


def gen_features(video_img, bin_mask):
    # Output vector of features
    feat_vect = []
    contour_vect = []
    centroid_vect = []
    all_feats = []
    for i in range(0, len(video_img)):
        # Segmentation image
        bw_img = bin_mask[i]
        # Find contours
        ret, thresh = cv2.threshold(bw_img, 127, 255, 0)
        _, contours, _ = cv2.findContours(thresh, 1, 2)
        contour_vect.append(contours)
        # Feature vector for current image
        img_feats = np.empty([16, len(contours)])
        centroid_contours = np.empty([2, len(contours)])
        for c in range(0, len(contours)):
            cnt = np.squeeze(contours[c])
            M = cv2.moments(cnt)
            # Centroid
            centroid = np.array([M['m10']/M['m00'], M['m01']/M['m00']])
            centroid_contours[:, c] = centroid
            img_feats[0, c] = centroid[0]
            img_feats[1, c] = centroid[1]
            # Area
            img_feats[2, c] = cv2.contourArea(cnt)
            # Perimeter
            img_feats[3, c] = cv2.arcLength(cnt, True)
            # Calculate distances from centroid and circularity measures
            dist = np.sum((cnt - centroid)**2, axis=1)
            v11 = np.sum(np.prod(cnt-centroid, axis=1))
            v02 = np.sum(np.square(cnt - centroid)[:, 1])
            v20 = np.sum(np.square(cnt - centroid)[:, 0])
            # Circularity
            m = 0.5 * (v02 + v20)
            n = 0.5 * np.sqrt(4 * v11**2 + (v20 - v02)**2)
            img_feats[4, c] = (m - n) / (m + n)
            # Min/max distance
            img_feats[5, c] = dist.min()
            img_feats[6, c] = dist.max()
            # Mean distance
            img_feats[7, c] = dist.mean()
            img_feats[8, c] = dist.std()
            img_feats[9:16, c] = cv2.HuMoments(M).flatten()
        feat_vect.append(img_feats)
        centroid_vect.append(centroid_contours)
        if i == 0:
            all_feats = img_feats
        else:
            all_feats = np.concatenate((all_feats, img_feats), axis=1)
    # Normalize features
    for i in range(0, len(feat_vect)):
        # NORMALIZATION ASSUMING GAUSSIAN DISTRIBUTION OF FEATS
        # mean_feats = np.tile(np.mean(all_feats, axis=1), (feat_vect[i].shape[1], 1)).T
        # std_feats = np.tile(np.std(all_feats, axis=1), (feat_vect[i].shape[1], 1)).T
        # feat_vect[i] = (feat_vect[i] - mean_feats)/std_feats
        # FEATURE SCALING
        min_feats = np.tile(np.min(all_feats, axis=1), (feat_vect[i].shape[1], 1)).T
        max_feats = np.tile(np.max(all_feats, axis=1), (feat_vect[i].shape[1], 1)).T
        feat_vect[i] = np.divide(np.subtract(feat_vect[i], min_feats), np.subtract(max_feats, min_feats))
    return feat_vect, contour_vect, centroid_vect
