import numpy as np
import cv2
import pickle
import time
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from skimage.feature import hog
from scipy.ndimage.measurements import label
from settings import *
from features import *
from train import *

# This function generates windows on th image according to the parameters specified
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
					xy_window=(64, 64), xy_overlap=(0.0, 0.0)):
	
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	
	nx_windows = np.int(xspan/nx_pix_per_step) - 1
	ny_windows = np.int(yspan/ny_pix_per_step) - 1

	window_list = []

	for ys in range(ny_windows):
		for xs in range(nx_windows):
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			window_list.append(((startx, starty), (endx, endy)))

	return window_list

#This function finds if there us a car in the window or not
def search_windows(img, windows, clf, scaler, color_space='HSV',
					spatial_size=(32, 32), hist_bins=32,
					hist_range=(0, 256), orient=9,
					pix_per_cell=8, cell_per_block=2,
					hog_channel=0, spatial_feat=True,
					hist_feat=True, hog_feat=True):

	
	on_windows = []
	
	for window in windows:
		# Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
		#Extract features for that window using single_img_features()
		features = single_img_features(test_img, color_space='HSV',
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)
		
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		prediction = clf.predict(test_features)
				
		if prediction == 1:
			on_windows.append(window)
	return on_windows


# This functions forms bouunding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=5):
	imcopy = np.copy(img)
	
	for bbox in bboxes:
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	
	return imcopy

# This function finds the finds the best window by removing the overlapping windows
def non_max_suppression_fast(boxes, overlapThresh):
	
	if len(boxes) == 0:
		return [] 
	
	boxes = np.array(boxes, dtype=np.float32)
	pick = []
 
	x1 = boxes[:,0,0]
	y1 = boxes[:,0,1]
	x2 = boxes[:,1,0]
	y2 = boxes[:,1,1]
 		
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
 		
	while len(idxs) > 0:
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates and the smallest (x, y) coordinates  
		# for start and end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],np.where(overlap > overlapThresh)[0])))
 	
	boxes = np.array(boxes[pick], dtype=np.int32)	
	x1=boxes[:,0,0]
	y1=boxes[:,0,1]
	x2=boxes[:,1,0]
	y2=boxes[:,1,1]
	bb=[]
	for i in range (int(boxes.size/4)):
		bb.append(((x1[i],y1[i]),(x2[i],y2[i])))	

	return bb