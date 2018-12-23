import numpy as np
import cv2
from skimage.feature import hog
import pickle
import matplotlib.pyplot as plt
from settings import *

#This function returns HOG features and visualization (optionally)
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=True, feature_vector=False)
		return features, hog_image
	else:
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
			cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False,
			visualise=False, feature_vector=feature_vec)
		return features

# This function computes binned color features
def bin_spatial(img, size=(32, 32)):
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel()
	return features

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	return hist_features

# This function extracts features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
	
	features = []	
	for image in imgs:
		image_features = []
	# Apply color conversion if other than 'RGB'
		if color_space != 'RGB':
			if color_space == 'HSV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
			elif color_space == 'LUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
			elif color_space == 'HLS':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
			elif color_space == 'YUV':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
			elif color_space == 'YCrCb':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
			elif color_space == 'GRAY':
				feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
				feature_image = np.stack((feature_image, feature_image, feature_image), axis=2)
		else: feature_image = np.copy(image)

		#Compute spatial features if flag is set
		if spatial_feat == True:
			spatial_features = bin_spatial(feature_image, size=spatial_size)
			image_features.append(spatial_features)
		
		#Compute histogram features if flag is set
		if hist_feat == True:			
			hist_features = color_hist(feature_image, nbins=hist_bins)
			image_features.append(hist_features)
		
		#Compute HOG features if flag is set
		if hog_feat == True:		
			if hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel],
										orient, pix_per_cell, cell_per_block,
										vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
							pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			
			image_features.append(hog_features)
		features.append(np.concatenate(image_features))
	return features


# This function  extracts features from a single image window
def single_img_features(img, color_space='HSV', spatial_size=(32, 32),
						hist_bins=32, orient=9,
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):
	
	img_features = []
	# Apply color conversion if other than 'RGB'
	if color_space != 'RGB':
		if color_space == 'HSV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
		elif color_space == 'LUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
		elif color_space == 'HLS':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
		elif color_space == 'YUV':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
		elif color_space == 'YCrCb':
			feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		elif color_space == 'GRAY':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
				feature_image = np.stack((feature_image, feature_image, feature_image), axis=2)  # keep shape
	else: feature_image = np.copy(img)
	
	#Compute spatial features if flag is set
	if spatial_feat == True:
		spatial_features = bin_spatial(feature_image, size=spatial_size)
		img_features.append(spatial_features)
	
	#Compute histogram features if flag is set
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		#6) Append features to list
		img_features.append(hist_features)
	
	#Compute HOG features if flag is set
	if hog_feat == True:
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.extend(get_hog_features(feature_image[:,:,channel],
									orient, pix_per_cell, cell_per_block,
									vis=False, feature_vec=True))
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
						pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		
		img_features.append(hog_features)

	return np.concatenate(img_features)