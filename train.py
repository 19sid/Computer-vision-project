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
from skimage.feature import hog
from scipy.ndimage.measurements import label
from settings import *
from features import *

def train(cars, notcars, svc, X_scaler):
	car_features = extract_features(cars, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)
	notcar_features = extract_features(notcars, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)

	X = np.vstack((car_features,notcar_features)).astype(np.float64)	
	X_scaler.fit(X)	
	scaled_X = X_scaler.transform(X)	
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
	
	rand_state = np.random.randint(0, 100)
	X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')	
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))


if __name__ == '__main__':
	svm = LinearSVC()
	X_scaler = StandardScaler()

	print('Reading training data and training classifier')
	with open('data.p', 'rb') as f:
		data = pickle.load(f)
	cars = data['vehicles']
	notcars = data['non_vehicles']
	train(cars, notcars, svm, X_scaler)
	print('Training complete, saving trained model to out.p')
	with open('out30false.p', 'wb') as f:
		pickle.dump({'svc': svm, 'X_scaler': X_scaler}, f)