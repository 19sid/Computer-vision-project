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
from train import *
from windows import *

if __name__ == '__main__':
	svc = LinearSVC()	
	X_scaler = StandardScaler()
	
	save_dict = pickle.load(open('out.p', 'rb'))
	svc = save_dict['svc']
	X_scaler = save_dict['X_scaler']

	imdir='test_images'	 
	image_file = 'image1.jpeg'
	image = mpimg.imread(os.path.join('test_images', image_file))
	draw_image = np.copy(image)
	
	windows = slide_window(image, x_start_stop=(0, image.shape[1]), y_start_stop=(0,image.shape[0]),
					xy_window=(40,40), xy_overlap=(0.8, 0.8))
	
	hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
							spatial_size=spatial_size, hist_bins=hist_bins,
							orient=orient, pix_per_cell=pix_per_cell,
							cell_per_block=cell_per_block,
							hog_channel=hog_channel, spatial_feat=spatial_feat,
							hist_feat=hist_feat, hog_feat=hog_feat)

	wow=non_max_suppression_fast(hot_windows,0.0)
	
	window_img = draw_boxes(draw_image, wow, color=(0, 0, 255), thick=2)
	print("Number of cars found "+str(len(wow)))
	plt.imshow(window_img)
	plt.show()