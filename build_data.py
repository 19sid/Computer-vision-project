import numpy as np
import pickle
import cv2
import glob
from scipy.misc import imread


non_vehicles = []
vehicles = []
nonvehicle_paths = glob.glob('non-vehicles/n_vehi/*.jpg')
vehicle_paths = glob.glob('vehicles/Vehi/*.jpg')
nonvehicle_paths1=glob.glob('non-vehicles/non-vehicles/*/*.png')
vehicle_paths1=glob.glob('vehicles/vehicles/*/*.png')


for path in nonvehicle_paths:
	im=imread(path)	
	non_vehicles.append(im)

for path1 in nonvehicle_paths1:
	im1=imread(path1)	
	non_vehicles.append(im1)

for path2 in vehicle_paths: 
	im2=imread(path2)	
	vehicles.append(im2)

for path3 in vehicle_paths1: 
	im3=imread(path3)	
	vehicles.append(im3)

print(len(vehicles))
print(len(non_vehicles))	

save_dict = {'non_vehicles': np.array(non_vehicles), 'vehicles': np.array(vehicles)}
with open('data.p', 'wb') as f:
	pickle.dump(save_dict, f)
