import pickle

with open('/home/sgitai/FRCT/real_data/train/bimanual_lift_box/all_variations/episodes/episode0/low_dim_obs.pkl', 'rb') as file:


    data = pickle.load(file)


print(data)

# import cv2
# import numpy as np

# img = cv2.imread('/home/sgitai/FRCT/real_data/train/bimanual_lift_box/all_variations/episodes/episode0/front_depth/depth_0.png')

# print(np.max(img))