import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from datasets_img import DatasetFrustumPointNetImgAugmentation, EvalDatasetFrustumPointNetImg, getBinCenters, wrapToPi # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)


from frustum_pointnet_img import FrustumPointNetImg

import numpy as np
import pickle
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# Define the network
network = FrustumPointNetImg("Extended-Frustum-PointNet_eval_val_seq", project_dir="./3DOD_thesis")

# Load the pretrained model
network.load_state_dict(torch.load("../pretrained_models/model_38_2_epoch_400.pth", map_location=torch.device('cpu')))

NH = network.BboxNet_network.NH

# Define the dataset
train_dataset = DatasetFrustumPointNetImgAugmentation(kitti_data_path="../data/kitti", 
                                                      kitti_meta_path="../data/kitti/meta",
                                                      type="train", NH=NH)

# Define the dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=1, shuffle=False,
                                           num_workers=0)

# Set the network to evaluation mode to remove dropout etc.
network.eval()

# Extract the first batch of train_loader
frustum_point_clouds, bbox_2d_imgs, labels_InstanceSeg, labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped = next(iter(train_loader))

frustum_point_clouds = Variable(frustum_point_clouds) # (shape: (batch_size, num_points, 4))
bbox_2d_imgs = Variable(bbox_2d_imgs) # (shape: (batch_size, 3, H, W))
frustum_point_clouds = frustum_point_clouds.transpose(2, 1)
outputs = network(frustum_point_clouds, bbox_2d_imgs)
outputs_InstanceSeg = outputs[0] # (shape: (batch_size, num_points, 2))
outputs_TNet = outputs[1] # (shape: (batch_size, 3))
outputs_BboxNet = outputs[2] # (shape: (batch_size, 3 + 3 + 2*NH))
seg_point_clouds_mean = outputs[3] # (shape: (batch_size, 3))
dont_care_mask = outputs[4] # (shape: (batch_size, ))
outputs_BboxNet = outputs_BboxNet.detach().numpy()

# Display estimated bounding box
print(np.around(outputs_BboxNet, 1))

# Do only one forward pass
# for step, (frustum_point_clouds, bbox_2d_imgs, labels_InstanceSeg, labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped) in enumerate(train_loader):
#     frustum_point_clouds = Variable(frustum_point_clouds) # (shape: (batch_size, num_points, 4))
#     bbox_2d_imgs = Variable(bbox_2d_imgs) # (shape: (batch_size, 3, H, W))
#     frustum_point_clouds = frustum_point_clouds.transpose(2, 1)
#     outputs = network(frustum_point_clouds, bbox_2d_imgs)
#     outputs_InstanceSeg = outputs[0] # (shape: (batch_size, num_points, 2))
#     outputs_TNet = outputs[1] # (shape: (batch_size, 3))
#     outputs_BboxNet = outputs[2] # (shape: (batch_size, 3 + 3 + 2*NH))
#     seg_point_clouds_mean = outputs[3] # (shape: (batch_size, 3))
#     dont_care_mask = outputs[4] # (shape: (batch_size, ))
#     break

# print(outputs_BboxNet)