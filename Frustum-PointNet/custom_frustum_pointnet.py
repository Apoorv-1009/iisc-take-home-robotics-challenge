from datasets import DatasetFrustumPointNetAugmentation, EvalDatasetFrustumPointNet, getBinCenters, wrapToPi # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from frustum_pointnet import FrustumPointNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Define the network
network = FrustumPointNet("Frustum-PointNet_1", project_dir="./3DOD_thesis")

# Load the pretrained model
network.load_state_dict(torch.load("../pretrained_models/model_37_2_epoch_400.pth", map_location=torch.device('cpu')))

NH = network.BboxNet_network.NH

# Define the dataset
train_dataset = DatasetFrustumPointNetAugmentation(kitti_data_path="../data/kitti",
                                                   kitti_meta_path="../data/kitti/meta",
                                                   type="train", NH=NH)

# Define the dataloader
batch_size = 1
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, shuffle=False,
                                           num_workers=0)

# Set the network to evaluation mode to remove dropout etc.
network.eval()

# Get the first batch of train_loader
frustum_point_clouds, labels_InstanceSeg,labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped = next(iter(train_loader))

frustum_point_clouds = Variable(frustum_point_clouds) # (shape: (batch_size, num_points, 4))

frustum_point_clouds = frustum_point_clouds.transpose(2, 1) # (shape: (batch_size, 4, num_points))

outputs = network(frustum_point_clouds)
outputs_InstanceSeg = outputs[0] # (shape: (batch_size, num_points, 2))
outputs_TNet = outputs[1] # (shape: (batch_size, 3))
outputs_BboxNet = outputs[2] # (shape: (batch_size, 3 + 3 + 2*NH))
seg_point_clouds_mean = outputs[3] # (shape: (batch_size, 3))
dont_care_mask = outputs[4] # (shape: (batch_size, ))

# Display estimated bounding box
outputs_BboxNet = outputs_BboxNet.detach().numpy()
print(np.around(outputs_BboxNet, 1))

# Do only one forward pass
# for step, (frustum_point_clouds, labels_InstanceSeg,
#            labels_TNet, labels_BboxNet, labels_corner, labels_corner_flipped) in enumerate(train_loader):
#     frustum_point_clouds = Variable(frustum_point_clouds) # (shape: (batch_size, num_points, 4))

#     frustum_point_clouds = frustum_point_clouds.transpose(2, 1) # (shape: (batch_size, 4, num_points))

#     print(frustum_point_clouds)

#     outputs = network(frustum_point_clouds)
#     outputs_InstanceSeg = outputs[0] # (shape: (batch_size, num_points, 2))
#     outputs_TNet = outputs[1] # (shape: (batch_size, 3))
#     outputs_BboxNet = outputs[2] # (shape: (batch_size, 3 + 3 + 2*NH))
#     seg_point_clouds_mean = outputs[3] # (shape: (batch_size, 3))
#     dont_care_mask = outputs[4] # (shape: (batch_size, ))
#     # print(step)
#     # outputs_BboxNet = outputs_BboxNet.detach().numpy()
#     # print(np.round(outputs_BboxNet))
#     break

# Convert the outputs to numpy arrays
# outputs_BboxNet = outputs_BboxNet.detach().numpy()
# print(np.around(outputs_BboxNet, 1))
# print(outputs_BboxNet)
# for i in range(len(outputs)):
#     print(outputs[i].shape)
#     print(outputs[i])





