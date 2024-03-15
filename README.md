# IISc Take-Home Robotics Challenge

## Part (a): Converting Dataset to .bag Files

### Datasets Used
The KITTI Raw Data, available at [KITTI Raw Data](https://www.cvlibs.net/datasets/kitti/raw_data.php), was used to generate .bag files. These .bag files were created using the kitti2bag converter, which transforms the raw data into ROS1 bag files. The resulting rosbag files can be played back to access the data as ROS topics. <br>
The rosbag can be accessed here: [rosbag](https://drive.google.com/file/d/1HQxCmjTR5fmLiyYgU6XPJbWF5x1veHSb/view?usp=sharing)

## Part (b): Implementing PointNet on ROS

### Datasets Used
The following datasets were utilized:
- Left Colour Images: [Data Object Image 2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)
- Velodyne Point Clouds: [Data Object Velodyne](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)
- Camera Calibration Matrices: [Data Object Calib](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)
- Training Labels: [Data Object Label 2](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)

These datasets were taken based on the thesis presented in the referenced repository. For simplicity in running the code, a sample dataset containing 100 images is provided, streamlining the process to executing the code only.

### PointNet
PointNet was implemented using the repository available at [3DOD Thesis](https://github.com/fregu856/3DOD_thesis/tree/master?tab=readme-ov-file#run-pretrained-extended-frustum-pointnet-model-on-kitti-val). Please ensure that all necessary dependencies are installed as per the instructions provided there.

#### Frustum PointNet
The Frustum PointNet architecture accepts two inputs: Point Cloud Data and a 2D Bounding Box. The 2D bounding box is geometrically extruded to extract the corresponding frustum point cloud, encompassing all points in the LiDAR point cloud lying inside the 2D box when projected onto the image plane.

##### Running the Code
To execute the code for Frustum PointNet, navigate to the Frustum-PointNet folder within the pointnet_implementation directory:
```bash
cd pointnet_implementation/Frustum-PointNet/
```
Then, run the following command to perform a forward pass on the training data:
```bash
python3 custom_frustum_pointnet.py
```
The output will resemble:
```bash
[[-0.2 -0.1  0.   0.1  0.   0.1 -8.   7.6 -6.4 -6.9  0.2 -0.2 -0.2 -0.2]]
```
This output represents the 3D bounding box of the detected car.

#### Extended-Frustum PointNet
As an extension to Frustum PointNet, Image Data was fused with Point Cloud Data. The resulting architecture is similar to Frustum PointNet with a modified Bbox-Net.

##### Running the Code
To execute the code for Extended-Frustum PointNet, navigate to the Extended-Frustum-PointNet folder within the pointnet_implementation directory:
```bash
cd pointnet_implementation/Extended-Frustum-PointNet/
```
Then, run the following command to perform a forward pass on the training data:
```bash
python3 custom_extended_frustum_pointnet.py
```
Upon closing the input car image, the output will resemble:
```bash
[[ -0.1  -0.1   0.1   0.2   0.1   0.1 -11.    8.1  -9.  -11.9   0.3  -0.1 -0.1  -0. ]]
```
This represents the 3D bounding box of the detected car in the image.

## Incomplete Aspects
### PointNet Integration in ROS
The integration of the running PointNet code into ROS remains outstanding. Below is the proposed framework for integrating PointNet into ROS:

- Play the rosbag.
- Read the PointCloud and Image data from the generated topics in the rosbag.
- Convert the PointCloud and Image data into a format suitable for input to the data loader.
- Feed the output of the data loader to the network
- Use the frustum_point_clouds and bbox_2d_imgs as inputs to the network (depends on usage of frustum or extended-frustum network)
- The generated output would be the 3D bounding box of the detected cars in the rosbag.
