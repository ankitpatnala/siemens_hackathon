# Siemens Hackathon

# P2X Points to Everything #


## We read roads!!!##

This is joint project of our teammebers.
The product provides the segmented point cloud based on the data collected from LIDAR data and from three cameras attached to left, center and right side of the image.

Segmented point clouds helps the engineers for faster visualization of the 3d space.

The product has the potential to extract valuable information from segmented point clouds like instances of objects like how many cars nearby, objects like side rails around you. Such information will aid eHighway project for better planning.

Future prospective of the work is to use state of the art point cloud architectures like PointNet++ ([https://github.com/charlesq34/pointnet2](url)) to calculate import information like minimum distance of car from side-rails, gap between  bridge and car, distance from lanes. Such information are essential for automatic driving applications. Extended target is to conncet point cloud information of each car to the cloud in order to prevent fatal accident on the road.

## Workflow

1.  Read point cloud data from .ply files collected from LIDAR
2.  Generate segmentation map of each images using DeepLabv3 model 
## Segmentations using pre-trained DeepLabV3 ##
![plot](./software/train_help/segm.png)
3.  One can also use labelme software([https://github.com/wkentaro/labelme](url) to generate segmentation of few examples and then fine-tune using the pre-trained model.
