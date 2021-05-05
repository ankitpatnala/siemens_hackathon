
"""
Create 2d projections of the pointclouds and render them.
"""

import numpy as np

from highway.camera import load_poses, show_image
from highway.pointcloud import read_ply, render_points


def main():
    ply_file = "highway_data/splitted/reduced_22_intensity.ply"
    pose_file = "highway_data/planar1/reference.json"

    poses = load_poses(pose_file)
    index = 750
    xyz = np.array([(pose['x'], pose['y'], pose['z']) for pose in poses])

    points, colors = read_ply(ply_file)

    # define vector orthogonal to the driving direction
    origin = xyz[index]
    vec_forward = xyz[index + 1] - xyz[index - 1]
    vec_forward /= np.linalg.norm(vec_forward)
    vec_up = np.array([0, 0, 1])
    vec_right = np.cross(vec_forward, vec_up)

    # get points within 2 meter of driving vector
    d = np.abs((points - origin) @ vec_right)
    sel = d < 2
    points_close = points[sel]
    intensity_close = colors[sel, 0]

    # project points onto driving direction
    x = (points_close - origin) @ vec_forward
    y = (points_close - origin) @ vec_up

    # create image coordinates
    img_width = 2000
    img_height = 1200
    scale = 100
    y_offset = 3  # meter

    x_img = x * scale
    y_img = img_height - (y + y_offset) * scale
    points_2d = np.c_[x_img, y_img]

    # render points
    img = render_points(points_2d, intensity_close * 255, (img_width, img_height))
    show_image(img)


if __name__ == "__main__":
    main()
