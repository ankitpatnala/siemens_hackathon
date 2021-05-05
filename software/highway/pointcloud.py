
"""
Read, write, display pointclouds. Render 2d points.
"""

import numpy as np
import open3d as o3d
import numba


def read_ply(filename):
    """ read points and intensity from *.ply file """
    pcd = o3d.io.read_point_cloud(str(filename))

    # get points as numpy array
    points = np.asarray(pcd.points)
    # colors are gray, so we select the first channel
    colors = np.asarray(pcd.colors)

    return points, colors


def to_pcd(points, colors):
    """ construct Open3D points cloud from numpy arrays for displaying """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def write_ply(filename, points, colors):
    """ save pointcloud as ply for viewing in Meshlab """
    pcd = to_pcd(points, colors)
    o3d.io.write_point_cloud(str(filename), pcd)


def render_points(points, intensity, size, radius=1, bg_color=(0, 0, 0)):
    """
    Render 2d points given in image coordinates on a image with given size.

    We use here the typical image coordinate system, where x points to the right and y down.

    :param points: 2d points as n x 2 numpy array. The points should be between 0 and image size.
    :param intensity: numpy array with intensity value for each point between 0 and 255.
    :param size: size of the image (width, height) that is generated
    :param radius: radius of each point drawn in pixel. If multiple points are drawn over each other
                   the function calculates the average intensity for each pixel.
    :param bg_color: background color as RGB.
    :return: rendered grayscale image as numpy array
    """
    return _render_points(points, intensity, size, radius, bg_color)


@numba.jit(nopython=True, cache=True)
def _render_points(points, intensity, size, radius, bg_color):
    """
    This internal function is compiles to C++ so that it runs very fast.
    """
    w, h = size

    img = np.zeros((h, w), np.uint16)
    cnt = np.zeros((h, w), np.uint8)

    for i in range(len(points)):
        for rx in range(-radius, radius + 1):
            for ry in range(-radius, radius + 1):
                px = int(points[i][0] + rx + 0.5)
                py = int(points[i][1] + ry + 0.5)
                if 0 <= px < w and 0 <= py < h:
                    img[py, px] += intensity[i]
                    cnt[py, px] += 1

    out = np.empty((h, w, 3), np.uint8)
    for y in range(h):
        for x in range(w):
            if cnt[y, x] != 0:
                out[y, x, :] = img[y, x] / cnt[y, x]
            else:
                out[y, x] = bg_color

    return out
