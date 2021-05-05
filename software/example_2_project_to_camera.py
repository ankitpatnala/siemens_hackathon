
"""
Project 3d points, like we have it for pointcloud, into the images.
"""

from highway.pointcloud import read_ply
from highway.camera import Projector, show_image


def main():
    ply_file = "highway_data/splitted/reduced_22_intensity.ply"
    planar_folder = "highway_data/planar2"

    points, colors = read_ply(ply_file)
    proj = Projector(planar_folder)

    # camera pose index - number between 0 and len(proj.poses) - 1
    index = 750

    img = proj.render(index, points, colors * 255)
    show_image(img)


if __name__ == '__main__':
    main()
