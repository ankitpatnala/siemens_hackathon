
"""
Project 3d points, like we have it for pointcloud, into the images.
"""

from highway.pointcloud import read_ply,to_pcd,write_ply
from highway.camera import Projector, show_image
import open3d as o3d
import glob
import copy
import json

def main():
 

    ply_file = r"highway_data/splitted/reduced_22_intensity.ply"
    planar_folders = ["highway_data/planar1",
                      "highway_data/planar2",
                      "highway_data/planar3"]
    points, colors = read_ply(ply_file)
    index = 713
    collected_points_list = []
    collected_colors_list = []
    for planar_folder in planar_folders:
        proj = Projector(planar_folder)
        img,points_needed,colors_needed = proj.render(index, points, colors * 255)
        if img is not None:
            show_image(img)
            collected_points_list.extend(points_needed)
            collected_colors_list.extend(colors_needed)


    pcd = to_pcd(collected_points_list,collected_colors_list)
    o3d.visualization.draw_geometries([pcd])
    
    write_ply("22_intensity_segmented_pose750.ply",collected_points_list,collected_colors_list)


if __name__ == '__main__':
    main()
