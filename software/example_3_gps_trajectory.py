
"""
Convert pointcloud coordinates into GPS and visualize them with OpenStreetMaps
"""

import numpy as np
import folium

from highway.camera import load_poses
from highway.transform import to_gps


def main():
    pose_file = "highway_data/planar1/reference.json"

    poses = load_poses(pose_file)
    xyz = np.array([(pose['x'], pose['y'], pose['z']) for pose in poses])

    print()
    print("XYZ coordinates of the camera trajectory")
    print(xyz)

    long_lat_height = np.array(to_gps(xyz[:, 0], xyz[:, 1], xyz[:, 2])).T

    print()
    print("GPS coordinates of the camera trajectory")
    print(long_lat_height)

    # export to maps
    m = folium.Map(location=long_lat_height[0, :2], zoom_start=13)
    folium.PolyLine(
        locations=long_lat_height[:, :2],
        popup="Planar 1 Reference",
        color="#000000"
    ).add_to(m)
    m.save("example_3_index.html")

    # open index.html in your browser


if __name__ == '__main__':
    main()
