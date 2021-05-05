
"""
Created center *.ply files.

This allows better viewing in Meshlab, which only supports single precision floats.
"""

import pathlib
import json

import numpy as np

from highway.pointcloud import read_ply, write_ply


def main():
    ply_folder = "highway_data/splitted/"
    center = None

    for input_file in sorted(pathlib.Path(ply_folder).glob("*_intensity.ply")):
        print(input_file)
        output_file = input_file.parent / (input_file.name[:-4] + "_centered.ply")

        points, colors = read_ply(input_file)

        if center is None:
            center = np.mean(points, axis=0)
            # store center, in case you need it later
            with open(str(input_file.parent / 'center.txt'), 'w') as f:
                json.dump(list(center), f)

        points_centered = points - center
        write_ply(output_file, points_centered, colors)


if __name__ == '__main__':
    main()
