"""
Convert between point cloud coordinate system and GPS coordinates.

The point clouds use a cartesian coordinate system referred as XYZ coordinates.
Each point XYZ can be converted to GPS coordinates given WKT string that defines that transformation.

You can use these functions here to convert between them.
"""

from typing import Tuple

import pyproj

# Tranformation definiton
WKT = """PROJCS["32_North",GEOGCS["World_wide/UTM",DATUM["WGS_1984",SPHEROID["World_Geodetic_System_1984",
6378137,298.25722356301],TOWGS84[0,0,0,0,0,0,0]],PRIMEM["Greenwich",0.0], UNIT["Degree",0.0174532925199433]],
PROJECTION["Transverse_Mercator"],PARAMETER["central_meridian",9],PARAMETER["scale_factor",0.9996],
PARAMETER["latitude_of_origin",0],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],
UNIT["Meter",1]]"""
H_OFFSET = 48.1


def to_gps(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """ Transforms XYZ coordinates to GPS coordinates (longitude, latitude, height)"""
    crs_ntm = pyproj.CRS.from_wkt(WKT)
    crs_4326 = pyproj.CRS.from_epsg(4326)  # EPSG 4326 / WGS84 used by GPS
    transformer = pyproj.Transformer.from_crs(crs_ntm, crs_4326)
    return transformer.transform(x, y, z + H_OFFSET)


def from_gps(longitude: float, latitude: float, height: float) -> Tuple[float, float, float]:
    """ Transforms GPS coordinates (longitude, latitude, height) to XYZ coordinates """
    crs_ntm = pyproj.CRS.from_wkt(WKT)
    crs_4326 = pyproj.CRS.from_epsg(4326)  # EPSG 4326 / WGS84 used by GPS
    transformer = pyproj.Transformer.from_crs(crs_4326, crs_ntm)
    return transformer.transform(longitude, latitude, height - H_OFFSET)


if __name__ == "__main__":
    # test the functions
    print("XYZ coordinates")
    xyz = 471884.362, 5533597.451, 114.849
    print(xyz)
    print("Convert to GPS")
    gps = to_gps(*xyz)
    print(gps)
    print("Convert back to XYZ")
    xyz2 = from_gps(*gps)
    print(xyz2)
