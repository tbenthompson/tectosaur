import os
import io
import sys
import shutil
import gdal
import subprocess
import tempfile
import urllib
import scipy.interpolate
import pyproj
import numpy as np
from itertools import product
import boto3
import botocore

def mercator(lat, lon, zoom):
    ''' Convert latitude, longitude to z/x/y tile coordinate at given zoom.'''
    print(lat, lon)
    # convert to radians
    x1, y1 = lon * np.pi/180, lat * np.pi/180

    # project to mercator
    x2, y2 = x1, np.log(np.tan(0.25 * np.pi + 0.5 * y1))

    # transform to tile space
    tiles, diameter = 2 ** zoom, 2 * np.pi
    x3, y3 = int(tiles * (x2 + np.pi) / diameter), int(tiles * (np.pi - y2) / diameter)

    return zoom, x3, y3

# def get_dem_bounds(lonlat_pts):
#     minlat = np.min(lonlat_pts[:,1])
#     minlon = np.min(lonlat_pts[:,0])
#     maxlat = np.max(lonlat_pts[:,1])
#     maxlon = np.max(lonlat_pts[:,0])
#     latrange = maxlat - minlat
#     lonrange = maxlon - minlon
#     bounds = (
#         minlat - latrange * 0.1,
#         minlon - lonrange * 0.1,
#         maxlat + latrange * 0.1,
#         maxlon + lonrange * 0.1
#     )
#     return bounds

def get_dem_bounds(lonlat_pts):
    minlat = np.min(lonlat_pts[:,1])
    minlon = np.min(lonlat_pts[:,0])
    maxlat = np.max(lonlat_pts[:,1])
    maxlon = np.max(lonlat_pts[:,0])
    latrange = maxlat - minlat
    lonrange = maxlon - minlon
    bounds = [
        minlat - latrange * 0.1,
        minlon - lonrange * 0.1,
        maxlat + latrange * 0.1,
        maxlon + lonrange * 0.1,
    ]
    
    assert(-180 < bounds[1] < 180)
    assert(-180 < bounds[3] < 180)

    return bounds

def tiles(zoom, lat1, lon1, lat2, lon2):
    ''' Convert geographic bounds into a list of tile coordinates at given zoom.'''

    # convert to geographic bounding box
    minlat, minlon = min(lat1, lat2), min(lon1, lon2)
    maxlat, maxlon = max(lat1, lat2), max(lon1, lon2)

    # convert to tile-space bounding box
    _, xmin, ymin = mercator(maxlat, minlon, zoom)
    _, xmax, ymax = mercator(minlat, maxlon, zoom)

    # generate a list of tiles
    xs, ys = range(xmin, xmax+1), range(ymin, ymax+1)
    tiles = [(zoom, x, y) for (y, x) in product(ys, xs)]

    return tiles

def download_file(x, y, z, save_path):
    BUCKET_NAME = 'elevation-tiles-prod'
    KEY = 'geotiff/{z}/{x}/{y}.tif'.format(x = x, y = y, z = z)

    s3 = boto3.resource('s3')

    try:
        bucket = s3.Bucket(BUCKET_NAME)
        bucket.download_file(KEY, save_path)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

def download(output_path, tiles, verbose=True):
    ''' Download list of tiles to a temporary directory and return its name.
    '''
    dir = tempfile.mkdtemp(prefix='collected-')
    try:
        files = []

        for (z, x, y) in tiles:
            save_path = os.path.join(dir, '{}-{}-{}.tif'.format(z, x, y))
            print('Downloading', save_path)
            download_file(x, y, z, save_path)
            files.append(save_path)

        print('Combining', len(files), 'into', output_path, '...', file=sys.stderr)
        temp_tif = os.path.join(dir, 'temp.tif')
        subprocess.check_call(['gdal_merge.py', '-o', temp_tif] + files)
        shutil.move(temp_tif, output_path)
    finally:
        if os.path.exists(dir):
            shutil.rmtree(dir)


def get_pt_elevations(lonlat_pts, zoom, n_interp = 100):
    bounds = get_dem_bounds(lonlat_pts)
    LON, LAT, DEM = get_dem(zoom, bounds, n_interp)
    return scipy.interpolate.griddata(
        (LON, LAT), DEM, (lonlat_pts[:,0], lonlat_pts[:,1])
    )

def get_dem(zoom, bounds, n_width, dest_dir = 'dem_download'):
    print('downloading dem data for bounds = ' + str(bounds) + ' and zoom = ' + str(zoom))
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    dest = os.path.join(dest_dir, 'raw_merc.tif')
    download(dest, tiles(zoom, *bounds), verbose = False)
    filebase, fileext = os.path.splitext(dest)
    dataset_merc = gdal.Open(dest)
    filename_latlon = os.path.join(dest_dir, 'latlon.tif')
    dataset_latlon = gdal.Warp(filename_latlon, dataset_merc, dstSRS = 'EPSG:4326')
    dem = dataset_latlon.ReadAsArray().astype(np.float64)
    width = dataset_latlon.RasterXSize
    height = dataset_latlon.RasterYSize
    gt = dataset_latlon.GetGeoTransform()
    xs = np.linspace(0, width - 1, width)
    ys = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(xs, ys)
    lon = gt[0] + X * gt[1] + Y * gt[2]
    lat = gt[3] + X * gt[4] + Y * gt[5]
    assert(gt[2] == 0)
    assert(gt[4] == 0)
    minlat, minlon = bounds[0], bounds[1]
    maxlat, maxlon = bounds[2], bounds[3]
    expand = 0
    LON, LAT = np.meshgrid(
        np.linspace(minlon - expand, maxlon + expand, n_width),
        np.linspace(minlat - expand, maxlat + expand, n_width)
    )
    DEM = scipy.interpolate.griddata(
        (lon.flatten(), lat.flatten()), dem.flatten(),
        (LON, LAT)
    )
    return LON.flatten(), LAT.flatten(), DEM.flatten()

def project(inx, iny, dem, proj_name, inverse = False):
    wgs84 = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    if proj_name == 'ellps':
        proj = pyproj.Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')
    elif proj_name.startswith('utm'):
        zone = proj_name[3:]
        print(zone)
        proj = pyproj.Proj("+proj=utm +zone=" + zone + ", +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    if inverse:
        x,y,z = pyproj.transform(proj, wgs84, inx, iny, dem)
    else:
        x,y,z = pyproj.transform(wgs84, proj, inx, iny, dem)
    projected_pts = np.vstack((x,y,z)).T.copy()
    return projected_pts
