import os

import requests
from requests.auth import HTTPBasicAuth

from multiprocessing.dummy import Pool as ThreadPool

from osgeo import gdal

# for processing image data
import cv2
import numpy as np

BASE_URL_TYPES = "https://api.planet.com/data/v1/item-types/" 

# NOTE: IMAGE FORMAT IS BGRN!

def get_image_subarea(img_url, aoi, downloadPath=None):
	'''
	img_url is obtained by activating the asset of interest using the activate_asset function below.
	aoi is the area of interest. Can be one of:
		- the path to a geojson file
		- a dict with a polygon in the geojson format 
	'''
	vsicurl_url = '/vsicurl/' + img_url
	if downloadPath is not None:
		output_file = download_path + '_subarea.tif'
	else:
		output_file = "tmp.tif"

	# GDAL Warp crops the image by our AOI, and saves it
	dat = gdal.Warp(output_file, vsicurl_url, 
		dstSRS='EPSG:4326', 
		cutlineDSName=aoi, 
		cropToCutline=True)
	if downloadPath is None:
		img = np.rollaxis(dat.ReadAsArray(), 0,3)
		img = img.astype(float) / (img.max(2)[:,:,None]+1)
		return img
    

def activate_assets(key, item_ids, item_type, asset_type, n_jobs=1):

	# An easy way to parallise I/O bound operations in Python
	# is to use a ThreadPool.
	thread_pool = ThreadPool(n_jobs)

	fn_activate = lambda myid: activate_asset(key,myid,item_type,asset_type)

	# In this example, all items will be sent to the `activate_item` function
	# but only 5 will be running at once
	ret = thread_pool.map(fn_activate, item_ids)
	return ret


def activate_asset(key, item_id, item_type, asset_type):
	# setup auth
	session = requests.Session()
	session.auth = (key, '')

	# request an item
	item = session.get(
	    (BASE_URL_TYPES + "{}/items/{}/assets/").format(item_type, item_id))

	if item.status_code == 429:
	    raise Exception("rate limit error")

	# extract the activation url from the item for the desired asset
	item_json = item.json()
	if asset_type not in item_json:
		return None
	item_activation_url = item_json[asset_type]["_links"]["activate"]

	# request activation
	response = session.post(item_activation_url)

	if response.status_code == 429:
	    raise Exception("rate limit error")

	if response.status_code == 204:
		download_url = item.json()[asset_type]['location']
		return download_url
	else:
		return response.status_code


def get_image_data(url, key):
	""" 
	Retrieve image data from a given URL into a numpy array.
	"""
	r = requests.get(url, stream=True, auth=(key, ''))
	imgstr = "".join([chunk for chunk in r.iter_content(chunk_size=1024)])
	nparr = np.fromstring(imgstr, np.uint8)
	imgarr = cv2.imdecode(nparr, cv2.IMREAD_LOAD_GDAL)
	return imgarr


def download_image(url, key, downloadPath=""):
	""" 
	Download image from URL to disc. Use chunking since images can be large.
	"""
	r = requests.get(url, stream=True, auth=(key, ''))
	if 'content-disposition' in r.headers:
	    localFilename = r.headers['content-disposition'] \
	        .split("filename=")[-1].strip("'\"")
	else:
	    localFilename = '.'.join(url.split('/')[-2:])
	    for c in ["?", ".", "=", "product"]:
	    	localFilename = localFilename.replace(c,"_")
	localFilename = downloadPath + "/" + localFilename  
	with open(localFilename, 'wb') as f:
	    for chunk in r.iter_content(chunk_size=1024):
	        if chunk: # filter out keep-alive new chunks
	            f.write(chunk)
	            f.flush()
	return localFilename

