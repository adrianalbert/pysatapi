import os

import requests
from requests.auth import HTTPBasicAuth

from osgeo import gdal

# for processing image data
import cv2
import numpy as np

BASE_URL_TYPES = "https://api.planet.com/data/v1/item-types/" 


def get_image_subarea(item_id, item_type, asset_type):

	item_url = '{}/{}/items/{}/assets'.format(BASE_URL_TYPES, item_type, item_id)

	# Request a new download URL
	result = requests.get(item_url, auth=HTTPBasicAuth(key, ''))
	download_url = result.json()[asset_type]['location']

	vsicurl_url = '/vsicurl/' + download_url
	output_file = item_id + '_subarea.tif'

	# GDAL Warp crops the image by our AOI, and saves it
	gdal.Warp(output_file, vsicurl_url, dstSRS = 'EPSG:4326', cutlineDSName = 'subarea.geojson', cropToCutline = True)
    

def activate_asset(key, item_id, item_type, asset_type):
	# setup auth
	session = requests.Session()
	session.auth = (key, '')

	# request an item
	item = session.get(
	    (BASE_URL_TYPES + "{}/items/{}/assets/").format(item_type, item_id))

	# extract the activation url from the item for the desired asset
	item_activation_url = item.json()[asset_type]["_links"]["activate"]

	# request activation
	response = session.post(item_activation_url)

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

