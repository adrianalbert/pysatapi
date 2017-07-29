import sys, os

# for handling web requests
import httplib
import urllib3 as urllib
import cStringIO 
import requests

# for handling geometry and AOIs
from shapely.geometry import Polygon, Point
from shapely.wkt import dumps as wkt_dumps
import geojson

# for processing image data
from PIL import Image
import cv2

import numpy as np

from datetime import datetime, timedelta

BASE_URL = "https://api.planet.com/v0/"


class PlanetLabsAPI(object):
	""" Lightweight wrapper class for the Planet Labs API v0.
    """

	def __init__(self, key=None, dataType="scenes", provider="ortho"):
		self._key = key
		self._dataType = dataType
		self._provider = provider
		self._baseURL  = self.construct_url_request()

	
	def get_filtered_scenes_metadata(self,\
			loc=None, startDate=None, endDate=None, limit=10):
		""" Returns list of scenes that correspond to a location and to a time
			interval.
		"""
		if endDate is None:
			endDate = datetime.now().date().isoformat()
		if startDate is None:
			startDate = (datetime.now()-timedelta(days=365)).date().isoformat()
		if loc is None:
			loc = (-122.4194, 37.7749) # coordinates of San Francisco, CA

		# create a small polygon around the location of interest with an 
		# approximate spatial scale of 10m (that of a building)
		delta = 0.0001
		nw = (loc[0] - delta, loc[1] + delta)
		se = (loc[0] + delta, loc[1] - delta)
		sw = (loc[0] - delta, loc[1] - delta)
		ne = (loc[0] + delta, loc[1] + delta)
		poly = Polygon([nw, ne, se, sw, nw])
		locPolygon = wkt_dumps(poly)

		filters = {
			"acquired.gte":startDate,
			"acquired.lte":endDate,
			"intersects":locPolygon
		}
		return self.get_scenes_metadata(filters=filters, limit=limit)		


	def get_scenes_metadata(self, filters=None, limit=10):
		""" Retrieve a list of images matching a list of filters up to a limit.
			To get all images matching the filters, set limit = -1.
		"""
		nrScenes = 0
		scenesData = []
		if limit is not None:
			filters = {"limit":limit}
		nextURL = self.construct_url_request(filters)
		while ((nrScenes < limit) if limit != -1 else True) and nextURL:			
			r = requests.get(nextURL, auth=(self._key, ''))
			r.raise_for_status()
			data = r.json()
			scenesData += data["features"]
			nrScenes = len(scenesData)
			nextURL = data["links"].get("next", None) 
		scenesData = scenesData[:limit]

		print "There are %d relevant scenes. Retrieved data for the first %d."%\
			(data['count'], len(scenesData))
		return scenesData


	def construct_url_request(self, filters=None):
		url = BASE_URL + "%s/%s/?"%(self._dataType, self._provider)
		if filters is not None:
			url += urllib.urlencode(filters)
		return url


	def get_image_data_for_scene(self, scene, image="full"):
		url = extract_scene_attributes(scene, attribute=image)
		return self.get_image_data_from_url(url)
		

	def download_for_scene(self, scene, downloadPath="./"):
		url = extract_scene_attributes(scene, attribute=image)
		return self.download_image_data_from_url(url, downloadPath=downloadPath)
		

	def download_image_from_url(self, url, downloadPath="./"):
		""" Returns raw image from Planet Labs"""
		return download_image(url, self._key, downloadPath=downloadPath)

	
	def get_image_data_from_url(self, url):
		""" Returns raw image from Planet Labs"""
		return get_image_data(url, self._key)


def extract_scene_attributes(scene, attribute=None):
	timeAcquired = scene["properties"]['acquired']
	cloudCoverEst= scene["properties"]['cloud_cover']['estimated']

	# links to actual scene data
	full_link  = scene["properties"]["links"]['self']
	thumb_link = scene["properties"]["links"]['thumbnail']
	anal_link  = scene["properties"]['data']['products']['analytic']['full']

	attributes = {"timeAcquired":timeAcquired, "cloudCoverEst":cloudCoverEst,\
			"full":full_link, "thumb":thumb_link, \
			"analytic":anal_link}
	if attribute:
		return attributes.get(attribute, None)
	else:
		return attributes


def get_image_data(url, key):
	""" Retrieve image data from a given URL into a numpy array.
	"""
	r = requests.get(url, stream=True, auth=(key, ''))
	imgstr = "".join([chunk for chunk in r.iter_content(chunk_size=1024)])
	nparr = np.fromstring(imgstr, np.uint8)
	imgarr = cv2.imdecode(nparr, cv2.IMREAD_LOAD_GDAL)
	return imgarr


def download_image(url, key, downloadPath=""):
	""" Download image from URL to disc.
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


