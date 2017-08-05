import sys, os

# for handling web requests
import httplib
import urllib3 as urllib
import cStringIO 
import requests
from requests.auth import HTTPBasicAuth

# for handling geometry and AOIs
from shapely.geometry import Polygon, Point
import shapely
import geojson

import numpy as np

from datetime import datetime, timedelta

# own packages
import filters, download
sys.path.append("../satimage-processing/")
import satimg

class PLClient(object):
	""" 
	Lightweight wrapper class for the Planet Labs API v1.
    """
	def __init__(self, key=None):
		if key is None:
			if os.environ.get('PLANET_API_KEY') is not None:
				self._key = os.environ.get('PLANET_API_KEY') 
			elif os.environ.get('PL_API_KEY') is not None:
				self._key = os.environ.get('PL_API_KEY') 
		else:
			self._key = key


def loc_to_AOI(lonlat, w=None):
	'''
	Define square AOI around (lon, lat) of interest with size w x w (in km).
	'''
	wLat, wLon = satimg.km_to_deg_at_location(lonlat[::-1], (w,w))
	nw = (lonlat[0] - wLon, lonlat[1] + wLat)
	se = (lonlat[0] + wLon, lonlat[1] - wLat)
	sw = (lonlat[0] - wLon, lonlat[1] - wLat)
	ne = (lonlat[0] + wLon, lonlat[1] + wLat)
	p = Polygon([nw, ne, se, sw, nw])
	g = geojson.Feature(geometry=p, properties={})

	return g.geometry



# def extract_scene_attributes(scene, attribute=None):
# 	timeAcquired = scene["properties"]['acquired']
# 	cloudCoverEst= scene["properties"]['cloud_cover']['estimated']

# 	# links to actual scene data
# 	full_link  = scene["properties"]["links"]['self']
# 	thumb_link = scene["properties"]["links"]['thumbnail']
# 	anal_link  = scene["properties"]['data']['products']['analytic']['full']

# 	attributes = {"timeAcquired":timeAcquired, "cloudCoverEst":cloudCoverEst,\
# 			"full":full_link, "thumb":thumb_link, \
# 			"analytic":anal_link}
# 	if attribute:
# 		return attributes.get(attribute, None)
# 	else:
# 		return attributes

# def get_scenes_metadata(self, filters=None, limit=10):
# 	""" Retrieve a list of images matching a list of filters up to a limit.
# 		To get all images matching the filters, set limit = -1.
# 	"""
# 	nrScenes = 0
# 	scenesData = []
# 	if limit is not None:
# 		filters = {"limit":limit}
# 	nextURL = self.construct_url_request(filters)
# 	while ((nrScenes < limit) if limit != -1 else True) and nextURL:			
# 		r = requests.get(nextURL, auth=(self._key, ''))
# 		r.raise_for_status()
# 		data = r.json()
# 		scenesData += data["features"]
# 		nrScenes = len(scenesData)
# 		nextURL = data["links"].get("next", None) 
# 	scenesData = scenesData[:limit]

# 	print "There are %d relevant scenes. Retrieved data for the first %d."%\
# 		(data['count'], len(scenesData))
# 	return scenesData
