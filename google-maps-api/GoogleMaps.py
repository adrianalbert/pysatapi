import sys, os
import httplib
import urllib
import cStringIO 
from PIL import Image

import numpy as np

class GoogleMaps(object):
    """ Lightweight wrapper class for the Google Maps API.
    """
    def __init__(self, key=None):
        self._key = key


    def construct_static_url(self, center=None, zoom=None, imgsize=(500,500),
                            maptype="roadmap", imgformat="jpeg"):
        return construct_googlemaps_url_request(
            center=center,
            zoom=zoom,
            imgsize=imgsize,
            maptype=maptype,
            imgformat=imgformat,
            apiKey=self._key)

    def get_static_map_image(self, request, filename=None):
        return get_static_google_map(request, filename=filename)


def construct_googlemaps_url_request(center=None, zoom=None, imgsize=(500,500),
                                     maptype="roadmap", apiKey="", imgformat="jpeg"):
    request =  "http://maps.google.com/maps/api/staticmap?" # base URL, append query params, separated by &
    if center is not None:
        request += "center=%s&"%center.replace(" ","+")
    if zoom is not None:
        request += "zoom=%d&"%zoom  # zoom 0 (all of the world scale ) to 22 (single buildings scale)
    if apiKey is not None:
        request += "key=%s&"%apiKey
    request += "size=%dx%d&"%imgsize  # tuple of ints, up to 640 by 640
    request += "format=%s&"%imgformat
    request += "maptype=%s&sensor=false"%maptype  # roadmap, satellite, hybrid, terrain
    return request


def get_static_google_map(request, filename=None):  
    if filename is not None:
        urllib.urlretrieve(request, filename) 
        return filename
    else:
        web_sock = urllib.urlopen(request)
        imgdata = cStringIO.StringIO(web_sock.read()) # constructs a StringIO holding the image
        try:
            img = Image.open(imgdata)
        except IOError:
            print "IOError:", imgdata.read() # print error (or it may return a image showing the error"
            return None
        else:
            return np.asarray(img.convert("RGB"))