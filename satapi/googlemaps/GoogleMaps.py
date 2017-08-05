import sys, os
import httplib
import urllib
import urlfetch
import cStringIO 
from PIL import Image
from skimage import io
import time


import numpy as np

class GoogleMaps(object):
    """ Lightweight wrapper class for the Google Maps API.
    """
    def __init__(self, key=None):
        self._key = key

    def construct_static_url(self, latlon, zoom=17, imgsize=(500,500),
                            maptype="roadmap", imgformat="jpeg"):
        center = "%2.5f,%2.5f"%latlon if type(latlon)==tuple else latlon
        return construct_googlemaps_url_request(
            center=center,
            zoom=zoom,
            imgsize=imgsize,
            maptype=maptype,
            imgformat=imgformat,
            apiKey=self._key)


    def get_static_map_image(self, request, max_tries=2, \
        filename=None, crop=False):
        numTries = 0
        while numTries < max_tries:
            numTries += 1
        # try:
            img = get_static_google_map(request, \
                filename=filename, crop=crop)
            if img is not None:
                return img
        # except:
            #     print "Error! Trying again (%d/%d) in 5 sec"%(numTries, max_tries)
                time.sleep(5)                
        return None


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


def get_static_google_map(request, filename=None, crop=False):  
    response = urlfetch.fetch(request)

    # check for an error (no image at requested location)
    if response.getheader('x-staticmap-api-warning') is not None:
        return None

    try:
        img = Image.open(cStringIO.StringIO(response.content))
    except IOError:
        print "IOError:", imgdata.read() # print error (or it may return a image showing the error"
        return None
    else:
        img = np.asarray(img.convert("RGB"))

    # there seems not to be any simple way to check for the gray error image
    # that Google throws when going above the API limit -- so here's a hack.
    if (img==224).sum() / float(img.size) > 0.95:
        return None

    # remove the Google watermark at the bottom of the image
    if crop:
        img_shape = img.shape
        img = img[:int(img_shape[0]*0.85),:int(img_shape[1]*0.85)]
    
    if filename is not None:
        basedir = os.path.dirname(filename)
        if not os.path.exists(basedir) and basedir not in ["","./"]:
            os.makedirs(basedir)
        io.imsave(filename, img)
    return img 
 