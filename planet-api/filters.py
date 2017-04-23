from datetime import datetime, timedelta

BASE_URL        = "https://api.planet.com/v1/"
BASE_URL_STATS  = 'https://api.planet.com/data/v1/stats'
BASE_URL_SEARCH = 'https://api.planet.com/data/v1/quick-search'


def search_assets(filters, key, url_type="stats", item_types=None):
  """ Returns list of scenes that correspond to a location and to a time
    interval.
  """
  if item_types is None:
    item_types = ["REOrthoTile"]

  if url_type == 'stats':
    endpoint_url = BASE_URL_STATS
  elif url_type == 'quick-search':
    endpoint_url = BASE_URL_SEARCH
  else:
    return None

  # Stats API request object
  endpoint_request = {
    "interval": "day",
    "item_types": item_types,
    "filter": filters
  }

  # fire off the POST request
  result = requests.post(
      endpoint_url,
      auth=HTTPBasicAuth(key, ''),
      json=endpoint_request)

  return result.json()    


# filter for items the overlap with our chosen geometry
def geometry_filter(aoi):
  return {
    "type": "GeometryFilter",
    "field_name": "geometry",
    "config": aoi
  }

# filter images acquired in a certain date range
def date_range_filter(gte=None, lte=None, field_name="acquired"):
  if lte is None:
    lte = endDate = datetime.now().date().isoformat()
  if gte is None:
    gte = (datetime.now()-timedelta(days=365)).date().isoformat()
  
  return {
    "type": "DateRangeFilter",
    "field_name": field_name,
    "config": {
      "gte": gte,
      "lte": lte
    }
  }

# filter any images which are more than 50% clouds
def range_filter(field, gte=None, lte=None):
  return {
    "type": "RangeFilter",
    "field_name": field,
    "config": {
      "lte": lte,
      "lte": gte
    }
  }

# create a filter that combines our geo and date filters
# could also use an "OrFilter"
def and_filter(*filters): 
  return {
    "type": "AndFilter",
    "config": filters
  }
