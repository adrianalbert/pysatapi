from datetime import datetime, timedelta
from dateutil import parser

import requests
from requests.auth import HTTPBasicAuth


BASE_URL  = 'https://api.planet.com/data/v1/'

def search_assets(filters, key, request_type="stats", item_types=None):
  """ 
    Returns list of scenes that correspond to a location and to a time
    interval.
    request_type = stats/quick-search/searches
  """
  if type(filters) in [list, tuple]:
    filters = and_filter(*filters)

  if item_types is None:
    item_types = ["REOrthoTile"]

  endpoint_url = BASE_URL + request_type

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


def search_assets_by_page(filters, key, page_size=10, limit=None):
  '''
  Implements a pagination mechanism for large requests.
  '''
  if type(filters) in [list, tuple]:
    filters = and_filter(*filters)

  session = requests.Session()
  session.auth = (key, '')

  # Create a Saved Search
  saved_search = \
      session.post(BASE_URL + 'searches/', json=filters)

  # after you create a search, save the id. This is what is needed
  # to execute the search.
  saved_search_id = saved_search.json()["id"]

  # What we want to do with each page of search results
  # in this case, just print out each id
  items = []
  def handle_page(page):
      # global items
      items.append(page["features"])

  # How to Paginate:
  # 1) Request a page of search results
  # 2) do something with the page of results
  # 3) if there is more data, recurse and call this method on the next page.
  def fetch_page(search_url):
      page = session.get(search_url).json()
      handle_page(page)
      next_url = page["_links"].get("_next")
      if next_url and ((limit is not None and len(items)<limit) or (limit is None)):
          fetch_page(next_url)

  first_page = \
      ("{}searches/{}" +
          "/results?_page_size={}").format(BASE_URL,saved_search_id,page_size)

  # kick off the pagination
  fetch_page(first_page)

  # items is a list of lists, each of which of size page_size, corresponding to a page of results
  items = [x for page in items for x in page]
  return items


# simple wrapper to define search for pagination requests
def define_search(filters, name="mysearch", item_types=[]):
  mysearch = {
    "name": name,
    "item_types": item_types,
    "filter":filters
  }
  return mysearch


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
    lte = datetime.now()
  elif type(lte)==str:
    lte = parser.parse(lte)
  lte = lte.isoformat().split(".")[0] + ".000Z"
  config = {'lte':lte}

  if gte is not None:
    if type(gte)==str:
      gte = parser.parse(gte)
    gte = gte.isoformat().split(".")[0] + ".000Z"
    config['gte'] = gte
  
  return {
    "type": "DateRangeFilter",
    "field_name": field_name,
    "config": config
  }

# filter any images which are more than 50% clouds
def range_filter(field, gt=None, lt=None):
  config = {}
  if gt is not None:
    config['gt'] = gt
  if lt is not None:
    config['lt'] = lt
  return {
    "type": "RangeFilter",
    "field_name": field,
    "config": config
  }

# create a filter that combines our geo and date filters
# could also use an "OrFilter"
def and_filter(*filters): 
  return {
    "type": "AndFilter",
    "config": filters
  }

# could also use an "OrFilter"
def or_filter(*filters): 
  return {
    "type": "OrFilter",
    "config": filters
  }
