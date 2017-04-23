import geopandas as gdp
import numpy as np


def get_bounds(gdf):
    bounds = np.array(gdf['geometry'].apply(lambda p: list(p.bounds)).values.tolist())
    xmin = bounds[:,[0,2]].min()
    xmax = bounds[:,[0,2]].max()
    ymin = bounds[:,[1,3]].min()
    ymax = bounds[:,[1,3]].max()
    return xmin, ymin, xmax, ymax


def compute_stats(gdf, prj=""):
    ''' 
    Statistics about the polygons in the geo data frame.
    '''
    lonmin, latmin, lonmax, latmax = get_bounds(gdf)
    xmin, ymin = satimg.lonlat2xy((lonmin, latmin), prj=prj)
    xmax, ymax = satimg.lonlat2xy((lonmax, latmax), prj=prj)

    box_area =  (xmax-xmin) / 1.0e3 * (ymax-ymin) / 1.0e3
    L = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2) / 1.0e3 / np.sqrt(2)
    classified_area = gdf['SHAPE_AREA'].sum()
    frac_classified = classified_area/box_area

    print "Spatial extent: %2.2f km." % L
    print "Land use classified area: %2.3f km^2 (%2.2f of total area covered within bounds %2.3f km^2)"%(classified_area, frac_classified, box_area)
    
    return L, frac_classified