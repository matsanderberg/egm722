import numpy as np
import rasterio as rio
import rasterio.mask as mask
import geopandas as gpd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from shapely.geometry.polygon import Polygon
from cartopy.feature import ShapelyFeature
import matplotlib.patches as mpatches

def generate_handles(labels, colors, edge='k', alpha=1):
    lc = len(colors)  # get the length of the color list
    handles = []
    for i in range(len(labels)):
        handles.append(mpatches.Rectangle((0, 0), 1, 1, facecolor=colors[i % lc], edgecolor=edge, alpha=alpha))
    return handles

def percentile_stretch(img, pmin=0., pmax=100.):
    '''
    This is where you should write a docstring.
    '''
    # here, we make sure that pmin < pmax, and that they are between 0, 100
    if not 0 <= pmin < pmax <= 100:
        raise ValueError('0 <= pmin < pmax <= 100')
    # here, we make sure that the image is only 2-dimensional
    if not img.ndim == 2:
        raise ValueError('Image can only have two dimensions (row, column)')

    minval = np.percentile(img, pmin)
    maxval = np.percentile(img, pmax)

    stretched = (img - minval) / (maxval - minval)  # stretch the image to 0, 1
    stretched[img < minval] = 0  # set anything less than minval to the new minimum, 0.
    stretched[img > maxval] = 1  # set anything greater than maxval to the new maximum, 1.

    return stretched


def img_display(img, ax, bands, stretch_args=None, **imshow_args):
    '''
    This is where you should write a docstring.
    '''
    dispimg = img.copy().astype(np.float32)  # make a copy of the original image,
    # but be sure to cast it as a floating-point image, rather than an integer

    for b in range(img.shape[0]):  # loop over each band, stretching using percentile_stretch()
        if stretch_args is None:  # if stretch_args is None, use the default values for percentile_stretch
            dispimg[b] = percentile_stretch(img[b])
        else:
            dispimg[b] = percentile_stretch(img[b], **stretch_args)

    # next, we transpose the image to re-order the indices
    dispimg = dispimg.transpose([1, 2, 0])

    # finally, we display the image
    handle = ax.imshow(dispimg[:, :, bands], **imshow_args)

    return handle, ax


# ------------------------------------------------------------------------
counties = gpd.read_file('../Week3/data_files/Counties.shp')
outline = gpd.read_file('../Week2/data_files/NI_outline.shp')
towns = gpd.read_file('../Week2/data_files/Towns.shp')
counties = counties.to_crs(epsg=32629)
outline = outline.to_crs(epsg=32629)
towns = towns.to_crs(epsg=32629)

# note - rasterio's open() function works in much the same way as python's - once we open a file,
# we have to make sure to close it. One easy way to do this in a script is by using the with statement shown
# below - once we get to the end of this statement, the file is closed.
with rio.open('data_files/NI_Mosaic.tif') as dataset:
    img = dataset.read()
    xmin, ymin, xmax, ymax = dataset.bounds

with rio.open('data_files/NI_Mosaic.tif') as dataset:
    masked_img = mask.mask(dataset, outline['geometry'], crop=True)

# your code goes here!
plt.ion()

myCRS = ccrs.UTM(29) # note that this matches with the CRS of our image
my_kwargs = {'extent': [xmin, xmax, ymin, ymax],
             'transform': myCRS}

my_stretch = {'pmin': 0.1, 'pmax': 99.9}

# create a figure of size 10x10 (representing the page size in inches
fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw=dict(projection=myCRS))

xmin, ymin, xmax, ymax = outline.total_bounds
# using the boundary of the shapefile features, zoom the map to our area of interest
ax.set_extent([xmin, xmax, ymin, ymax], crs=myCRS)

# add county outlines to the map
county_outlines = ShapelyFeature(counties['geometry'], myCRS, edgecolor='r', facecolor='none')
ax.add_feature(county_outlines)
ax.patch.set_alpha(0)

# Cities and Towns
# ShapelyFeature creates a polygon, so for point data we can just use ax.plot()
cities = towns[towns.STATUS == 'City']
towns_only = towns[towns.STATUS == 'Town']
city_handle = ax.plot(cities.geometry.x, cities.geometry.y, 'or', ms=6, transform=myCRS)
town_handle = ax.plot(towns_only.geometry.x, towns_only.geometry.y, 's', color='0.5', ms=6, transform=myCRS)

# add the text labels for the towns and cities
for i, row in towns.iterrows():
    x, y = row.geometry.x, row.geometry.y
    plt.text(x, y, row['TOWN_NAME'].title(), fontsize=8, transform=myCRS) # use plt.text to place a label at x,y

# Add the raster image
h, ax = img_display(img, ax, [2, 1, 0], stretch_args=my_stretch, **my_kwargs)

# save the figure
fig.savefig('map.png', dpi=300, bbox_inches='tight')