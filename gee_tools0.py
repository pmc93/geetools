# -*- coding: utf-8 -*-
#%%
"""
Created on Mon Oct 31 22:51:14 2022


@author: au701230
"""

#download NDVI

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 10:38:39 2022

@author: au701230
"""

import gdal 
import matplotlib.pyplot as plt  
import numpy as np

import fiona
import ee
import pandas as pd
import webbrowser


import glob

import requests

import utm

from pyproj import CRS

import os

from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from zipfile import ZipFile


#%%
def authenticate():
    ee.Authenticate()
    
def initialize():
    ee.Initialize()

def getAOI(lat, long, dist_x=5000, dist_y=5000):
    
    utm_coords = utm.from_latlon(lat, long)
    utm_x = utm_coords[0]
    utm_y = utm_coords[1]
    utm_zone_number = utm_coords[2]
    utm_zone_letter = utm_coords[3]
    
    print("UTM Coordinates:")
    print("Easting:", utm_x)
    print("Northing:", utm_y)
    print("Zone Number:", utm_zone_number)
    print("Zone Letter:", utm_zone_letter)
    
    min_x = utm_x - dist_x
    max_x = utm_x + dist_x
    min_y = utm_y - dist_y
    max_y = utm_y + dist_y
    
    lat1, long1 = utm.to_latlon(min_x, min_y, utm_zone_number, utm_zone_letter)
    lat2, long2 = utm.to_latlon(min_x, max_y, utm_zone_number, utm_zone_letter)
    lat3, long3 = utm.to_latlon(max_x, max_y, utm_zone_number, utm_zone_letter)
    lat4, long4 = utm.to_latlon(max_x, min_y, utm_zone_number, utm_zone_letter)
    
    aoi = [[long1, lat1],
           [long2, lat2],
           [long3, lat3],
           [long4, lat4],
           [long1, lat1]]
    
    aoi = ee.Geometry.Polygon(aoi)
    
    espg_str = getEPSG(long1, long3, lat1, lat3)
    
    return aoi, espg_str

def getEPSG(min_long, max_long, min_lat, max_lat):
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=min_long,
            south_lat_degree=min_lat,
            east_lon_degree=max_lat,
            north_lat_degree=max_lat,
        ),
    )
    
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    
    return f'{utm_crs.to_authority()[0]}:{utm_crs.to_authority()[1]}'
    
#%%

def createS2Coll(start_date, end_date, aoi, cloud_threshold=1):
    
    s2_coll = ee.ImageCollection('COPERNICUS/S2').filterBounds(aoi).filterDate(start_date, end_date).filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_threshold)

    return s2_coll

def downloadColl(coll, aoi, path_to_downloads, scale=20, idx_start=0, idx_end=-1):
        
    coll_list = coll.toList(coll.size())
    
    os.chdir(path_to_downloads)
    
    total_items = coll.size().getInfo()
    print(f'{coll.size().getInfo()} images in collection.')
    if idx_end == -1:
        idx_end = coll.size().getInfo()
    
    print(f'{total_items} images will be downloaded.')
    start_time = time.time()
    for i in range(idx_start, idx_end):
        
        img = ee.Image(coll_list.get(i))
        
        url = img.getDownloadURL({
            'scale': scale,
            'bands' : ['B1', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
            'fileFormat': 'GeoTIFF',
            'region': aoi,
            'maxPixels': 1e10})
        
        dashboardFile = requests.get(url, allow_redirects=True)
        
        date = ee.Date(img.get('system:time_start'))
        
        img_date = date.format('Y-M-d').getInfo()
        
        open(img_date+'.zip', 'wb').write(dashboardFile.content)
        
        # Calculate progress percentage
        progress = int(i / total_items * progress_width)
        percent = int(i / total_items * 100)
        
        elapsed_time = time.time() - start_time
        remaining_time = (elapsed_time / (i+1)) * (total_items - i)

        # Print the progress bar and percentage
        bar = '[' + '#' * progress + ' ' * (progress_width - progress) + ']'
        print(f'\r{bar} {percent}% Complete | ETA: {remaining_time:.1f}s', end='', flush=True)



#%%
import time
def process_data(data):
    # Simulating some time-consuming process
    time.sleep(0.1)

# Sample loop
data = range(100)

# Initialize variables
total_items = len(data)
progress_width = 50

# Loop through the data
for i, item in enumerate(data, 1):
    process_data(item)

    # Calculate progress percentage
    progress = int(i / total_items * progress_width)
    percent = int(i / total_items * 100)

    # Print the progress bar and percentage
    bar = '[' + '#' * progress + ' ' * (progress_width - progress) + ']'
    print(f'\r{bar} {percent}% Complete', end='', flush=True)

# Print a new line after the loop finishes
print()

#%%

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df

def t_modis_to_celsius(t_modis):
    """Converts MODIS LST units to degrees Celsius."""
    t_celsius =  0.02*t_modis - 273.15
    return t_celsius


#%%


def calcNDVI(folder_path):

    tif_files = os.listdir(folder_path)
    
    tif_file = tif_files[0].split('.')[0] 
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B8.tif')

    nir = tif.GetRasterBand(1).ReadAsArray()  / 10000 

    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B4.tif')

    red = tif.GetRasterBand(1).ReadAsArray()  / 10000 

    ndvi = (nir - red) / (nir + red)
    
    tif = None

    
    return ndvi


#%%
from osgeo import gdal


folder_path = r'C:\Users\au701230\OneDrive - Aarhus Universitet\Desktop\gee_stuff\downloads\2015-12-16'
calcNDVI(folder_path)

#%%

from zipfile import ZipFile



#folders = os.listdir(base_dir)


#folder_path = os.path.join(base_dir, folders)



def calcVARI(folder_path):

    tif_files = os.listdir(folder_path)
    
    tif_file = tif_files[0].split('.')[0] 
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B3.tif')
    
    green = tif.GetRasterBand(1).ReadAsArray()  / 10000 

    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B4.tif')

    red = tif.GetRasterBand(1).ReadAsArray()  / 10000 
    
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B2.tif')

    blue = tif.GetRasterBand(1).ReadAsArray()  / 10000 

    vari = (green - red) / (green + red - blue)

    
    return vari, np.nanmean(vari)


#%%get temperature

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df

def t_modis_to_celsius(t_modis):
    """Converts MODIS LST units to degrees Celsius."""
    t_celsius =  0.02*t_modis - 273.15
    return t_celsius

def getLST(lat, long, start_date, end_date, scale=1000):
    
    poi = ee.Geometry.Point(long, lat)
    
    lst = ee.ImageCollection('MODIS/006/MOD11A1')

    # Selection of appropriate bands and dates for LST.
    lst = lst.select('LST_Day_1km', 'QC_Day').filterDate(start_date, end_date)
    
    lst_poi = lst.getRegion(poi, scale).getInfo()
    
    mean_T = t_modis_to_celsius(lst.mean().sample(poi, scale).first().get('LST_Day_1km').getInfo())
    print(f'Average daytime LST at ({np.round(lat, 2)}, {np.round(long, 2)}) is {np.round(mean_T, 2)} °C.')
    
    lst_df = ee_array_to_df(lst_poi, ['LST_Day_1km'])
    lst_df['LST_Day_1km'] = lst_df['LST_Day_1km'].apply(t_modis_to_celsius)

    return lst_df



#%%
def sum_resampler(coll, freq, unit, scale_factor, band_name):
    """
    This function aims to resample the time scale of an ee.ImageCollection.
    The function returns an ee.ImageCollection with the averaged sum of the
    band on the selected frequency.

    coll: (ee.ImageCollection) only one band can be handled
    freq: (int) corresponds to the resampling frequence
    unit: (str) corresponds to the resampling time unit.
                must be 'day', 'month' or 'year'
    scale_factor (float): scaling factor used to get our value in the good unit
    band_name (str) name of the output band
    """

    # Define initial and final dates of the collection.
    firstdate = ee.Date(
        coll.sort("system:time_start", True).first().get("system:time_start")
    )

    lastdate = ee.Date(
        coll.sort("system:time_start", False).first().get("system:time_start")
    )

    # Calculate the time difference between both dates.
    # https://developers.google.com/earth-engine/apidocs/ee-date-difference
    diff_dates = lastdate.difference(firstdate, unit)

    # Define a new time index (for output).
    new_index = ee.List.sequence(0, ee.Number(diff_dates), freq)

    # Define the function that will be applied to our new time index.
    def apply_resampling(date_index):
        # Define the starting date to take into account.
        startdate = firstdate.advance(ee.Number(date_index), unit)

        # Define the ending date to take into account according
        # to the desired frequency.
        enddate = firstdate.advance(ee.Number(date_index).add(freq), unit)

        # Calculate the number of days between starting and ending days.
        diff_days = enddate.difference(startdate, "day")

        # Calculate the composite image.
        image = (
            coll.filterDate(startdate, enddate)
            .mean()
            .multiply(diff_days)
            .multiply(scale_factor)
            .rename(band_name)
        )

        # Return the final image with the appropriate time index.
        return image.set("system:time_start", startdate.millis())

    # Map the function to the new time index.
    res = new_index.map(apply_resampling)

    # Transform the result into an ee.ImageCollection.
    res = ee.ImageCollection(res)

    return res

#%%

def getP_PET(lat, long, start_date, end_date, scale=1000):
    
    poi = ee.Geometry.Point(long, lat)
    
    pet = ee.ImageCollection("MODIS/006/MOD16A2").select(["PET", "ET_QC"])
    pet = pet.filterDate(start_date, end_date)

    pet_m = sum_resampler(pet.select("PET"), 1, "month", 0.0125, "PET")
    
    if (lat < 50 and lat > 50) == True:
        print('Chirps data is not available.')
        
        meteo = pet_m

        meteo_arr = meteo.getRegion(poi, scale).getInfo()
        
        meteo_df = ee_array_to_df(meteo_arr, ["PET"]).sort_index()
        
        return meteo_df

    else:
        
        p = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").select("precipitation")
        p = p.filterDate(start_date, end_date)
        
        p_m = sum_resampler(p, 1, "month", 1, "P")
        
        meteo = p_m.combine(pet_m)
    
        meteo_arr = meteo.getRegion(poi, scale).getInfo()
    
        # Transform the array into a pandas dataframe and sort the index.
        meteo_df = ee_array_to_df(meteo_arr, ["P", "PET"]).sort_index()
    
    return meteo_df

#%%
start_date = '2015-1-1'
end_date = '2023-4-5'
scale = 1000
lat, long = -27.35479730880535, 32.682550782680124

#long, lat = 10.295510429407095, 55.75336123237324

poi = ee.Geometry.Point(long, lat)

meteo_df = getP_PET(lat, long, start_date, end_date, scale=1000)

lst_df = getLST(lat, long, start_date, end_date, scale=1000)

#%%


fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(lst_df['datetime'], lst_df['LST_Day_1km'],
           label=' ', alpha=0.2, c='C3', s= 20)

ax.set_xlabel('Date')
ax.set_ylabel('Temperature [°C]', c='C3')
#ax.set_ylim(-0, 40)
#ax.grid(lw=0.2)
ax.tick_params(axis='y', colors='C3')

ax2 = ax.twinx()
ax2.bar(meteo_df['datetime'], meteo_df['PET'], width=20, color='C1', alpha=0.8)
ax2.bar(meteo_df['datetime'], meteo_df['P'], width=20, color='C0', alpha=0.5)
ax2.set_ylim(0, 800)
ax2.set_yticks([0, 100, 200, 300])  


fig.text(1.055, 0.05, 'Preciptation [mm]', color='C0', rotation='vertical', transform=ax.transAxes)
fig.text(1.085, 0.0, 'Evapotranspiration [mm]', color='C1', rotation='vertical', transform=ax.transAxes)



#%% Download S2 data
initialize()
start_date = '2015-1-1'
end_date = '2023-4-5'
aoi, espg_str = getAOI(-27.251377, 32.5)

aoi = ee.Geometry.Polygon([[32.54957741070813,-27.42496233553337],
                           [32.72707527447766,-27.42496233553337],
                           [32.72707527447766,-27.2611967344842],
                           [32.54957741070813,-27.2611967344842],
                           [32.54957741070813,-27.42496233553337]])

s2_coll = createS2Coll(start_date, end_date, aoi, cloud_threshold=0.1)
path_to_downloads = r'C:\Users\au701230\OneDrive - Aarhus Universitet\Desktop\gee_stuff\downloads'
downloadColl(s2_coll, aoi, path_to_downloads, idx_start=0, idx_end=-1)
    

#%%
path_to_downloads = r'C:\Users\au701230\OneDrive - Aarhus Universitet\Desktop\gee_stuff\downloads'

zip_files = glob.glob(path_to_downloads+'/*.zip')

for i in range(len(zip_files)):
    
    with ZipFile(zip_files[i], 'r') as zip:
        
        #zip.printdir()
        zip.extractall(zip_files[i][:-4])
    
    os.remove(zip_files[i])
#%%
from osgeo import gdal


folder_path = r'C:\Users\au701230\OneDrive - Aarhus Universitet\Desktop\gee_stuff\downloads\2016-1-5'

def calcNDVI(folder_path):

    tif_files = os.listdir(folder_path)
    
    tif_file = tif_files[0].split('.')[0] 
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B8.tif')
    
    nir = tif.GetRasterBand(1).ReadAsArray()  / 10000 
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B4.tif')
    
    red = tif.GetRasterBand(1).ReadAsArray()  / 10000 
    
    ndvi = (nir - red) / (nir + red)
    
    width = tif.RasterXSize
    height = tif.RasterYSize
    
    gt_input = tif.GetGeoTransform()
    
    out_path = os.path.join(folder_path, 'ndvi3.tif')
    
    with rasterio.open(out_path, 'w', 
                       width=width, 
                       height=height,  
                       count=1,
                       dtype='float32', 
                       transform=Affine.from_gdal(*gt_input)) as dst:
        
        dst.write(ndvi, indexes=1)
        
def calcNDWI(folder_path, threshold=0.3):

    
    tif_files = os.listdir(folder_path)
    
    tif_file = tif_files[0].split('.')[0] 
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B12.tif')
    
    swir = tif.GetRasterBand(1).ReadAsArray()  / 10000 
    
    tif = gdal.Open(os.path.join(folder_path,
                                 tif_file) + '.B3.tif')
    
    green = tif.GetRasterBand(1).ReadAsArray()  / 10000 
    
    ndwi = (green - swir) / (green + swir)
    
    idx = np.where(ndwi > threshold)

    ndwi2 = ndwi.copy()

   
    
    width = tif.RasterXSize
    height = tif.RasterYSize
    
    gt_input = tif.GetGeoTransform()
    
    out_path = os.path.join(folder_path, 'ndwi.tif')
    
    with rasterio.open(out_path, 'w', 
                       width=width, 
                       height=height,  
                       count=1,
                       dtype='float32', 
                       transform=Affine.from_gdal(*gt_input)) as dst:
        
        dst.write(ndwi, indexes=1)

#%%
import rasterio
from rasterio import Affine

dst_crs=espg_str



out_path = r'C:\Users\au701230\OneDrive - Aarhus Universitet\Desktop\gee_stuff\downloads\2016-1-5\ndvi.tif'


calcNDWI(folder_path)

#%%

def adjust_contrast(image, contrast):
    midpoint = 0.5  # Midpoint value around which to scale the pixel values
    adjusted_image = (image - midpoint) * contrast + midpoint
    adjusted_image = np.clip(adjusted_image, 0, 1)
    return adjusted_image

folder_path = r'C:\Users\au701230\OneDrive - Aarhus Universitet\Desktop\gee_stuff\downloads\2016-1-5'

tif_files = os.listdir(folder_path)

tif_file = tif_files[0].split('.')[0] 

tif = gdal.Open(os.path.join(folder_path,
                             tif_file) + '.B3.tif')

green = tif.GetRasterBand(1).ReadAsArray()  / 10000 

tif = gdal.Open(os.path.join(folder_path,
                             tif_file) + '.B4.tif')

red = tif.GetRasterBand(1).ReadAsArray()  / 10000 


tif = gdal.Open(os.path.join(folder_path,
                             tif_file) + '.B2.tif')

blue = tif.GetRasterBand(1).ReadAsArray()  / 10000 



# Extract individual bands from the multiband data
scale = 1
red_band = np.clip(red, 0, scale)/scale
green_band = np.clip(green, 0, scale)/scale
blue_band = np.clip(blue, 0, scale)/scale


#%%
rgb_image = np.dstack((red_band, green_band, blue_band))
def visualise_rgb(img, clip=[0.3,0.3,0.3],display=True):
        """Visulaise RGB image with given clip values and return image"""

        # Scale image
        
        
        # Get RGB channels
        rgb = img

        #clip rgb values
        rgb[:,:,0] = np.clip(rgb[:,:,0],0,clip[0])/clip[0]
        rgb[:,:,1] = np.clip(rgb[:,:,1],0,clip[1])/clip[1]
        rgb[:,:,2] = np.clip(rgb[:,:,2],0,clip[2])/clip[2]

        

        if display:

                #Display histograms of pixel intesity with given clip values
                fig, axs = plt.subplots(1,4,figsize=(22,5))
                fig.patch.set_facecolor('xkcd:white')

                labels = ['Red','Green','Blue']
                for i,ax in enumerate(axs[0:3]):
                        ax.hist(img[3-i].flatten(),bins=100)
                        ax.set_title(labels[i],size=20,fontweight="bold")
                        ax.axvline(clip[i],color="red",linestyle="--")
                        ax.set_yticks([])

                #Display RGB image
                axs[3].imshow(rgb)
                axs[3].set_title("RGB",size=20,fontweight="bold")
                axs[3].set_xticks([])
                axs[3].set_yticks([])

        return rgb

#%%
rgb_image = np.dstack((red_band, green_band, blue_band))
rgb = visualise_rgb(rgb_image,[0.25,0.2,0.26])
#%%%
#red_band = red_band / np.max(red_band)
#green_band = green_band / np.max(green_band)
#blue_band = blue_band / np.max(blue_band)

# Stack the bands together to create a color image


brightness_factor = 1
rgb_image = adjust_contrast(rgb_image, contrast=1)
rgb_image = np.clip(rgb_image * brightness_factor, 0, 1)


def apply_color_filter(image, red_coeff, green_coeff, blue_coeff):
    filtered_image = np.dstack((
        image[:, :, 0] * red_coeff,
        image[:, :, 1] * green_coeff,
        image[:, :, 2] * blue_coeff
    ))
    filtered_image = np.clip(filtered_image, 0, 1)
    return filtered_image

red_coeff = 1
green_coeff = 1
blue_coeff = 1
rgb_image = apply_color_filter(rgb_image, red_coeff, green_coeff, blue_coeff)
# Plot the RGB image
plt.figure(figsize=(10, 10))
plt.imshow(rgb_image)
plt.axis('off')
plt.show()

#%%

def plotRaster(tif_file, cmap='terrain',clabel='',
               vmin = None, vmax = None,
               xlims = None, ylims = None,
               n_xticks=3, n_yticks=6):
    
    tif = gdal.Open(tif_file)

    width = tif.RasterXSize
    height = tif.RasterYSize
    gt = tif.GetGeoTransform()
    min_x = gt[0]
    min_y = gt[3] + width*gt[4] + height*gt[5] 
    max_x = gt[0] + width*gt[1] + height*gt[2]
    max_y = gt[3] 

    range_x = max_x - min_x
    range_y = max_y - min_y

    sig_fig = 4
    dx = (max_x - min_x)  / n_xticks
    dy = (max_y - min_y) / n_yticks

    dx = int(round(dx, -sig_fig))
    dy = int(round(dy, -sig_fig))

    xtick_labs = np.arange(min_x, max_x, dx).astype(int)
    xtick_locs = np.linspace(0, width, len(xtick_labs)).astype(int)

    ytick_labs = np.arange(min_y, max_y, dx).astype(int)
    ytick_locs = np.linspace(0, height, len(ytick_labs)).astype(int)
    
    band_1 = tif.GetRasterBand(1) 

    img = band_1.ReadAsArray()  

    fig, ax = plt.subplots(figsize=(6,8))
    
    shw = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    
    bar = plt.colorbar(shw, orientation='horizontal')
    bar.set_label(clabel, rotation=0)
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labs)

    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_labs[::-1])
    
        
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')

    ax.grid('both', c='black')

    fig.tight_layout()

#%%
plotRaster(r'C:/Users/au701230/OneDrive - Aarhus Universitet/Desktop/gee_stuff/downloads/2016-1-5/ndwi.tif', vmin=0.25, vmax=0.3, cmap='Blues')

#%%

imshow(edges_prewitt_vertical, cmap='gray')
#%%

def plotDEM(tif_file, cmap='terrain',clabel='',
            vmin = None, vmax = None,
            xlims = None, ylims = None,
            n_xticks=3, n_yticks=6):
    
    tif = gdal.Open(tif_file)

    width = tif.RasterXSize
    height = tif.RasterYSize
    gt = tif.GetGeoTransform()
    min_x = gt[0]
    min_y = gt[3] + width*gt[4] + height*gt[5] 
    max_x = gt[0] + width*gt[1] + height*gt[2]
    max_y = gt[3] 

    range_x = max_x - min_x
    range_y = max_y - min_y

    sig_fig = 4
    dx = (max_x - min_x)  / n_xticks
    dy = (max_y - min_y) / n_yticks

    dx = int(round(dx, -sig_fig))
    dy = int(round(dy, -sig_fig))

    xtick_labs = np.arange(min_x, max_x, dx).astype(int)
    xtick_locs = np.linspace(0, width, len(xtick_labs)).astype(int)

    ytick_labs = np.arange(min_y, max_y, dx).astype(int)
    ytick_locs = np.linspace(0, height, len(ytick_labs)).astype(int)
    
    band_1 = tif.GetRasterBand(1) 

    img = band_1.ReadAsArray()  

    fig, ax = plt.subplots(figsize=(6,8))
    
    shw = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    
    bar = plt.colorbar(shw, orientation='horizontal')
    bar.set_label(clabel, rotation=0)
    ax.set_xticks(xtick_locs)
    ax.set_xticklabels(xtick_labs)

    ax.set_yticks(ytick_locs)
    ax.set_yticklabels(ytick_labs[::-1])
    
        
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')

    ax.grid('both', c='black')

    fig.tight_layout()
    
#%%



n = S2_coll.size().getInfo()

S2_list = S2_coll.toList(S2_coll.size())

n = S2_list.size().getInfo()



for i in range(3):
    img = ee.Image(S2_list.get(i))
    
    ndvi_img = img.select('ndvi').clip(poly)
    
    url = ndvi_S2.getDownloadURL({
        'scale': 50,
        'crs':'EPSG:32336',
        'fileFormat': 'GeoTIFF',
        'region': poly,
        'maxPixels': 1e10})
    
    webbrowser.open(url) 

#%%


plt.show(img)