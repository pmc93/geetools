
#%%

import numpy as np
import matplotlib.pyplot as plt
import contextily as cx
import ee
import time
import requests
import os

import glob
from zipfile import ZipFile

from osgeo import gdal 
from skimage import exposure





#%%

class Project:

    def __init__(self, start_date, end_date, bounds):
        self.start_date = start_date
        self.end_date = end_date
        self.lats = np.array(bounds)[:,1]
        self.longs = np.array(bounds)[:,0]
        self.bounds = bounds
        
    def authenticate(self):
        ee.Initialize(project='ee-pamcl')
        ee.Authenticate()
    
    def initialize(self):
        self.aoi = ee.Geometry.Polygon(self.bounds)

    def downloadLandsat(self, source, bands=['B1', 'B2', 'B3', 'B4', 'B5', 'QA_PIXEL']):

        if source == 'L4':
            L_collection = ee.ImageCollection("LANDSAT/LT04/C02/T1").filterBounds(self.aoi)
        
        if source == 'L5':
            L_collection = ee.ImageCollection("LANDSAT/LT05/C02/T1").filterBounds(self.aoi)

        if source == 'L7':
            L_collection = ee.ImageCollection("LANDSAT/LE07/C02/T1").filterBounds(self.aoi)

        if source == 'L8':
            L_collection = ee.ImageCollection("LANDSAT/LC08/C02/T1").filterBounds(self.aoi)

        if source == 'L9':
            L_collection = ee.ImageCollection("LANDSAT/LC09/C02/T1").filterBounds(self.aoi)

        
        L_collection = L_collection.filterDate(self.start_date, self.end_date)
        #L_collection = L_collection.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_threshold)

        L_collection_list = L_collection.toList(L_collection.size())

        total_items = L_collection.size().getInfo()
        print(f'{L_collection.size().getInfo()} images in collection.')

        print(f'{total_items} images will be downloaded.')
        start_time = time.time()

        parent_dir = os.getcwd()

        os.chdir(os.path.join(parent_dir, 'downloads'))

        for i in range(total_items):
            
            img = ee.Image(L_collection_list.get(i))
            
            url = img.getDownloadURL({
                'scale': 30,
                'bands' : bands,
                'fileFormat': 'GeoTIFF',
                'region': self.aoi,
                'maxPixels': 1e10})
            
            dashboardFile = requests.get(url, allow_redirects=True)
            
            date = ee.Date(img.get('system:time_start'))
            
            img_date = date.format('Y-M-d').getInfo()
            
            open(img_date+'.zip', 'wb').write(dashboardFile.content)
            
            # Calculate progress percentage
            progress_width = 15
            progress = int(i / total_items * progress_width)
            percent = int(i / total_items * 100)
            
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (i+1)) * (total_items - i)

            # Print the progress bar and percentage
            bar = '[' + '#' * progress + ' ' * (progress_width - progress) + ']'
            print(f'\r{bar} {percent}% Complete | ETA: {remaining_time:.1f}s', end='', flush=True)

        os.chdir(parent_dir)



    def downloadS2(self, bands=['B1', 'B2', 'B3', 'B4', 'B8', 'B11', 'B12'], cloud_threshold=1):

        S2_collection = ee.ImageCollection('COPERNICUS/S2').filterBounds(self.aoi)
        S2_collection = S2_collection.filterDate(self.start_date, self.end_date)
        S2_collection = S2_collection.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', cloud_threshold)

        S2_collection_list = S2_collection.toList(S2_collection.size())

        total_items = S2_collection.size().getInfo()
        print(f'{S2_collection.size().getInfo()} images in collection.')

        print(f'{total_items} images will be downloaded.')
        start_time = time.time()

        parent_dir = os.getcwd()

        os.chdir(os.path.join(parent_dir, 'downloads'))
        
        for i in range(total_items):
            
            img = ee.Image(S2_collection_list.get(i))
            
            url = img.getDownloadURL({
                'scale': 15,
                'bands' : bands,
                'fileFormat': 'GeoTIFF',
                'region': self.aoi,
                'maxPixels': 1e10})
            
            dashboardFile = requests.get(url, allow_redirects=True)
            
            date = ee.Date(img.get('system:time_start'))
            
            img_date = date.format('Y-M-d').getInfo()
            
            open(img_date+'.zip', 'wb').write(dashboardFile.content)
            
            # Calculate progress percentage
            progress_width = 15
            progress = int(i / total_items * progress_width)
            percent = int(i / total_items * 100)
            
            elapsed_time = time.time() - start_time
            remaining_time = (elapsed_time / (i+1)) * (total_items - i)

            # Print the progress bar and percentage
            bar = '[' + '#' * progress + ' ' * (progress_width - progress) + ']'
            print(f'\r{bar} {percent}% Complete | ETA: {remaining_time:.1f}s', end='', flush=True)

        os.chdir(parent_dir)

    def plotMap(self, background='imagery'):

        if background=='imagery':
            source = cx.providers.Esri.WorldImagery
            tem_color = 'w'

        elif background=='osm':
            source = cx.providers.OpenStreetMap.Mapnik
            tem_color = 'k'

        fig, ax = plt.subplots(1, 1)

        ax.plot(self.longs, self.lats, lw=0)

        cx.add_basemap(ax, crs='epsg:4326',  source=source,
                       attribution=False)
        
        ax.grid()

        ax.set_ylabel('Latitude [°]')
        ax.set_xlabel('Longitude [°]')


    def calcNDWI(self, folder_path, method, threshold=0.3):

        tif_files = os.listdir(folder_path)
        
        tif_file = tif_files[0].split('.')[0] 
        
        if method == 'Sentinel':
        
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B12.tif')
            
        if method == 'Landsat':
            
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B5.tif')
            
        
        swir = tif.GetRasterBand(1).ReadAsArray()  / 10000 
        
        if method == 'Sentinel':
        
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B3.tif')
            
        if method == 'Landsat':
            
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B2.tif')
        
        green = tif.GetRasterBand(1).ReadAsArray()  / 10000 
        
        ndwi = (green - swir) / (green + swir)
        
        idx = np.where(ndwi > threshold)

        ndwi_mask = ndwi.copy()

        ndwi_mask[:] = 0

        ndwi_mask[idx] = 1

        return ndwi, ndwi_mask

    def calcNDVI(self, folder_path, method):

        tif_files = os.listdir(folder_path)
        
        tif_file = tif_files[0].split('.')[0] 
        
        if method == 'Sentinel':
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B8.tif')
        
        if method == 'Landsat':
        
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B4.tif')

        nir = tif.GetRasterBand(1).ReadAsArray()  / 10000 
        
        if method == 'Sentinel':

            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B4.tif')
            
        if method == 'Landsat':
            
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B3.tif')

        red = tif.GetRasterBand(1).ReadAsArray()  / 10000 

        ndvi = (nir - red) / (nir + red)

        
        return ndvi


    def readCloud(self, folder_path):

        tif_files = os.listdir(folder_path)

        tif_file = tif_files[0].split('.')[0]

        tif = gdal.Open(os.path.join(folder_path,
                                    tif_file) + '.QA_PIXEL.tif')

        cloud = tif.GetRasterBand(1).ReadAsArray()

        return cloud
    

    def adjust_contrast(self, image, contrast):
        midpoint = 0.5  # Midpoint value around which to scale the pixel values
        adjusted_image = (image - midpoint) * contrast + midpoint
        adjusted_image = np.clip(adjusted_image, 0, 1)
        return adjusted_image


    def getRGB(self, folder_path, method = 'Sentinel'):


        tif_files = os.listdir(folder_path)
        
        tif_file = tif_files[0].split('.')[0] 
        
        if method == 'L8':
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B3.tif')
            
        if method == 'Landsat' or method == 'Sentinel':
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B2.tif')
        
        green = tif.GetRasterBand(1).ReadAsArray()

        if method == 'L8':
        
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B4.tif')
            
        if method == 'Landsat' or method == 'Sentinel':
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B3.tif')
        
        red = tif.GetRasterBand(1).ReadAsArray()  
        
        if method == 'L8' or method == 'Sentinel':
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B2.tif')
            
        if method == 'Landsat' or method == 'Sentinel':
            tif = gdal.Open(os.path.join(folder_path,
                                        tif_file) + '.B1.tif')
        
        blue = tif.GetRasterBand(1).ReadAsArray()
        
        
        # Extract individual bands from the multiband data
        if method == 'Landsat' or method == 'L8':
            scale =  255
        else:
            scale = 10000
        red_band = np.clip(red, 0, scale)/scale
        green_band = np.clip(green, 0, scale)/scale
        blue_band = np.clip(blue, 0, scale)/scale
        
        
        # Stack the bands together to create a color image
        
        rgb_image = np.dstack((red_band, green_band, blue_band))
        
        return rgb_image


    def apply_color_filter(self, image, red_coeff, green_coeff, blue_coeff):
        filtered_image = np.dstack((
            image[:, :, 0] * red_coeff,
            image[:, :, 1] * green_coeff,
            image[:, :, 2] * blue_coeff
        ))
        filtered_image = np.clip(filtered_image, 0, 1)
        return filtered_image
    
    def unzip_files(self, path_to_downloads):
        
        zip_files = glob.glob(path_to_downloads+'/*.zip')

        for i in range(len(zip_files)):

            print(zip_files[i])
            
        
            with ZipFile(zip_files[i], 'r') as zip:
                zip.extractall(zip_files[i][:-4])
        
            os.remove(zip_files[i])

    def match_exposure(self, rgb1, rgb2)
        rgb2_matched = exposure.match_histograms(rgb1, rgb2, channel_axis=-1)
        return rgb2_matched
    
    def plot_raster(self, img, title, ax, real_d, pixel_size, vmin=None, vmax=None, cmap='viridis', clabel='Not Specified'):

        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(im).set_label(clabel)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        scale_bar_length =  real_d * 1000 * pixel_size  # in pixels
        real_world_distance = real_d

        # Calculate the pixels per kilometer
        pixels_per_km = scale_bar_length / real_world_distance

        # Add the scale bar
        x_position = 50  # x position of the scale bar
        y_position = 580  # y position of the scale bar (near the bottom)

        ax.hlines(y=y_position, xmin=x_position, xmax=x_position + scale_bar_length, colors='k', linewidth=3)
        ax.text(x_position + scale_bar_length / 2, y_position - 20, f'{real_world_distance} km', color='k', ha='center')

    
    def plot_rgb(self, rgb, title, ax, real_d, pixel_size, percentile=99):
        norm_rgb = rgb / np.percentile(rgb, percentile)  # Normalize to 99th percentile for better contrast
        norm_rgb = np.clip(norm_rgb, 0, 1) 
        ax.imshow(norm_rgb)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        scale_bar_length =  real_d * 1000 * pixel_size  # in pixels
        real_world_distance = real_d

        # Calculate the pixels per kilometer
        pixels_per_km = scale_bar_length / real_world_distance

        # Add the scale bar
        x_position = 50  # x position of the scale bar
        y_position = 580  # y position of the scale bar (near the bottom)

        ax.hlines(y=y_position, xmin=x_position, xmax=x_position + scale_bar_length, colors='k', linewidth=3)
        ax.text(x_position + scale_bar_length / 2, y_position - 20, f'{real_world_distance} km', color='k', ha='center')

    def adjust_colors(self, rgb, red_factor=1.0, green_factor=1.0, blue_factor=1.0):
        rgb_adjusted = np.empty_like(rgb)
        rgb_adjusted[..., 0] = rgb[..., 0] * red_factor   # Red channel
        rgb_adjusted[..., 1] = rgb[..., 1] * green_factor # Green channel
        rgb_adjusted[..., 2] = rgb[..., 2] * blue_factor  # Blue channel
        
        return np.clip(rgb_adjusted, 0, 1)  # Ensure values are within the valid range


    def plot_tif(tif_file, ax, pts=None, box=None):

        if tif_file is not None:

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
        
        shw = ax.imshow(img, cmap='terrain', vmin=vmin, vmax=vmax)
        
        bar = plt.colorbar(shw, orientation='horizontal', shrink=0.6)
        bar.set_label('Elevation [m]', rotation=0)
        ax.set_xticks(xtick_locs)
        ax.set_xticklabels(xtick_labs)

        ax.set_yticks(ytick_locs)
        ax.set_yticklabels(ytick_labs[::-1])
        
        if xlims is not None and ylims is not None:
            out = glob2Loc(np.array((xlims, ylims)).T, tif_file)
            xlims = out[:,0]
            ylims = out[:,1]
            
            ax.set_ylim(ylims)
            ax.set_xlim(xlims)
        
        if pts is not None:
            
            y = (pts[:,1] - max_y) / range_y * -height
            x = (pts[:,0] - min_x) / range_x * width
            
            for i in range(len(x)):
                
                ax.plot(x[i], y[i], 'k'+pt_markers[i], ms=8, label=pt_labels[i])
                
            
            
        if box is not None:
            y = (box[:,1] - max_y) / range_y * -height
            x = (box[:,0] - min_x) / range_x * width
            
            ax.plot(x, y, c='r', ls='--', label='Study Area')
            
        ax.legend(loc='upper left')
            
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')

        ax.grid('both', c='black')

        fig.tight_layout()

