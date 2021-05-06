############################################
#
#   Script with definition and tools for the table view - list of loaded models
#
############################################
"""
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QAction, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpacerItem, QSizePolicy, QPushButton, QSlider, QListWidget
from PyQt5.QtGui import QIcon, QBrush, QColor 
from PyQt5.QtCore import QSize, QAbstractListModel, Qt, QModelIndex
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#import random
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT
import matplotlib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
matplotlib.use("Qt5Agg")
import numpy as np
from .. import xarray as xr
import math
import os


#   create the tableview class
class ModelListTable(QAbstractListModel):
    def __init__(self, model_list = None):
        super().__init__()
        #   set the model list to be empty 
        self.model_list = model_list or []
    
    def data(self, index, role):
        if role == Qt.DisplayRole:
            text = self.model_list[index.row()]
            
            return text
        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter
        
    def rowCount(self, index):
        return len(self.model_list)

#   create class for the QListWidget
class ListWidget(QListWidget):
    def clicked(self, sel_item):
        #   first change background of all items to white
        all_items = []
        for x in range(self.count()):
            all_items.append(self.item(x))
        for item in all_items:
            item.setBackground(Qt.white)
        sel_item.setBackground(Qt.gray)

#   define class for the plotting area
class ModelPlotCanvas(FigureCanvas):
    
    def __init__(self, parent = None, width = 12, height = 4, dpi = 300):
        #   define the figure constants etc
        #       define the colobrar
        self.cmap = plt.cm.jet
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        self.cmap = self.cmap.from_list('Custom cmap', self.cmaplist, self.cmap.N)
        #       define the bins and normalize
        self.bounds = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 20.0, 35.0]            
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)
        self.cbar_title = 'Salinity (ppt)'
        #      define the figure and all the different suplots
        self.fig = Figure(figsize = (12, 4))         
        self.ax1 = plt.subplot2grid((3, 3), (0, 1), fig = self.fig)     #   title           
        self.ax2 = plt.subplot2grid((3, 3), (1, 0), fig = self.fig)     #   y axis label           
        self.ax3 = plt.subplot2grid((3, 3), (1, 1), fig = self.fig)     #   plot itself
        self.ax4 = plt.subplot2grid((3, 3), (2, 1), fig = self.fig)     #   x axis label           
        self.ax5 = plt.subplot2grid((3, 3), (1, 2), fig = self.fig)     #   colorbar          
        #       define position of each subplot [left, bottom, width, height]
        self.ax1.set_position([0.3, 0.925, 0.35, 0.05])            
        self.ax2.set_position([0.005, 0.33, 0.02, 0.33])
        self.ax3.set_position([0.075, 0.125, 0.85, 0.775])
        self.ax4.set_position([0.3, 0.01, 0.35, 0.05])                      
        self.ax5.set_position([0.96, 0.25, 0.01, 0.5])
        #   add the title
        txt_title = self.ax1.text(0.5, 0.1, 'Salinity profile - region', fontsize = 9, fontweight = 'bold', ha = 'center')
        txt_title.set_clip_on(False)
        self.ax1.axis('off')
        #   add the y axis label 
        txt_y_axis = self.ax2.text(0.3, 0.5, 'Elavation (m bsl)', fontsize = 8, rotation = 90, va = 'center')
        txt_y_axis.set_clip_on(False)
        self.ax2.axis('off')       
        #   add the x axis label  
        txt_x_axis = self.ax4.text(0.5, 0.3, 'Distance from current coastline (km)', fontsize = 8, ha = 'center')
        txt_x_axis.set_clip_on(False)
        self.ax4.axis('off')  
        self.im = self.ax3.imshow(np.ones((10, 10)) * np.nan, aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, animated = True) #vmin = 0, vmax = 35.0,
        self.cbar = plt.colorbar(self.im, cax = self.ax5, orientation = 'vertical', pad = -0.2, shrink = 0.9, aspect = 20, spacing = 'uniform', ticks = self.bounds, boundaries = self.bounds) #cmap = self.cmap, norm = self.norm,
        self.cbar.ax.set_yticklabels([str(i) for i in self.bounds])
        self.cbar.ax.set_ylabel(self.cbar_title, rotation = 90, fontsize = 8)
        self.cbar.ax.yaxis.set_label_position('left')
        self.cbar.ax.tick_params(labelsize = 6)   
        self.canvas_to_print = FigureCanvas(self.fig)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.canvas_to_print.setSizePolicy(sizePolicy)
        self.canvas_to_print.draw()
        self.xlim = self.ax3.get_xlim()
        self.ylim = self.ax3.get_ylim()
        #return self.canvas_to_print       

    def zoom(self, xmin, xmax, ymin, ymax):
        self.ax3.set_xlim([xmin, xmax])
        self.ax3.set_ylim([ymin - 5.0, ymax + 5.0])        
        self.canvas_to_print.draw()
    
    def on_draw(self, event):
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
        
    def plotConcProfile(self, netcf_obj, ts, rcp_sc, dem_sc, xmin, xmax, ymin, ymax):
        #   transform the DEM texts into the values in th netcdf
        if dem_sc == 'GEBCO':
            dem_sc = 'gebco'
        elif dem_sc == 'TopoDEM':
            dem_sc = 'merit'
        elif dem_sc == 'CoastalDEM':
            dem_sc = 'coastal'
        #   same with the RCP scenarios
        #rcp_sc = rcp_sc.replace('.', '')
        if rcp_sc == '2.6':
            rcp_sc = '26'
        elif rcp_sc == '4.5':
            rcp_sc = '45'
        elif rcp_sc == '8.5':
            rcp_sc = '85'
            
        #   read in the x, y coordinates and the salinity
        self.x_coord, self.y_coords = netcf_obj['x'].values.tolist(), netcf_obj['y'].values.tolist()
        #   select the right RCP and DEM scenario salinity, and divide the values by 100 to get the right units
        salinity = netcf_obj.sel(time = ts).sel(dem = dem_sc).sel(rcp = rcp_sc)['salinity'].values[0] / 100.    
        salinity[salinity < 0] = np.nan
        self.ax3.clear()
        #   add the plot itself
        self.im = self.ax3.imshow(salinity, aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm,\
                        extent = (self.x_coord[0], self.x_coord[-1], min(self.y_coords)  + 5.0, max(self.y_coords) + 5.0), animated = True) #vmin = 0, vmax = 35.0,
        self.ax3.set_xlim([self.x_coord[0], self.x_coord[-1]])
        self.ax3.set_ylim([min(self.y_coords) - 5.0, max(self.y_coords) + 5.0])
        #       add the grid and coastline position - current sea level position
        x_major_ticks = np.arange(math.floor(self.x_coord[0] / 5.) * 5., math.ceil(self.x_coord[-1]), 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(self.x_coord[0] / 1.) * 1., math.ceil(self.x_coord[-1]), 1.0)
        y_major_ticks = np.arange(math.floor((min(self.y_coords) + 5.0) / 100.) * 100., max(self.y_coords) + 5.0 + 1.0, 100.0)[1:]
        y_minor_ticks = np.arange(math.floor((min(self.y_coords) + 5.0) / 50.) * 50., max(self.y_coords) + 5.0 + 1.0, 50.0) 
        self.ax3.set_xticks(x_major_ticks)
        self.ax3.set_xticks(x_minor_ticks, minor=True)
        self.ax3.set_yticks(y_major_ticks)
        self.ax3.set_yticks(y_minor_ticks, minor=True)      
        self.ax3.grid(which='minor', alpha=0.2)
        self.ax3.grid(which='major', alpha=0.5)    
        self.zoom(xmin, xmax, ymin, ymax)
        plt.tight_layout()
        #   for testing purposes these two lines below are for saving the canvas
        #canvas = FigureCanvas(fig)
        #canvas.print_figure(r'g:\_modelbuilder\test_plot.png', dpi=200)
        #self.ax3.set_xlim(self.xlim)
        #self.ax3.set_ylim(self.ylim)
        #self.canvas_to_print.draw()
        #return canvas_to_print
    
    def plotConcProfile_SEAWAT_output(self, netcf_obj, ts, xmin, xmax, ymin, ymax):
        #   read in the x, y coordinates and the salinity
        self.x_coord, self.y_coords = netcf_obj['x'].values.tolist(), netcf_obj['y'].values.tolist()
        #   select the right RCP and DEM scenario salinity, and divide the values by 100 to get the right units
        salinity = netcf_obj.sel(time = ts)['solute concentration'].values
        #salinity[salinity < 0] = np.nan
        salinity[salinity > 100] = np.nan
        salinity[salinity > 35] = 35
        self.ax3.clear()
        #   add the plot itself
        self.im = self.ax3.imshow(salinity, aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm,\
                        extent = (self.x_coord[0], self.x_coord[-1], min(self.y_coords) + 5.0, max(self.y_coords) + 5.0), animated = True) #vmin = 0, vmax = 35.0,
        self.ax3.set_xlim([self.x_coord[0], self.x_coord[-1]])
        self.ax3.set_ylim([min(self.y_coords) - 5.0, max(self.y_coords) + 5.0])
        #       add the grid and coastline position - current sea level position
        x_major_ticks = np.arange(math.floor(self.x_coord[0] / 5.) * 5., math.ceil(self.x_coord[-1]), 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(self.x_coord[0] / 1.) * 1., math.ceil(self.x_coord[-1]), 1.0)
        y_major_ticks = np.arange(math.floor((min(self.y_coords) + 5.0) / 100.) * 100., max(self.y_coords) + 5.0 + 1.0, 100.0)[1:]
        y_minor_ticks = np.arange(math.floor((min(self.y_coords) + 5.0) / 50.) * 50., max(self.y_coords) + 5.0 + 1.0, 50.0) 
        self.ax3.set_xticks(x_major_ticks)
        self.ax3.set_xticks(x_minor_ticks, minor=True)
        self.ax3.set_yticks(y_major_ticks)
        self.ax3.set_yticks(y_minor_ticks, minor=True)      
        self.ax3.grid(which='minor', alpha=0.2)
        self.ax3.grid(which='major', alpha=0.5)    
        self.zoom(xmin, xmax, ymin, ymax)
        plt.tight_layout()
        #   for testing purposes these two lines below are for saving the canvas
        #canvas = FigureCanvas(fig)
        #canvas.print_figure(r'g:\_modelbuilder\test_plot.png', dpi=200)
        #self.ax3.set_xlim(self.xlim)
        #self.ax3.set_ylim(self.ylim)
        #self.canvas_to_print.draw()
        #return canvas_to_print

    
class NavigationToolbar(NavigationToolbar2QT):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2QT.toolitems if
                 t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

"""
netcdf_dir = r'g:\_modelbuilder\swampy\data\test_model10\temp_files\test_model10_INPUT.nc'
netcdf_dir_out = r'g:\_modelbuilder\swampy\data\test_model10\temp_files\test_model10_INPUT_v2.nc'
netcdf_dir = r'g:\Water_Nexus\_A4_models\_SLR_models_OUT_files\_avg_nc_files_MERGING\0016_SRM_14.nc'
netcdf_dir = r'g:\Water_Nexus\_A4_models\_SLR_models_OUT_files\_avg_nc_files_MERGING\0016_SRM_v2_13.nc'
netcf_obj = xr_data
"""
#   define function that reads in the netcdf file
def readNetcdf(netcdf_dir):
    xr_data = xr.open_mfdataset(netcdf_dir)
    x_coord, y_coord = xr_data['x'].values.tolist(), xr_data['y'].values.tolist()
    return xr_data, x_coord, y_coord

#   define function that reads in the netcdf file
def readNetcdf_SEAWAT(netcdf_dir):
    xr_data = xr.open_mfdataset(netcdf_dir)
    x_coord, y_coord = xr_data['x'].values.tolist(), xr_data['y'].values.tolist()
    time = xr_data['time'].values.tolist()
    return xr_data, x_coord, y_coord, time

def readNetcdfSEAWATModelInput(netcdf_dir):
    xr_data = xr.open_mfdataset(netcdf_dir)
    x_coord, y_coord = xr_data['x'].values.tolist(), xr_data['y'].values.tolist()
    ibound_arr = xr_data['ibound_arr'].values
    top_elev = xr_data['top_elev_midcell'].values.tolist()
    bot_elev = xr_data['bot_elev_midcell'].values.tolist()
    inl_pt = xr_data['inl_pt'].values
    cst_pt = xr_data['cst_pt'].values
    shlf_pt = xr_data['shlf_pt'].values
    fos_pt = xr_data['fos_pt'].values
    hk_arr = xr_data['Hk_arr'].values
    vk_arr = xr_data['Vk_arr'].values
    cont_shlf_edg = xr_data['cont_shlf_edg'].values
    ic_head_arr = xr_data['ic_head_arr'].values
    ic_salinity_arr = xr_data['ic_salinty_arr'].values
    xr_data.close()
    return x_coord, y_coord, ibound_arr, top_elev, bot_elev, hk_arr, vk_arr, inl_pt, cst_pt, shlf_pt, fos_pt, cont_shlf_edg, ic_head_arr, ic_salinity_arr

def updateNetcdfGeologyArray(netcdf_dir, new_hk_arr, new_vk_arr):
    xr_data = xr.open_mfdataset(netcdf_dir)
    xr_data_new = xr_data.copy()
    x = xr_data['x'].values
    y = xr_data['y'].values
    """
    new_hk_arr = xr_data['Hk_arr'].values * 10
    new_vk_arr = xr_data['Vk_arr'].values * 10
    """    
    #   copy the rest of the variables, if this is not done for some reason xarray creates NaN arrays and lists
    for varname, da in xr_data.data_vars.items():
        if 'arr' in varname:
            xr_data_new[varname] = xr.DataArray(xr_data[varname].values, coords = {'y': y, 'x': x}, dims=['y', 'x'])    
        else:
            xr_data_new[varname] = xr.DataArray(xr_data[varname].values, coords = {'x': x}, dims=['x'])   

    #   save the new Hk and Vk arrays
    new_hk_da = xr.DataArray(new_hk_arr, coords = {'y': y, 'x': x}, dims=['y', 'x'])
    xr_data_new['Hk_arr'] = new_hk_da
    new_vk_da = xr.DataArray(new_vk_arr, coords = {'y': y, 'x': x}, dims=['y', 'x'])
    xr_data_new['Vk_arr'] = new_vk_da
    
    #   replace the netcdf file and clear from memory
    xr_data.close()  
    os.remove(netcdf_dir)
    xr_data_new.to_netcdf(netcdf_dir)
    xr_data_new.close()

def updateNetcdfICSalinityArray(netcdf_dir, new_ic_salinity_arr):
    xr_data = xr.open_mfdataset(netcdf_dir)
    xr_data_new = xr_data.copy()
    x = xr_data['x'].values
    y = xr_data['y'].values
    #   copy the rest of the variables, if this is not done for some reason xarray creates NaN arrays and lists
    for varname, da in xr_data.data_vars.items():
        if 'arr' in varname:
            xr_data_new[varname] = xr.DataArray(xr_data[varname].values, coords = {'y': y, 'x': x}, dims=['y', 'x'])    
        else:
            xr_data_new[varname] = xr.DataArray(xr_data[varname].values, coords = {'x': x}, dims=['x'])   
    #   save the new Hk and Vk arrays
    new_new_ic_salinity_arr_da = xr.DataArray(new_ic_salinity_arr, coords = {'y': y, 'x': x}, dims=['y', 'x'])
    xr_data_new['ic_salinty_arr'] = new_new_ic_salinity_arr_da
    #   replace the netcdf file and clear from memory
    xr_data.close()  
    os.remove(netcdf_dir)
    xr_data_new.to_netcdf(netcdf_dir)
    xr_data_new.close() 
    
def updateNetcdfICHeadsArray(netcdf_dir, new_ic_heads_arr):
    xr_data = xr.open_mfdataset(netcdf_dir)
    xr_data_new = xr_data.copy()
    x = xr_data['x'].values
    y = xr_data['y'].values
    #   copy the rest of the variables, if this is not done for some reason xarray creates NaN arrays and lists
    for varname, da in xr_data.data_vars.items():
        if 'arr' in varname:
            xr_data_new[varname] = xr.DataArray(xr_data[varname].values, coords = {'y': y, 'x': x}, dims=['y', 'x'])    
        else:
            xr_data_new[varname] = xr.DataArray(xr_data[varname].values, coords = {'x': x}, dims=['x'])   
    #   save the new Hk and Vk arrays
    new_new_ic_heads_arr_da = xr.DataArray(new_ic_heads_arr, coords = {'y': y, 'x': x}, dims=['y', 'x'])
    xr_data_new['ic_head_arr'] = new_new_ic_heads_arr_da
    #   replace the netcdf file and clear from memory
    xr_data.close()  
    os.remove(netcdf_dir)
    xr_data_new.to_netcdf(netcdf_dir)
    xr_data_new.close() 

""" ------------------------------------------------------------------------
                CREATE THE CLASS TO PLOT THE IBOUND ARRAY
    ------------------------------------------------------------------------"""

#   define class for the plotting area
class IboundArrayPlotCanvas(FigureCanvas):
    
    def __init__(self, parent = None, width = 12, height = 4, dpi = 300):
        #   define the figure constants etc
        #       define the colobrar
        #self.cmap = plt.cm.ocean
        self.cmap = ListedColormap(["white", "lawngreen"])
        #self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        #self.cmap = self.cmap.from_list('Custom cmap', self.cmaplist, self.cmap.N)
        #       define the bins and normalize
        self.bounds = [0.0, 0.5, 10.]            
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)
        #self.cbar_title = 'Salinity (ppt)'
        #      define the figure and all the different suplots
        self.fig = Figure(figsize = (12, 4))         
        self.ax1 = plt.subplot2grid((3, 3), (0, 1), fig = self.fig)     #   title           
        self.ax2 = plt.subplot2grid((3, 3), (1, 0), fig = self.fig)     #   y axis label           
        self.ax3 = plt.subplot2grid((3, 3), (1, 1), fig = self.fig)     #   plot itself
        self.ax4 = plt.subplot2grid((3, 3), (2, 1), fig = self.fig)     #   x axis label           
        #self.ax5 = plt.subplot2grid((3, 3), (1, 2), fig = self.fig)     #   colorbar          
        #       define position of each subplot [left, bottom, width, height]
        self.ax1.set_position([0.3, 0.925, 0.35, 0.05])            
        self.ax2.set_position([0.005, 0.33, 0.02, 0.33])
        self.ax3.set_position([0.075, 0.125, 0.9, 0.775])
        self.ax4.set_position([0.3, 0.01, 0.35, 0.05])                      
        #self.ax5.set_position([0.96, 0.25, 0.01, 0.5])
        #   add the title
        txt_title = self.ax1.text(0.5, 0.1, 'IBOUND array (active model domain)', fontsize = 9, fontweight = 'bold', ha = 'center')
        txt_title.set_clip_on(False)
        self.ax1.axis('off')
        #   add the y axis label 
        txt_y_axis = self.ax2.text(0.3, 0.5, 'Elavation (m bsl)', fontsize = 8, rotation = 90, va = 'center')
        txt_y_axis.set_clip_on(False)
        self.ax2.axis('off')       
        #   add the x axis label  
        txt_x_axis = self.ax4.text(0.5, 0.3, 'Distance from current coastline (km)', fontsize = 8, ha = 'center')
        txt_x_axis.set_clip_on(False)
        self.ax4.axis('off')  
        self.im = self.ax3.imshow(np.ones((10, 10)) * np.nan, aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, animated = True) #vmin = 0, vmax = 35.0,
        """
        self.cbar = plt.colorbar(self.im, cax = self.ax5, orientation = 'vertical', pad = -0.2, shrink = 0.9, aspect = 20, spacing = 'uniform', ticks = self.bounds, boundaries = self.bounds) #cmap = self.cmap, norm = self.norm,
        self.cbar.ax.set_yticklabels([str(i) for i in self.bounds])
        self.cbar.ax.set_ylabel(self.cbar_title, rotation = 90, fontsize = 8)
        self.cbar.ax.yaxis.set_label_position('left')
        self.cbar.ax.tick_params(labelsize = 6)   
        """
        self.canvas_to_print = FigureCanvas(self.fig)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.canvas_to_print.setSizePolicy(sizePolicy)
        self.canvas_to_print.draw()
        self.xlim = self.ax3.get_xlim()
        self.ylim = self.ax3.get_ylim()
        #return self.canvas_to_print       

    def zoom(self, xmin, xmax, ymin, ymax):
        self.ax3.set_xlim([xmin, xmax])
        self.ax3.set_ylim([ymin - 5.0, ymax + 5.0])        
        self.canvas_to_print.draw()
    
    def on_draw(self, event):
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
    
    """
    ib_arr = ibound_arr
    top_elevation_lst = top_elev_lst
    inl_bot = 50
    cst_bot = 
    shlf_bot
    fos_bot
    x_lst = x_mid_cell_lst
    y_lst = mid_lay_elev
    
    """
    
    def plotIBOUNDarray(self, ib_arr, top_elevation_lst, bot_elevation_lst, inl_pt, cst_pt, shlf_pt, fos_pt, x_lst, y_lst):
        #   get the min and max extents for plotting
        x_min = round(x_lst[0] - abs(x_lst[0] - x_lst[1]) / 2., 3)
        x_max = round(x_lst[-1] + abs(x_lst[-1] - x_lst[-2]) / 2., 3)
        y_min = y_lst[-1] - abs(y_lst[-1] - y_lst[-2]) / 2.
        y_max = y_lst[0] + abs(y_lst[0] - y_lst[1]) / 2.
        #   clear the axis if the plot needs to be redrawn 
        self.ax3.clear()
        #   add the plot itself, IndexError happens when plotting from reading a netcdf file where the array is 2D (instead of 3D which is default in seawat)
        try:
            self.im = self.ax3.imshow(ib_arr[:, 0, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        except IndexError:
            self.im = self.ax3.imshow(ib_arr[:, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        self.ax3.set_xlim([x_min, x_max])
        self.ax3.set_ylim([y_min, y_max])
        #       add the grid and coastline position - current sea level position      
        x_major_ticks = np.arange(math.floor(x_min / 5.) * 5., math.ceil(x_max / 5.) * 5., 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(x_min / 1.) * 1., math.ceil(x_max / 1.) * 1., 1.0)
        y_major_ticks = np.arange(math.ceil(y_max / 250.) * 250., math.floor(y_min / 250.) * 250. - 250., -250.0)#[1:]
        y_minor_ticks = np.arange(y_major_ticks[0], y_major_ticks[-1], -50.0)         
        
        #   plot the top elevation, bot elevation and the thickness points
        #x_lst_topo = [round(i, 3) for i in np.arange(x_min, x_max + abs(x_lst[0] - x_lst[1]), round(abs(x_lst[0] - x_lst[1]), 3)).tolist()]  
        x_lst_topo = [round(i, 3) for i in np.linspace(x_min + abs(x_lst[0] - x_lst[1]) / 2., round(x_min + abs(x_lst[0] - x_lst[1]) / 2. + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        self.ax3.plot(x_lst_topo, top_elevation_lst, color = 'black', lw = 1)
        self.ax3.plot(x_lst_topo, bot_elevation_lst, color = 'black', lw = 0.75)
        
        self.ax3.plot(inl_pt[1], inl_pt[0], 'bo')
        self.ax3.plot(cst_pt[1], cst_pt[0], 'bo')
        self.ax3.plot(shlf_pt[1], shlf_pt[0], 'bo')
        self.ax3.plot(fos_pt[1], fos_pt[0], 'bo')
        
        self.ax3.set_xticks(x_major_ticks)
        self.ax3.set_xticks(x_minor_ticks, minor=True)
        self.ax3.set_yticks(y_major_ticks)
        self.ax3.set_yticks(y_minor_ticks, minor=True)      
        self.ax3.grid(which='minor', alpha=0.2)
        self.ax3.grid(which='major', alpha=0.5)    
        self.zoom(x_min, x_max, y_major_ticks[-1], y_major_ticks[0])
        plt.tight_layout()
        #   for testing purposes these two lines below are for saving the canvas
        #canvas = FigureCanvas(fig)
        #canvas.print_figure(r'g:\_modelbuilder\test_plot.png', dpi=200)
        #self.ax3.set_xlim(self.xlim)
        #self.ax3.set_ylim(self.ylim)
        #self.canvas_to_print.draw()
        #return canvas_to_print





""" ------------------------------------------------------------------------
                CREATE THE CLASS TO PLOT THE GEOLOGY ARRAY
    ------------------------------------------------------------------------"""

#   define class for the plotting area
class GeologyArrayPlotCanvas(FigureCanvas):
    
    def __init__(self, parent = None, width = 12, height = 4, dpi = 300):
        #   define the figure constants etc
        #       define the colobrar
        self.cmap = plt.cm.viridis
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        # define the bins and normalize
        self.bounds = [0.00000001, 0.0001, 0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10., 15., 25., 50., 100., 1000.]# + list(np.arange(10.0, round(max(unique_nan) / 10) * 10, 10.))
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)  
        self.cmap.set_under('w')
        self.cbar_title = 'Hydraulic conductivity (m/d)'
        
        #      define the figure and all the different suplots
        self.fig = Figure(figsize = (12, 4))         
        self.ax1 = plt.subplot2grid((3, 3), (0, 1), fig = self.fig)     #   title           
        self.ax2 = plt.subplot2grid((3, 3), (1, 0), fig = self.fig)     #   y axis label           
        self.ax3 = plt.subplot2grid((3, 3), (1, 1), fig = self.fig)     #   plot itself
        self.ax4 = plt.subplot2grid((3, 3), (2, 1), fig = self.fig)     #   x axis label           
        self.ax5 = plt.subplot2grid((3, 3), (1, 2), fig = self.fig)     #   colorbar          
        #       define position of each subplot [left, bottom, width, height]
        self.ax1.set_position([0.3, 0.925, 0.35, 0.05])            
        self.ax2.set_position([0.005, 0.33, 0.02, 0.33])
        self.ax3.set_position([0.075, 0.125, 0.85, 0.775])
        self.ax4.set_position([0.3, 0.01, 0.35, 0.05])                      
        self.ax5.set_position([0.96, 0.25, 0.01, 0.5])
        #   add the title
        txt_title = self.ax1.text(0.5, 0.1, 'Hydraulic conductivity (m/d)', fontsize = 9, fontweight = 'bold', ha = 'center')
        txt_title.set_clip_on(False)
        self.ax1.axis('off')
        #   add the y axis label 
        txt_y_axis = self.ax2.text(0.3, 0.5, 'Elavation (m bsl)', fontsize = 8, rotation = 90, va = 'center')
        txt_y_axis.set_clip_on(False)
        self.ax2.axis('off')       
        #   add the x axis label  
        txt_x_axis = self.ax4.text(0.5, 0.3, 'Distance from current coastline (km)', fontsize = 8, ha = 'center')
        txt_x_axis.set_clip_on(False)
        self.ax4.axis('off')  
        self.im = self.ax3.imshow(np.ones((10, 10)) * np.nan, aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, animated = True) #vmin = 0, vmax = 35.0,
        
        self.cbar = plt.colorbar(self.im, cax = self.ax5, orientation = 'vertical', pad = -0.2, shrink = 0.9, aspect = 20, spacing = 'uniform', ticks = self.bounds, boundaries = self.bounds) #cmap = self.cmap, norm = self.norm,
        self.cbar.ax.set_yticklabels([str(i) for i in self.bounds])
        self.cbar.ax.set_ylabel(self.cbar_title, rotation = 90, fontsize = 8)
        self.cbar.ax.yaxis.set_label_position('left')
        self.cbar.ax.tick_params(labelsize = 6)   
        
        self.canvas_to_print = FigureCanvas(self.fig)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.canvas_to_print.setSizePolicy(sizePolicy)
        self.canvas_to_print.draw()
        self.xlim = self.ax3.get_xlim()
        self.ylim = self.ax3.get_ylim()
        #return self.canvas_to_print       
        
    def zoom(self, xmin, xmax, ymin, ymax):
        self.ax3.set_xlim([xmin, xmax])
        self.ax3.set_ylim([ymin - 5.0, ymax + 5.0])        
        self.canvas_to_print.draw()
    
    def on_draw(self, event):
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
    
    """
    ib_arr = ibound_arr
    top_elevation_lst = top_elev_lst
    inl_bot = 50
    cst_bot = 
    shlf_bot
    fos_bot
    x_lst = x_mid_cell_lst
    y_lst = mid_lay_elev
    
    """
    
    def plotGeologyArray(self, geology_arr, top_elevation_lst, x_lst, y_lst):
        #   get the min and max extents for plotting
        x_min = round(x_lst[0] - abs(x_lst[0] - x_lst[1]) / 2., 3)
        x_max = round(x_lst[-1] + abs(x_lst[-1] - x_lst[-2]) / 2., 3)
        y_min = y_lst[-1] - abs(y_lst[-1] - y_lst[-2]) / 2.
        y_max = y_lst[0] + abs(y_lst[0] - y_lst[1]) / 2.
        #   clear the axis if the plot needs to be redrawn 
        self.ax3.clear()
        #   add the plot itself, IndexError happens when plotting from reading a netcdf file where the array is 2D (instead of 3D which is default in seawat)
        try:
            self.im = self.ax3.imshow(geology_arr[:, 0, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        except IndexError:
            self.im = self.ax3.imshow(geology_arr[:, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        self.ax3.set_xlim([x_min, x_max])
        self.ax3.set_ylim([y_min, y_max])
        #       add the grid and coastline position - current sea level position      
        x_major_ticks = np.arange(math.floor(x_min / 5.) * 5., math.ceil(x_max / 5.) * 5., 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(x_min / 1.) * 1., math.ceil(x_max / 1.) * 1., 1.0)
        y_major_ticks = np.arange(math.ceil(y_max / 250.) * 250., math.floor(y_min / 250.) * 250. - 250., -250.0)#[1:]
        y_minor_ticks = np.arange(y_major_ticks[0], y_major_ticks[-1], -50.0)         
        
        #   plot the top elevation, bot elevation and the thickness points
        #x_lst_topo = [round(i, 3) for i in np.arange(x_min, x_max + abs(x_lst[0] - x_lst[1]), round(abs(x_lst[0] - x_lst[1]), 3)).tolist()]  
        #x_lst_topo = [round(i, 3) for i in np.linspace(x_min, round(x_min + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        x_lst_topo = [round(i, 3) for i in np.linspace(x_min + abs(x_lst[0] - x_lst[1]) / 2., round(x_min + abs(x_lst[0] - x_lst[1]) / 2. + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        self.ax3.plot(x_lst_topo, top_elevation_lst, color = 'black', lw = 1)
        self.ax3.set_xticks(x_major_ticks)
        self.ax3.set_xticks(x_minor_ticks, minor=True)
        self.ax3.set_yticks(y_major_ticks)
        self.ax3.set_yticks(y_minor_ticks, minor=True)      
        self.ax3.grid(which='minor', alpha=0.2)
        self.ax3.grid(which='major', alpha=0.5)    
        self.zoom(x_min, x_max, y_major_ticks[-1], y_major_ticks[0])
        plt.tight_layout()
        #   for testing purposes these two lines below are for saving the canvas
        #canvas = FigureCanvas(fig)
        #canvas.print_figure(r'g:\_modelbuilder\test_plot.png', dpi=200)
        #self.ax3.set_xlim(self.xlim)
        #self.ax3.set_ylim(self.ylim)
        #self.canvas_to_print.draw()
        #return canvas_to_print




        """
        #   define the figure constants etc
        #       define the colobrar
        #cmap = plt.cm.ocean
        cmap = ListedColormap(["white", "lawngreen"])
        #cmaplist = [cmap(i) for i in range(cmap.N)]
        #cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
        #       define the bins and normalize
        bounds = [0.0, 0.5, 10.]            
        norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
        #cbar_title = 'Salinity (ppt)'
        #      define the figure and all the different suplots
        fig = Figure(figsize = (12, 4))         
        ax1 = plt.subplot2grid((3, 3), (0, 1), fig = fig)     #   title           
        ax2 = plt.subplot2grid((3, 3), (1, 0), fig = fig)     #   y axis label           
        ax3 = plt.subplot2grid((3, 3), (1, 1), fig = fig)     #   plot itself
        ax4 = plt.subplot2grid((3, 3), (2, 1), fig = fig)     #   x axis label           
        #ax5 = plt.subplot2grid((3, 3), (1, 2), fig = fig)     #   colorbar          
        #       define position of each subplot [left, bottom, width, height]
        ax1.set_position([0.3, 0.925, 0.35, 0.05])            
        ax2.set_position([0.005, 0.33, 0.02, 0.33])
        ax3.set_position([0.075, 0.125, 0.9, 0.775])
        ax4.set_position([0.3, 0.01, 0.35, 0.05])                      
        #ax5.set_position([0.96, 0.25, 0.01, 0.5])
        #   add the title
        txt_title = ax1.text(0.5, 0.1, 'IBOUND array (active model domain)', fontsize = 9, fontweight = 'bold', ha = 'center')
        txt_title.set_clip_on(False)
        ax1.axis('off')
        #   add the y axis label 
        txt_y_axis = ax2.text(0.3, 0.5, 'Elavation (m bsl)', fontsize = 8, rotation = 90, va = 'center')
        txt_y_axis.set_clip_on(False)
        ax2.axis('off')       
        #   add the x axis label  
        txt_x_axis = ax4.text(0.5, 0.3, 'Distance from current coastline (km)', fontsize = 8, ha = 'center')
        txt_x_axis.set_clip_on(False)
        ax4.axis('off')  
        im = ax3.imshow(ib_arr[:, 0, :], aspect = 'auto', interpolation = 'none', cmap = cmap, norm = norm, extent = (x_min, x_max, y_min, y_max), animated = True)

        ax3.plot(np.arange(x_min, x_max, abs(x_lst[0] - x_lst[1])), top_elevation_lst, color = 'black', lw = 1)
        ax3.plot(x_lst[0], y_max - inl_bot, 'bo', ms = 1.)
        ax3.plot(0.,  - cst_bot, 'bo', ms = 1.)

        ax3.set_xlim([x_min, x_max])
        ax3.set_ylim([y_min, y_max])
        #       add the grid and coastline position - current sea level position      
        x_major_ticks = np.arange(math.floor(x_min / 5.) * 5., math.ceil(x_max / 5.) * 5., 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(x_min / 1.) * 1., math.ceil(x_max / 1.) * 1., 1.0)
        y_major_ticks = np.arange(math.floor(y_max / 250.) * 250., math.ceil(y_min / 250.) * 250., -250.0)#[1:]
        y_minor_ticks = np.arange(y_major_ticks[0] + 100, y_major_ticks[-1] - 100., -50.0)         
        
        #x_major_ticks = np.arange(math.floor(x_lst[0] / 5.) * 5., math.ceil(x_lst[-1]), 5.0)[1:]
        #x_minor_ticks = np.arange(math.floor(x_lst[0] / 1.) * 1., math.ceil(x_lst[-1]), 1.0)
        #y_major_ticks = np.arange(math.floor(y_min / 100.) * 100., y_max, 100.0)[1:]
        #y_minor_ticks = np.arange(math.floor(y_min / 50.) * 50., y_max, 50.0) 
        
        ax3.set_xticks(x_major_ticks)
        ax3.set_xticks(x_minor_ticks, minor=True)
        ax3.set_yticks(y_major_ticks)
        ax3.set_yticks(y_minor_ticks, minor=True)      
        ax3.grid(which='minor', alpha=0.2)
        ax3.grid(which='major', alpha=0.5)    
        
        ax3.set_xlim([x_min, x_max])
        ax3.set_ylim([y_major_ticks[-1] - 100., y_major_ticks[0] + 100])          
        
        #zoom(x_min, x_max, y_major_ticks[-1] - 100., y_major_ticks[0] + 100.)
        
        plt.tight_layout()
        canvas_to_print = FigureCanvas(fig)
        #sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        #sizePolicy.setHorizontalStretch(0)
        #sizePolicy.setVerticalStretch(0)
        #canvas_to_print.setSizePolicy(sizePolicy)
        #canvas_to_print.draw()
        
        canvas_to_print.print_figure(r'g:\_modelbuilder\swampy\data\test_model\temp_files\test_plot.png', dpi=200) 
        """

""" ------------------------------------------------------------------------
                CREATE THE CLASS TO PLOT THE HEAD ELEVATION ARRAY
    ------------------------------------------------------------------------"""

#   define class for the plotting area
class ICArrayPlotCanvas(FigureCanvas):
    
    def __init__(self, parent = None, width = 12, height = 4, dpi = 300):
        #   define the figure constants etc
        #       define the colobrar
        self.cmap = plt.cm.winter
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        # define the bins and normalize
        self.bounds = [-1000., -750., -500., -250., -100., -50., -25, -10, -5, 0, 5, 10, 25, 50., 250.]# + list(np.arange(10.0, round(max(unique_nan) / 10) * 10, 10.))
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)  
        self.cmap.set_under('w')
        self.cbar_title = 'Head elevation (m)'
        
        #      define the figure and all the different suplots
        self.fig = Figure(figsize = (12, 4))         
        self.ax1 = plt.subplot2grid((3, 3), (0, 1), fig = self.fig)     #   title           
        self.ax2 = plt.subplot2grid((3, 3), (1, 0), fig = self.fig)     #   y axis label           
        self.ax3 = plt.subplot2grid((3, 3), (1, 1), fig = self.fig)     #   plot itself
        self.ax4 = plt.subplot2grid((3, 3), (2, 1), fig = self.fig)     #   x axis label           
        self.ax5 = plt.subplot2grid((3, 3), (1, 2), fig = self.fig)     #   colorbar          
        #       define position of each subplot [left, bottom, width, height]
        self.ax1.set_position([0.3, 0.925, 0.35, 0.05])            
        self.ax2.set_position([0.005, 0.33, 0.02, 0.33])
        self.ax3.set_position([0.075, 0.125, 0.85, 0.775])
        self.ax4.set_position([0.3, 0.01, 0.35, 0.05])                      
        self.ax5.set_position([0.96, 0.25, 0.01, 0.5])
        #   add the title
        txt_title = self.ax1.text(0.5, 0.1, 'Head elevation (m)', fontsize = 9, fontweight = 'bold', ha = 'center')
        txt_title.set_clip_on(False)
        self.ax1.axis('off')
        #   add the y axis label 
        txt_y_axis = self.ax2.text(0.3, 0.5, 'Elavation (m bsl)', fontsize = 8, rotation = 90, va = 'center')
        txt_y_axis.set_clip_on(False)
        self.ax2.axis('off')       
        #   add the x axis label  
        txt_x_axis = self.ax4.text(0.5, 0.3, 'Distance from current coastline (km)', fontsize = 8, ha = 'center')
        txt_x_axis.set_clip_on(False)
        self.ax4.axis('off')  
        self.im = self.ax3.imshow(np.ones((10, 10)) * np.nan, aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, animated = True) #vmin = 0, vmax = 35.0,
        
        self.cbar = plt.colorbar(self.im, cax = self.ax5, orientation = 'vertical', pad = -0.2, shrink = 0.9, aspect = 20, spacing = 'uniform', ticks = self.bounds, boundaries = self.bounds) #cmap = self.cmap, norm = self.norm,
        self.cbar.ax.set_yticklabels([str(i) for i in self.bounds])
        self.cbar.ax.set_ylabel(self.cbar_title, rotation = 90, fontsize = 8)
        self.cbar.ax.yaxis.set_label_position('left')
        self.cbar.ax.tick_params(labelsize = 6)   
        
        self.canvas_to_print = FigureCanvas(self.fig)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.canvas_to_print.setSizePolicy(sizePolicy)
        self.canvas_to_print.draw()
        self.xlim = self.ax3.get_xlim()
        self.ylim = self.ax3.get_ylim()
        #return self.canvas_to_print       
        
    def zoom(self, xmin, xmax, ymin, ymax):
        self.ax3.set_xlim([xmin, xmax])
        self.ax3.set_ylim([ymin - 5.0, ymax + 5.0])        
        self.canvas_to_print.draw()
    
    def on_draw(self, event):
        self.xlim = self.axes.get_xlim()
        self.ylim = self.axes.get_ylim()
    
    """
    ib_arr = ibound_arr
    top_elevation_lst = top_elev_lst
    inl_bot = 50
    cst_bot = 
    shlf_bot
    fos_bot
    x_lst = x_mid_cell_lst
    y_lst = mid_lay_elev
    
    """
    
    def plotHeadElevationArray(self, head_elev_arr, top_elevation_lst, x_lst, y_lst):

        #       define the colobrar
        self.ax5.clear()
        self.cmap = plt.cm.winter
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        # define the bins and normalize
        self.bounds = [-10000., -1000., -750., -500., -250., -100., -50., -25, -10, -5, 0, 5, 10, 25, 50., 250.]# + list(np.arange(10.0, round(max(unique_nan) / 10) * 10, 10.))
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)  
        self.cmap.set_under('w')
        self.cbar_title = 'Head elevation (m)'

        #   get the min and max extents for plotting
        x_min = round(x_lst[0] - abs(x_lst[0] - x_lst[1]) / 2., 3)
        x_max = round(x_lst[-1] + abs(x_lst[-1] - x_lst[-2]) / 2., 3)
        y_min = y_lst[-1] - abs(y_lst[-1] - y_lst[-2]) / 2.
        y_max = y_lst[0] + abs(y_lst[0] - y_lst[1]) / 2.
        #   clear the axis if the plot needs to be redrawn 
        self.ax3.clear()
        #   add the plot itself, IndexError happens when plotting from reading a netcdf file where the array is 2D (instead of 3D which is default in seawat)
        #try:
        #    self.im = self.ax3.imshow(head_elev_arr[:, 0, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        #except IndexError:
        self.im = self.ax3.imshow(head_elev_arr[:, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        self.ax3.set_xlim([x_min, x_max])
        self.ax3.set_ylim([y_min, y_max])
        #       add the grid and coastline position - current sea level position      
        x_major_ticks = np.arange(math.floor(x_min / 5.) * 5., math.ceil(x_max / 5.) * 5., 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(x_min / 1.) * 1., math.ceil(x_max / 1.) * 1., 1.0)
        y_major_ticks = np.arange(math.ceil(y_max / 250.) * 250., math.floor(y_min / 250.) * 250. - 250., -250.0)#[1:]
        y_minor_ticks = np.arange(y_major_ticks[0], y_major_ticks[-1], -50.0)         

        self.cbar = plt.colorbar(self.im, cax = self.ax5, orientation = 'vertical', pad = -0.2, shrink = 0.9, aspect = 20, spacing = 'uniform', ticks = self.bounds, boundaries = self.bounds) #cmap = self.cmap, norm = self.norm,
        self.cbar.ax.set_yticklabels([str(i) for i in self.bounds])
        self.cbar.ax.set_ylabel(self.cbar_title, rotation = 90, fontsize = 8)
        self.cbar.ax.yaxis.set_label_position('left')
        self.cbar.ax.tick_params(labelsize = 6)   
        
        #   plot the top elevation, bot elevation and the thickness points
        #x_lst_topo = [round(i, 3) for i in np.arange(x_min, x_max + abs(x_lst[0] - x_lst[1]), round(abs(x_lst[0] - x_lst[1]), 3)).tolist()]  
        #x_lst_topo = [round(i, 3) for i in np.linspace(x_min, round(x_min + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        #x_lst_topo = [round(i, 3) for i in np.linspace(x_min + abs(x_lst[0] - x_lst[1]) / 2., round(x_min + abs(x_lst[0] - x_lst[1]) / 2. + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        #self.ax3.plot(x_lst_topo, top_elevation_lst, color = 'black', lw = 1)
        self.ax3.set_xticks(x_major_ticks)
        self.ax3.set_xticks(x_minor_ticks, minor=True)
        self.ax3.set_yticks(y_major_ticks)
        self.ax3.set_yticks(y_minor_ticks, minor=True)      
        self.ax3.grid(which='minor', alpha=0.2)
        self.ax3.grid(which='major', alpha=0.5)    
        self.zoom(x_min, x_max, y_major_ticks[-1], y_major_ticks[0])
        plt.tight_layout()
        #   for testing purposes these two lines below are for saving the canvas
        #canvas = FigureCanvas(fig)
        #canvas.print_figure(r'g:\_modelbuilder\test_plot.png', dpi=200)
        #self.ax3.set_xlim(self.xlim)
        #self.ax3.set_ylim(self.ylim)
        #self.canvas_to_print.draw()
        #return canvas_to_print

    def plotSalinityArray(self, conc_arr, top_elevation_lst, x_lst, y_lst):
        #       define the colobrar
        self.ax5.clear()
        self.cmap = plt.cm.jet
        self.cmaplist = [self.cmap(i) for i in range(self.cmap.N)]
        # define the bins and normalize
        self.bounds = [0.0, 0.1, 0.25, 0.5, 1., 2.5, 5., 10, 15., 20., 35.]# + list(np.arange(10.0, round(max(unique_nan) / 10) * 10, 10.))
        self.norm = matplotlib.colors.BoundaryNorm(self.bounds, self.cmap.N)  
        self.cmap.set_under('w')
        self.cbar_title = 'Salinity TDS (g/l)'

        #   get the min and max extents for plotting
        x_min = round(x_lst[0] - abs(x_lst[0] - x_lst[1]) / 2., 3)
        x_max = round(x_lst[-1] + abs(x_lst[-1] - x_lst[-2]) / 2., 3)
        y_min = y_lst[-1] - abs(y_lst[-1] - y_lst[-2]) / 2.
        y_max = y_lst[0] + abs(y_lst[0] - y_lst[1]) / 2.
        #   clear the axis if the plot needs to be redrawn 
        self.ax3.clear()
        #   add the plot itself, IndexError happens when plotting from reading a netcdf file where the array is 2D (instead of 3D which is default in seawat)
        #try:
        #    self.im = self.ax3.imshow(conc_arr[:, 0, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        #except IndexError:
        self.im = self.ax3.imshow(conc_arr[:, :], aspect = 'auto', interpolation = 'none', cmap = self.cmap, norm = self.norm, extent = (x_min, x_max, y_min, y_max), animated = True) #vmin = 0, vmax = 35.0,
        self.ax3.set_xlim([x_min, x_max])
        self.ax3.set_ylim([y_min, y_max])
        #       add the grid and coastline position - current sea level position      
        x_major_ticks = np.arange(math.floor(x_min / 5.) * 5., math.ceil(x_max / 5.) * 5., 5.0)[1:]
        x_minor_ticks = np.arange(math.floor(x_min / 1.) * 1., math.ceil(x_max / 1.) * 1., 1.0)
        y_major_ticks = np.arange(math.ceil(y_max / 250.) * 250., math.floor(y_min / 250.) * 250. - 250., -250.0)#[1:]
        y_minor_ticks = np.arange(y_major_ticks[0], y_major_ticks[-1], -50.0)         

        self.cbar = plt.colorbar(self.im, cax = self.ax5, orientation = 'vertical', pad = -0.2, shrink = 0.9, aspect = 20, spacing = 'uniform', ticks = self.bounds, boundaries = self.bounds) #cmap = self.cmap, norm = self.norm,
        self.cbar.ax.set_yticklabels([str(i) for i in self.bounds])
        self.cbar.ax.set_ylabel(self.cbar_title, rotation = 90, fontsize = 8)
        self.cbar.ax.yaxis.set_label_position('left')
        self.cbar.ax.tick_params(labelsize = 6)   
        
        #   plot the top elevation, bot elevation and the thickness points
        #x_lst_topo = [round(i, 3) for i in np.arange(x_min, x_max + abs(x_lst[0] - x_lst[1]), round(abs(x_lst[0] - x_lst[1]), 3)).tolist()]  
        #x_lst_topo = [round(i, 3) for i in np.linspace(x_min, round(x_min + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        #x_lst_topo = [round(i, 3) for i in np.linspace(x_min + abs(x_lst[0] - x_lst[1]) / 2., round(x_min + abs(x_lst[0] - x_lst[1]) / 2. + (len(top_elevation_lst) - 1) * abs(x_lst[0] - x_lst[1]), 3), len(top_elevation_lst), endpoint=True)]    
        #self.ax3.plot(x_lst_topo, top_elevation_lst, color = 'black', lw = 1)
        self.ax3.set_xticks(x_major_ticks)
        self.ax3.set_xticks(x_minor_ticks, minor=True)
        self.ax3.set_yticks(y_major_ticks)
        self.ax3.set_yticks(y_minor_ticks, minor=True)      
        self.ax3.grid(which='minor', alpha=0.2)
        self.ax3.grid(which='major', alpha=0.5)    
        self.zoom(x_min, x_max, y_major_ticks[-1], y_major_ticks[0])
        plt.tight_layout()
        #   for testing purposes these two lines below are for saving the canvas
        #canvas = FigureCanvas(fig)
        #canvas.print_figure(r'g:\_modelbuilder\test_plot.png', dpi=200)
        #self.ax3.set_xlim(self.xlim)
        #self.ax3.set_ylim(self.ylim)
        #self.canvas_to_print.draw()
        #return canvas_to_print


