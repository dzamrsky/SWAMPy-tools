# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 11:11:06 2020

@author: daniel
"""

#from .model_build_tools import create_ibound_arr, find_lonely_cells, create_hk_arr, start_head_conc, create_ghb_input,\
#create_rch_input, create_drn_input, create_btn_input, create_oc_input, create_riv_input, find_FOS, find_shlf_edge
import pandas as pd
import numpy as np
import os
import math
from .. import xarray as xr
import statistics
from statistics import StatisticsError
#from qgis.core import QgsTask
import flopy
import flopy.utils.binaryfile as bf
import random
from time import time
from qgis.core import QgsRasterLayer, QgsPointXY, QgsProcessingContext, QgsTaskManager, QgsTask, QgsProcessingAlgRunnerTask, Qgis, QgsProcessingFeedback, QgsApplication, QgsMessageLog
import shutil

MESSAGE_CATEGORY = 'TaskFromFunction'

"""
name = r'test_model17'
dC =0.100
dL = 0.010
x_st = 21.20
y_bot_st = 100
y_bot_cst = 250 
y_bot_shlf = 500
y_bot_fos = 50
csv_in = r'g:\_modelbuilder\swampy\data\test_model17\temp_files\cs_points.csv '

df = pd.read_csv(csv_in)
top_elev = df['DEM'].values.tolist()        

delcol = dC
dellay = dL

x_st = x_st * -1
y_bot_st = top_elev[0] - y_bot_st

x_coast = 0.
y_coast_top = top_elev[int(abs(x_st / delcol))]
y_bot_cst = y_coast_top - y_bot_cst

#   then we need to find the continental shelf and foot of cont. slope X coordinates 
sb_pt = find_shlf_edge(top_elev)
x_shelf_edge = round(x_st + (sb_pt[0] * delcol), 3)
y_shelf_top = top_elev[sb_pt[0]]
y_bot_shlf = y_shelf_top - y_bot_shlf

#fos_pt = find_FOS(top_elev)[1]
fos_pt = len(top_elev) - 1, top_elev[-1]
x_fos = round(x_st + (len(top_elev) * delcol), 3)
y_fos_top = top_elev[-1]
y_bot_fos = y_fos_top - y_bot_fos
"""  
 

"""
-- Function to build the IBOUND array based on geometry input from SWAMPY

del_col = delcol
del_lay = dellay
x_start = x_st
y_start_top = top_elev[0]
y_start_bot = y_bot_st
x_coast = x_coast
y_coast_top = y_coast_top
y_coast_bot = y_bot_cst
x_shelf_edge = x_shelf_edge
y_shelf_edge_top = sb_pt[1]
y_shelf_edge_bot = y_bot_shlf
x_foot_of_slope = x_fos
y_foot_of_slope_top = fos_pt[1]
y_foot_of_slope_bot = y_bot_fos
top_elev_lst = top_elev
"""

def create_ibound_arr(del_col, del_lay, x_start, y_start_top, y_start_bot, x_coast, y_coast_top, y_coast_bot, x_shelf_edge,
                      y_shelf_edge_top, y_shelf_edge_bot, x_foot_of_slope, y_foot_of_slope_top, y_foot_of_slope_bot, top_elev_lst):

    if x_start > 0.:
        print('Inland start of model domain is > coast')
        return  #   makes the function stop

    del_col_m = del_col * 1000.
    del_lay_m = del_lay * 1000.
    
    #   create the first version of the array based on the number of layers and columns
    #       1) get the number of columns based on the total extent of the domain in the x direction
    x_len_m = (abs(x_start) + abs(x_foot_of_slope)) * 1000         #   total distance in meters
    ncols = int(x_len_m / del_col_m)      #   total distance in columns (has to be an integer number so you can create a loop later!)
    #       2) calculate the number of layers based on the extent in the y direction - select the overall top and bottom elevations of model domain
    y_top_all = math.ceil(max(top_elev_lst) / del_lay_m) * del_lay_m
    y_bot_all = math.floor(min(y_start_bot, y_start_bot, y_start_bot, y_foot_of_slope_bot, min(top_elev_lst)) / del_lay_m) * del_lay_m
    y_len_m = y_top_all - y_bot_all
    nlays = int(y_len_m / del_lay_m)
    ## y_len_m = abs(y_start_top) + abs(y_foot_of_slope_bot)
    ## nlays = int(y_len_m / del_lay_m)
    #       3) create an array with the dimensions calculated above, fill with zeros first
    ibound_arr = np.zeros((nlays, 1, ncols), dtype = np.int16)
    
    #   define a list with mid_layer elevations
    mid_lay_elev = np.arange(y_top_all - (0.5 * del_lay_m), y_bot_all - (0.5 * del_lay_m), -del_lay_m)
    ## mid_lay_elev = np.arange(y_start_top - (0.5 * del_lay_m), y_foot_of_slope_bot + (0.5 * del_lay_m), -del_lay_m)
    
    #   calculate the amount columns that span across the distance between the starting and ending point           
    n_cols_inland = int(((abs(x_start) + 0.0) * 1000.) / del_col_m)
    
    #   calculate the drop in elevations for the top and bottom
    ## diff_y_inland_top = abs((y_start_top - y_coast_top)) / n_cols_inland
    diff_y_inland_bot = abs((y_coast_bot - y_start_bot)) / n_cols_inland  
    
    #   create lists with elevation at each column between the inland boundary and the coast
    ## mid_col_top_elev_inland = np.linspace(y_start_top - (0.5 * diff_y_inland_top), y_coast_top + (0.5 * diff_y_inland_top), n_cols_inland).tolist()
    mid_col_bot_elev_inland = np.linspace(y_start_bot - (0.5 * diff_y_inland_bot), y_coast_bot + (0.5 * diff_y_inland_bot), n_cols_inland).tolist() 

    #   round the numbers in the list to integers
    ## inland_top_elev_lst = [round(i, 1) for i in mid_col_top_elev_inland]
    inland_bot_elev_lst = [round(i, 1) for i in mid_col_bot_elev_inland]       
    
    #   create bot elevation list for the whole model domain
    inland_bot_elev_all_lst = inland_bot_elev_lst
    """
    inland_bot_elev_all_lst = []
    for i in range(n_cols_inland):
        inland_bot_elev_all_lst.append(top_elev_lst[i] - inland_bot_elev_lst[i])
    """
    #   first setup the active model domain in the inland part 
    for a in range(0, n_cols_inland):   
        #   find the top and bottom elevation at the respective column
        ## top_elev_col = inland_top_elev_lst[a]
        top_elev_col = top_elev_lst[a]
        bot_elev_col = inland_bot_elev_lst[a]
        #   loop through the layers 
        for b in range(len(mid_lay_elev)):
            
            #   check if the mid_lay_elev is between the top and bot elevation of the column
            mid_lay = mid_lay_elev[b]
            
            if mid_lay < top_elev_col and mid_lay > bot_elev_col:
                #print(top_elev_col, bot_elev_col, mid_lay, a, b) 
                #   change the cells to be active in the ibound_arr
                ibound_arr[b, 0, a] = 1
   
    " ******** COAST PART ********* "
    #   calculate the amount columns that span across the distance between the starting and ending point           
    n_cols_coast = int(abs((x_coast + x_shelf_edge) * 1000.) / del_col_m)        
    
    #   calculate the drop in elevations for the top and bottom
    ## diff_y_coast_top = abs((y_coast_top - y_shelf_edge_top)) / n_cols_coast   
    # y_shelf_edge_bot condition
    if y_shelf_edge_bot == None:
        diff_y_coast_bot = abs((y_coast_bot - y_foot_of_slope_bot)) / n_cols_coast
        #   you need to specify the value of the y_shelf_edge_bot because you use it in the next step and it cant calculate with None value
        y_shelf_edge_bot = y_coast_bot - (y_coast_bot - y_foot_of_slope_bot) / 2
    else:
        diff_y_coast_bot = abs((y_coast_bot - y_shelf_edge_bot)) / n_cols_coast        
    
    #   create lists with elevation at each column between the inland boundary and the coast
    ## mid_col_top_elev_coast = np.linspace(y_coast_top - (0.5 * diff_y_coast_top), y_shelf_edge_top + (0.5 * diff_y_coast_top), n_cols_coast).tolist() 
    #mid_col_bot_elev_coast = np.linspace(y_coast_bot - (0.5 * diff_y_coast_bot), y_shelf_edge_bot + (0.5 * diff_y_coast_bot), n_cols_coast).tolist()
    mid_col_bot_elev_coast = np.linspace(inland_bot_elev_all_lst[-1] - (0.5 * diff_y_coast_bot), y_shelf_edge_bot + (0.5 * diff_y_coast_bot), n_cols_coast).tolist() 
    inland_bot_elev_all_lst = inland_bot_elev_all_lst + mid_col_bot_elev_coast
    
    #   round the numbers in the list to integers
    ## coast_top_elev_lst = [round(i, 1) for i in mid_col_top_elev_coast]        
    coast_bot_elev_lst = [round(i, 1) for i in mid_col_bot_elev_coast]       
    
    # 
    for a in range(n_cols_inland, n_cols_inland + n_cols_coast):   
        #   find the top and bottom elevation at the respective column
        ##top_elev_col = coast_top_elev_lst[a]
        top_elev_col = top_elev_lst[a]
        bot_elev_col = coast_bot_elev_lst[a - n_cols_inland]
        #   loop through the layers 
        for b in range(len(mid_lay_elev)):
            
            #   check if the mid_lay_elev is between the top and bot elevation of the column
            mid_lay = mid_lay_elev[b]
            
            if mid_lay < top_elev_col and mid_lay > bot_elev_col:
                #print top_elev_col, bot_elev_col, mid_lay, a, b 
                #   change the cells to be active in the ibound_arr
                ibound_arr[b, 0, a] = 1        
   

    " ******** SHELF PART ********* "
    #   calculate the amount columns that span across the distance between the starting and ending point           
    #n_cols_shelf = int(((x_foot_of_slope - abs(x_shelf_edge)) * 1000.) / del_col_m)
    n_cols_shelf = ibound_arr.shape[-1] - n_cols_inland - n_cols_coast
    
    #   calculate the drop in elevations for the top and bottom
    ## diff_y_shelf_top = abs((y_shelf_edge_top - y_foot_of_slope_top)) / n_cols_shelf
    
    # y_shelf_edge_bot condition
    if y_shelf_edge_bot == None:
        diff_y_shelf_bot = abs((y_coast_bot - y_foot_of_slope_bot)) / n_cols_coast
    else:
        diff_y_shelf_bot = abs((y_shelf_edge_bot - y_foot_of_slope_bot)) / n_cols_coast            
    
    #   create lists with elevation at each column between the inland boundary and the coast
    ## mid_col_top_elev_shelf = np.linspace(y_shelf_edge_top - (0.5 * diff_y_shelf_top), y_foot_of_slope_top + (0.5 * diff_y_shelf_top), n_cols_shelf).tolist() 
    mid_col_bot_elev_shelf = np.linspace(y_shelf_edge_bot - (0.5 * diff_y_shelf_bot), y_foot_of_slope_bot + (0.5 * diff_y_shelf_bot), n_cols_shelf).tolist() 
    inland_bot_elev_all_lst = inland_bot_elev_all_lst + mid_col_bot_elev_shelf

    #   round the numbers in the list to integers
    ## shelf_top_elev_lst = [round(i, 1) for i in mid_col_top_elev_shelf]        
    shelf_bot_elev_lst = [round(i, 1) for i in mid_col_bot_elev_shelf]       
    
    #   
    for a in range(n_cols_inland + n_cols_coast, len(top_elev_lst) - 1):   
        #   find the top and bottom elevation at the respective column
        ## top_elev_col = shelf_top_elev_lst[a]
        top_elev_col = top_elev_lst[a]
        bot_elev_col = shelf_bot_elev_lst[a - n_cols_inland - n_cols_coast]
        #   loop through the layers 
        for b in range(len(mid_lay_elev)):
            
            #   check if the mid_lay_elev is between the top and bot elevation of the column
            mid_lay = mid_lay_elev[b]
            
            if mid_lay < top_elev_col and mid_lay > bot_elev_col:
                #print top_elev_col, bot_elev_col, mid_lay, a, b 
                #   change the cells to be active in the ibound_arr
                ibound_arr[b, 0, a] = 1
    
    ## top_elev = inland_top_elev_lst + coast_top_elev_lst + shelf_top_elev_lst
    bot_elev = inland_bot_elev_lst + coast_bot_elev_lst + shelf_bot_elev_lst
    x_mid_cell_lst = [round(i, 3) for i in np.arange((x_start) + del_col / 2., (x_start) + ibound_arr.shape[-1] * del_col + del_col / 2., del_col).tolist()]

    bot_elev_out = [round(i, 2) for i in inland_bot_elev_all_lst]

    #   find the lonely cells and adapt the ibound array
    left, top, right, bot = 0, 0, -1, -1
    
    # trim the ibound array and other lists 
    ibound_arr = ibound_arr[top : bot, :, left : right]
    bot_elev = bot_elev[left : right]
    bot_elev_out = bot_elev_out[left : right]
    mid_lay_elev = mid_lay_elev[top : bot].tolist()
    x_mid_cell_lst = x_mid_cell_lst[left : right]
    top_elev_lst = top_elev_lst[left : right]
    
    ## return ibound_arr, top_elev, bot_elev
    return ibound_arr, bot_elev_out, mid_lay_elev, x_mid_cell_lst, top_elev_lst

"""
ibound_array = ibound_arr
bot_elev_lst = bot_elev
y_mid_cell_lst = mid_lay_elev
x_mid_cell_lst = x_mid_cell_lst
top_elev_lst = top_elev
"""

#   function to find shelf
def find_shlf_edge(topo_lst, shlf_edge_depth = -175):
    #   find the first cell with elevation higher than the shlf_edge_depth from inversed list
    last_shelf = next(x[0] for x in enumerate(topo_lst[::-1]) if x[1] > shlf_edge_depth)
    shlf_idx = len(topo_lst) - last_shelf - 1
    return shlf_idx, topo_lst[shlf_idx]


#   loading data from raster input datasets into the csv file
def loadRCHvalsFromDir(csv_dir, raster_dir):
    #   load in the csv file and the raster input file, then loop through the coastal points and extract value from the raster
    df_in = pd.read_csv(csv_dir)
    rlayer = QgsRasterLayer(raster_dir, 'rch_in')        
    val_lst = []
    for index, row in df_in.iterrows():
        x, y = row['x_wgs84'], row['y_wgs84']
        val, res = rlayer.dataProvider().sample(QgsPointXY(x, y), 1)      
        val_lst.append(round(val, 8))
    return val_lst


#   define the SEAWAT model class
class CS_model(object):
    
    #   create the object with the necessary input
    def __init__(self, name, dC, dL, x_st, y_bot_st, y_bot_cst, y_bot_shlf, y_bot_fos, csv_in):

        self.name = name
        self.csv_in = csv_in
        df = pd.read_csv(self.csv_in)
        self.top_elev = df['DEM'].values.tolist()        

        self.delcol = dC
        self.dellay = dL
        if x_st >= 0.:
            self.x_st = x_st * -1
        else:
            self.x_st = x_st
        self.y_bot_st = self.top_elev[0] - y_bot_st
        
        self.x_coast = 0.
        self.y_coast_top = self.top_elev[int(abs(self.x_st / self.delcol))]        
        self.y_bot_cst = self.y_coast_top - y_bot_cst
        
        #   then we need to find the continental shelf and foot of cont. slope X coordinates 
        self.sb_pt = find_shlf_edge(self.top_elev)
        self.x_shelf_edge = round(self.x_st + (self.sb_pt[0] * self.delcol), 3)
        self.y_shelf_top = self.top_elev[self.sb_pt[0]]
        self.y_bot_shlf = self.y_shelf_top - y_bot_shlf        
            
        #self.fos_pt = find_FOS(top_elev)[1]
        self.fos_pt = [len(self.top_elev) - 1, self.top_elev[-1]]
        self.x_fos = self.x_st + len(self.top_elev) * self.delcol
        self.y_fos_top = self.top_elev[-1]
        self.y_bot_fos = self.y_fos_top - y_bot_fos        
        
    def create_IBOUND_arr(self):
        #   create the initial ibound array and top and bottom elevation        
        initial_model = create_ibound_arr(self.delcol, self.dellay, self.x_st, self.top_elev[0], self.y_bot_st, self.x_coast, self.y_coast_top, self.y_bot_cst,\
                                          self.x_shelf_edge, self.sb_pt[1], self.y_bot_shlf, self.x_fos, self.top_elev[-1], self.y_bot_fos, self.top_elev)
        self.ibound_arr, self.bot_elev_lst, self.y_mid_cell_lst, self.x_mid_cell_lst, self.top_elev_lst = initial_model[0], initial_model[1], initial_model[2], initial_model[3], initial_model[4]

        #   create initial condition arrays - as ibound arrays
        self.ic_head_arr = self.ibound_arr.astype('float')
        self.ic_head_arr[self.ic_head_arr == 0] = np.nan
        self.ic_salinity_arr = self.ibound_arr.astype('float')
        self.ic_salinity_arr[self.ic_salinity_arr == 0] = np.nan
        
        #   save the ibound_array as csv
        #x_header = [round(i, 3) for i in np.arange((self.x_st) + self.delcol / 2., (self.x_st) + self.ibound_arr.shape[-1] * self.delcol + self.delcol / 2., self.delcol).tolist()]
        x_header = [round(i, 3) for i in np.arange((self.x_st) + self.delcol / 2., (self.x_st) + self.ibound_arr.shape[-1] * self.delcol, self.delcol).tolist()]
        x_header = [str(i) for i in x_header]
        ##y_index = np.arange(max(self.top_elev) - (dL * 1000.) / 2., min(self.bot_elev) + (dL * 1000.) / 2., dL).tolist()
        df = pd.DataFrame(self.ibound_arr[:, 0, :])
        df.columns = x_header
        df.insert(0, column = 'xy_midcell', value = self.y_mid_cell_lst)
        df.to_csv(os.path.join(os.path.dirname(self.csv_in), 'ibound_arr.csv'), index = 'xy_midcell')

    #   clean the IBOUND array - remove lonely cell islands
    def clean_IBOUND_array(self):
            #   find the lonely cells and adapt the ibound array
        left, top, right, bot = 0, 0, -1, -1
        #   go through the ibound_arr cell by cell
        #       first find the left boundary - first column with active cells, usually will be 0 anyways
        for j in range(self.ibound_arr.shape[-1]):
            col_lst = self.ibound_arr[:, 0, j].tolist()
            if len([f for f in col_lst if f == 1]) != 0:
                left = j
                break
            else:
                pass
        #       first find the left boundary - first column with active cells, usually will be 0 anyways
        for i in range(self.ibound_arr.shape[0]):
            lay_lst = self.ibound_arr[i, 0, :].tolist()
            if len([f for f in lay_lst if f == 1]) != 0:
                top = i
                break   
            else:
                pass
        ##      then check the top boundary
        for i in range(top + 1, self.ibound_arr.shape[0]):
            lay_lst = self.ibound_arr[i, 0, left : right].tolist()
            if len([f for f in lay_lst if f == 1]) != 0:
                pass
            else:
                bot = i
                break
                   
        ##      then check the right boundary
        for j in range(left, self.ibound_arr.shape[-1]):
            col_lst = self.ibound_arr[top : bot, 0, j].tolist()
            if len([f for f in col_lst if f == 1]) != 0:
                pass
            else:
                right = j
                break    
        # trim the ibound array and other lists 
        self.ibound_arr = self.ibound_arr[top : bot, :, left : right]
        self.ic_head_arr = self.ic_head_arr[top : bot, :, left : right]
        self.ic_salinity_arr = self.ic_salinity_arr[top : bot, :, left : right]
        self.bot_elev_lst = self.bot_elev_lst[left : right]
        self.y_mid_cell_lst = self.y_mid_cell_lst[top : bot]
        self.x_mid_cell_lst = self.x_mid_cell_lst[left : right]
        self.top_elev_lst = self.top_elev_lst[left : right]

        #   change the CSV files
        x_header = [round(i, 3) for i in np.arange((self.x_st) + self.delcol / 2., (self.x_st) + self.ibound_arr.shape[-1] * self.delcol, self.delcol).tolist()]
        x_header = [str(i) for i in x_header]
        ##y_index = np.arange(max(self.top_elev) - (dL * 1000.) / 2., min(self.bot_elev) + (dL * 1000.) / 2., dL).tolist()
        df = pd.DataFrame(self.ibound_arr[:, 0, :])
        df.columns = x_header
        df.insert(0, column = 'xy_midcell', value = self.y_mid_cell_lst)
        df.to_csv(os.path.join(os.path.dirname(self.csv_in), 'ibound_arr.csv'), index = 'xy_midcell')

    #   create the netcdf file with IBOUND array and other model parameters
    def create_NC_file(self):
        self.nc_path = os.path.join(os.path.dirname(self.csv_in), self.name + '_INPUT.nc')
        if self.ibound_arr.shape[-1] != self.x_mid_cell_lst: 
            self.x_mid_cell_lst = [round(i, 3) for i in np.arange((self.x_st) + self.delcol / 2., (self.x_st) + self.ibound_arr.shape[-1] * self.delcol, self.delcol).tolist()]
            if self.ibound_arr.shape[-1] != self.x_mid_cell_lst: 
                self.x_mid_cell_lst = self.x_mid_cell_lst[: self.ibound_arr.shape[-1]]
                self.top_elev_lst = self.top_elev_lst[: self.ibound_arr.shape[-1]]
                self.bot_elev_lst = self.bot_elev_lst[: self.ibound_arr.shape[-1]]
                
        xa_sum = xr.Dataset(data_vars = {'ibound_arr' : (('y', 'x'), self.ibound_arr[:, 0, :]),
                                         'ic_head_arr' : (('y', 'x'), self.ic_head_arr[:, 0, :]),
                                         'ic_salinty_arr' : (('y', 'x'), self.ic_salinity_arr[:, 0, :]),
                                         'top_elev_midcell' : (('x'), self.top_elev_lst),
                                         'bot_elev_midcell' : (('x'), self.bot_elev_lst),
                                         'Hk_arr' : (('y', 'x'), self.ibound_arr[:, 0, :]),
                                         'Vk_arr' : (('y', 'x'), self.ibound_arr[:, 0, :]),
                                         'inl_pt' : ([self.y_bot_st, self.x_st]),
                                         'cst_pt' : ([self.y_bot_cst, self.x_coast]),
                                         'shlf_pt' : ([self.y_bot_shlf, self.x_shelf_edge]),
                                         'fos_pt' : ([self.y_bot_fos, self.x_fos]),
                                         'cont_shlf_edg' : ([self.sb_pt[0], self.sb_pt[1]])},
                            coords = {'y' : self.y_mid_cell_lst,
                                      'x' : self.x_mid_cell_lst})
        xa_sum.to_netcdf(self.nc_path)
        xa_sum.close()            
        
    #   create the initial csv file with the stress periods
    def create_SP_file(self):
         csv_headers = ['ID', 'Name', 'Duration (ka)', 'TS total', 'TS duration (ka)', 'Sea-level (m)', 'RCH type', 'RCH mean (m/d)', 'RCH stdev (m/d)', 'RCH datasource',\
                       'DRN type', 'DRN elev.', 'DRN conductance', 'BC inland type', 'BC inland head elev.', 'BC inland conductance', 'BC inland conc.', 'BC offshore type',\
                       'BC offshore head elev.', 'BC offshore conductance', 'BC offshore conc.']
         df_out = pd.DataFrame([[0, '', 1, 10, 0.1, 0., 'Randomized', 0.001, 0.00025, '', 'Constant', -0.25, 100., 'GHB', -1, 100., 0., 'GHB', 0., 100., 35.]])
         self.csv_dir = os.path.join(os.path.dirname(self.csv_in), self.name + '_SP_input.csv')
         df_out.to_csv(self.csv_dir, index = False, header = csv_headers)    

    #   create dictionary for each stress period, with all the other SEAWAT packages
    def create_SEAWAT_pckg_dictionaries(self):
        self.seawat_dict_dir = os.path.join(os.path.dirname(self.csv_in), self.name + '_SEAWAT_pckgs.npy')
        #   create dictionary with all the default values 
        dict_out = {}
        dict_out['DIS_laycbd'] = 0
        dict_out['LPF_laytyp'] = 0
        dict_out['LPF_ipakcb'] = 1
        dict_out['GHB_CHB_ipakcb'] = 1
        dict_out['RCH_nrchop'] = 3
        dict_out['RCH_conc'] = 0.
        dict_out['RCH_ipakcb'] = 1
        dict_out['DRN_ipakcb'] = 1
        dict_out['OC_ts_freq'] = 1
        dict_out['BTN_porosity'] = 0.3
        dict_out['BTN_dt0'] = 365.25
        dict_out['BTN_nprs'] = 1
        dict_out['BTN_ts_freq'] = 1
        dict_out['BTN_ifmtcn'] = 0
        dict_out['BTN_chkmas'] = False
        dict_out['BTN_nprmas'] = 10
        dict_out['BTN_nprobs'] = 10
        dict_out['DSP_al'] = 1.
        dict_out['DSP_trpt'] = 0.1
        dict_out['DSP_trpv'] = 0.1
        dict_out['DSP_dmcoef'] = 0.0000864
        dict_out['ADV_mixelm'] = 0
        dict_out['ADV_mxpart'] = 2000000
        dict_out['ADV_itrack'] = 1
        dict_out['ADV_wd'] = 0.5
        dict_out['ADV_dceps'] = 0.00001
        dict_out['ADV_nplane'] = 0
        dict_out['ADV_npl'] = 16
        dict_out['ADV_nph'] = 16
        dict_out['ADV_npmin'] = 0
        dict_out['ADV_npmax'] = 32
        dict_out['ADV_interp'] = 1
        dict_out['ADV_nlsink'] = 0
        dict_out['ADV_npsink'] = 16
        dict_out['ADV_dchmoc'] = 0.001
        dict_out['VDF_iwtable'] = 0
        dict_out['VDF_densemin'] = 1000
        dict_out['VDF_densemax'] = 1025
        dict_out['VDF_denseref'] = 1000
        dict_out['VDF_denseslp'] = 0.7143
        dict_out['VDF_firstdt'] = 0.001
        dict_out['GCG_iter1'] = 100
        dict_out['GCG_mxiter'] = 1
        dict_out['GCG_isolve'] = 1
        dict_out['GCG_cclose'] = 0.000001
        dict_out['PCG_hclose'] = 0.000100
        dict_out['PCG_rclose'] = 1
        np.save(self.seawat_dict_dir, dict_out)        

    
#   define the SEAWAT model class
class SEAWAT_model(object):
    
    """
    NC_input_dir = r'g:\_modelbuilder\swampy\data\test_model9\temp_files\test_model9_INPUT.nc'
    SP_tb_dir = r'g:\_modelbuilder\swampy\data\test_model9\temp_files\test_model9_SP_input.csv'
    SEAWAT_params_dir = r'g:\_modelbuilder\swampy\data\test_model9\temp_files\test_model9_SEAWAT_pckgs.npy'
    cs_points_dir = r'g:\_modelbuilder\swampy\data\test_model9\temp_files\cs_points.csv'
    """
    
    #   create the object with the necessary input
    def __init__(self, foldername, swat_exe_dir, NC_input_dir, SP_tb_dir, SEAWAT_params_dir, cs_points_dir, rch_foldrer_dir):
                 
        self.foldername = foldername
        self.swat_exe_dir = swat_exe_dir
        self.cs_points_dir = cs_points_dir
        self.rch_foldrer_dir = rch_foldrer_dir
        
        #   open the netcdf and get the basic input into the SEAWAT model
        xr_data = xr.open_mfdataset(NC_input_dir)
        self.x_coord, self.y_coord = xr_data['x'].values.tolist(), xr_data['y'].values.tolist()
        self.ibound_arr = xr_data['ibound_arr'].values
        self.top_elev = xr_data['top_elev_midcell'].values.tolist()
        self.bot_elev = xr_data['bot_elev_midcell'].values.tolist()
        self.inl_pt = xr_data['inl_pt'].values
        self.cst_pt = xr_data['cst_pt'].values
        self.shlf_pt = xr_data['shlf_pt'].values
        self.fos_pt = xr_data['fos_pt'].values
        self.hk_arr = xr_data['Hk_arr'].values
        self.vk_arr = xr_data['Vk_arr'].values
        self.cont_shlf_edg = xr_data['cont_shlf_edg'].values
        self.ic_head_arr = xr_data['ic_head_arr'].values
        self.ic_salinity_arr = xr_data['ic_salinty_arr'].values
        xr_data.close()
        self.delcol = abs(round((self.x_coord[0] - self.x_coord[1]) * 1000., 2))  
        self.dellay = abs(self.y_coord[0] - self.y_coord[1])
        self.nlay = self.ibound_arr.shape[0]
        self.nrow = 1
        self.ncol = self.ibound_arr.shape[-1]
        self.nper = 1
        self.delr = self.ncol * [self.delcol] 
        self.delc = 1.
        self.top = self.y_coord[0] + (self.dellay / 2.)
        self.bot = self.y_coord[-1] - (self.dellay / 2.)

        #   expand the axes of the arrays to fit into the SEAWAT model (which is quasi 3D with on row only)
        self.ibound_arr = np.expand_dims(self.ibound_arr, axis=1)
        self.hk_arr = np.expand_dims(self.hk_arr, axis=1)
        self.vk_arr = np.expand_dims(self.vk_arr, axis=1)
        self.ic_head_arr = np.expand_dims(self.ic_head_arr, axis=1)
        self.ic_salinity_arr = np.expand_dims(self.ic_salinity_arr, axis=1)

        #   open the csv file with the Stress Period information, the model will loop through the rows later on
        df_in = pd.read_csv(SP_tb_dir)
        self.sp_list = []
        for index, row in df_in.iterrows():
            self.sp_list.append(row)
        
        #   open the SEAWAT parameter file and assign all the parameters
        dir_dict_in = np.load(SEAWAT_params_dir, allow_pickle = True)  
        self.DIS_laycbd = dir_dict_in.item()['DIS_laycbd']
        self.LPF_laytyp = dir_dict_in.item()['LPF_laytyp']
        self.LPF_ipakcb = dir_dict_in.item()['LPF_ipakcb']
        self.GHB_CHB_ipakcb = dir_dict_in.item()['GHB_CHB_ipakcb']
        self.RCH_nrchop = dir_dict_in.item()['RCH_nrchop']
        self.RCH_conc = dir_dict_in.item()['RCH_conc']          
        self.RCH_ipakcb = dir_dict_in.item()['RCH_ipakcb']
        self.DRN_ipakcb = dir_dict_in.item()['DRN_ipakcb']
        self.OC_ts_freq = dir_dict_in.item()['OC_ts_freq']
        self.BTN_porosity = dir_dict_in.item()['BTN_porosity']
        self.BTN_dt0 = dir_dict_in.item()['BTN_dt0']
        self.BTN_nprs = dir_dict_in.item()['BTN_nprs']
        self.BTN_ts_freq = dir_dict_in.item()['BTN_ts_freq']
        self.BTN_ifmtcn = dir_dict_in.item()['BTN_ifmtcn']     
        self.BTN_nprmas = dir_dict_in.item()['BTN_nprmas']
        self.BTN_nprobs = dir_dict_in.item()['BTN_nprobs']
        self.BTN_chkmas = dir_dict_in.item()['BTN_chkmas']
        self.DSP_al = dir_dict_in.item()['DSP_al']
        self.DSP_trpt = dir_dict_in.item()['DSP_trpt']
        self.DSP_trpv = dir_dict_in.item()['DSP_trpv']
        self.ADV_mixelm = dir_dict_in.item()['ADV_mixelm']
        self.ADV_mxpart = dir_dict_in.item()['ADV_mxpart']
        self.ADV_itrack = dir_dict_in.item()['ADV_itrack']
        self.ADV_wd = dir_dict_in.item()['ADV_wd']
        self.ADV_dceps = dir_dict_in.item()['ADV_dceps']
        self.ADV_nplane = dir_dict_in.item()['ADV_nplane']
        self.ADV_npl = dir_dict_in.item()['ADV_npl']
        self.ADV_nph = dir_dict_in.item()['ADV_nph']
        self.ADV_npmin = dir_dict_in.item()['ADV_npmin']
        self.ADV_npmax = dir_dict_in.item()['ADV_npmax']
        self.ADV_interp = dir_dict_in.item()['ADV_interp']
        self.ADV_nlsink = dir_dict_in.item()['ADV_nlsink']
        self.ADV_npsink = dir_dict_in.item()['ADV_npsink']   
        self.ADV_dchmoc = dir_dict_in.item()['ADV_dchmoc']
        self.VDF_iwtable = dir_dict_in.item()['VDF_iwtable']
        self.VDF_densemin = dir_dict_in.item()['VDF_densemin']
        self.VDF_densemax = dir_dict_in.item()['VDF_densemax']
        self.VDF_denseref = dir_dict_in.item()['VDF_denseref']
        self.VDF_denseslp = dir_dict_in.item()['VDF_denseslp']
        self.VDF_firstdt = dir_dict_in.item()['VDF_firstdt']
        self.GCG_iter1 = dir_dict_in.item()['GCG_iter1']
        self.GCG_mxiter = dir_dict_in.item()['GCG_mxiter']
        self.GCG_isolve = dir_dict_in.item()['GCG_isolve']
        self.GCG_cclose = dir_dict_in.item()['GCG_cclose']
        self.PCG_hclose = dir_dict_in.item()['PCG_hclose']
        self.PCG_rclose = dir_dict_in.item()['PCG_rclose']
        self.DSP_dmcoef = dir_dict_in.item()['DSP_dmcoef']          
        
    def write_SEAWAT_packages(self, SP_tb_row_id, BAS_strt_arr, BTN_sconc_arr, test = False):
        
        #   1) create a model name for the model input, directory to save model output and load the stress period input
        sp_id = self.sp_list[SP_tb_row_id]['ID']
        sp_name = self.sp_list[SP_tb_row_id]['Name']
        if test:
            sp_perlen = 1.
            sp_nstp = 1
        else:
            sp_perlen = self.sp_list[SP_tb_row_id]['Duration (yrs)'] * 365.25
            sp_nstp = self.sp_list[SP_tb_row_id]['TS total']
            
        sp_sealevel = self.sp_list[SP_tb_row_id]['Sea-level (m)']
        sp_rch_type = self.sp_list[SP_tb_row_id]['RCH type']
        sp_rch_mean = self.sp_list[SP_tb_row_id]['RCH mean (m/d)']
        sp_rch_std = self.sp_list[SP_tb_row_id]['RCH stdev (m/d)']
        sp_rch_dir = self.sp_list[SP_tb_row_id]['RCH datasource']
        sp_drn_typ = self.sp_list[SP_tb_row_id]['DRN type']
        sp_drn_elev = self.sp_list[SP_tb_row_id]['DRN elev.']
        sp_drn_cond = self.sp_list[SP_tb_row_id]['DRN conductance']
        sp_bc_inl_typ = self.sp_list[SP_tb_row_id]['BC inland type']
        sp_bc_inl_head = self.sp_list[SP_tb_row_id]['BC inland head elev.']
        sp_bc_inl_cond = self.sp_list[SP_tb_row_id]['BC inland conductance']
        sp_bc_inl_conc = self.sp_list[SP_tb_row_id]['BC inland conc.']
        sp_bc_off_typ = self.sp_list[SP_tb_row_id]['BC offshore type']
        sp_bc_off_head = self.sp_list[SP_tb_row_id]['BC offshore head elev.']
        sp_bc_off_cond = self.sp_list[SP_tb_row_id]['BC offshore conductance']
        sp_bc_off_conc = self.sp_list[SP_tb_row_id]['BC offshore conc.']

        if math.isnan(sp_name):
            self.name = 'SP_' + str(sp_id)
        else:
            self.name = 'SP_' + str(sp_id) + '_' + sp_name

        self.sp_out_dir = os.path.join(self.foldername, self.name)
        self.nc_dir = os.path.join(self.foldername, self.name, 'NC_files')
        self.mswt = flopy.seawat.Seawat(self.name, 'nam_swt',  model_ws = self.sp_out_dir, exe_name = self.swat_exe_dir)

        #   set the random seed to 0, to always get the same values
        random.seed(0)
        
        #   2) define all necessary inputs and write the DIS (discretization) package    
        self.botm = np.arange(self.top - self.dellay, self.bot - self.dellay, -self.dellay)
        self.dis = flopy.modflow.ModflowDis(self.mswt, self.nlay, self.nrow, self.ncol, self.nper, self.delr, self.delc,\
                                            self.DIS_laycbd, self.top, self.botm, sp_perlen, sp_nstp)
        
        #   3) write the BAS package
        self.strt_arr = np.nan_to_num(BAS_strt_arr, copy=True, nan=0.0, posinf=None, neginf=None)
        self.sconc_arr = np.nan_to_num(BTN_sconc_arr, copy=True, nan=0.0, posinf=None, neginf=None)        
        
        self.bas = flopy.modflow.ModflowBas(self.mswt, self.ibound_arr, self.strt_arr)    
    
        #   4) write the LPF package
        self.lpf = flopy.modflow.ModflowLpf(self.mswt, laytyp = self.LPF_laytyp, hk = self.hk_arr, vka = self.vk_arr,  ipakcb = self.LPF_ipakcb)
        
        #   5) write the GHB package (if any of the boundaries is GHB)
        if sp_bc_inl_typ == 'GHB' or sp_bc_off_typ == 'GHB':
            #   create the GHB array
            cond_fact = 1.          #   multiplication factor to calculate conductance (default is 1)
            clay_incr = True        #   if the CONDUCTANCE in a cell falls below a treshold cond_limit_val, increase the value to the cond_limit_val  
            cond_limit_val = 0.01
            clay_cond_val = 0.1     #   conductance of a clay cell, constant value. Set because low conductance cells can lead to non-convergeance


            #   because laycon = 0 we need to specify the vcont and tran (optional)
            tran = np.zeros(((self.ibound_arr.shape[0] - 1), 1, self.ibound_arr.shape[-1]), dtype = np.float)
            tran = np.dot(self.hk_arr, self.dellay)
            tran = np.round(tran, 2)
            tran[tran == 0.] = 0.01
            #   define and initiate the output arrays and lists
            self.ghb_arr = self.ibound_arr * 1
            self.ghb_input_lst = []
            if clay_incr:
                self.cond_val = (tran * cond_fact) / self.delc
                self.cond_val_no_incr = (tran * cond_fact) / self.delc
                #   for all cells that are lower than 0.01 (which is the conductance of clay cell * 100 - to be on the safe side) assign the clay_cond_val
                self.cond_val[self.cond_val < cond_limit_val] = clay_cond_val
            else:
                self.cond_val = (tran * cond_fact) / self.delc
            self.ssmdata = []
            #  create the SSM dictionary where the ssm input will be written to
            itype = flopy.mt3d.Mt3dSsm.itype_dict()

            if sp_bc_inl_typ == 'GHB':
                #   add a GHB cell in the cells on the fresh-water boundary, only in active cells
                for b in range(self.ibound_arr.shape[0]):
                    if self.ibound_arr[b, 0, 0] == 1:
                        #   write the BC condition in each cell depending on the input BC inland head and conductance
                        if sp_bc_inl_head < 0:
                            if sp_bc_inl_cond < 0:
                                self.ghb_input_lst.append([b, 0, 0, self.top_elev[0], self.cond_val[b,0,0]])
                            else:
                                self.ghb_input_lst.append([b, 0, 0, self.top_elev[0], sp_bc_inl_cond])
                        else:
                            self.ghb_input_lst.append([b, 0, 0, sp_bc_inl_head, sp_bc_inl_cond])
                        self.ghb_arr[b, 0, 0] = -1
                        self.ssmdata.append([b, 0, 0, sp_bc_inl_conc, itype['GHB']])                

            if sp_bc_off_typ == 'GHB':
                #   assign the sea water boundary at the top cell in each offshore (under sea-level) column
                cst_col = self.x_coord.index(0.05)  #len(top_elev)
                """
                cst_col = len(self.top_elev)
                for f in reversed(self.top_elev):
                    if f < sp_sealevel:
                        cst_col -= 1
                    else:
                        break          
                """
                for a in range(cst_col, len(self.top_elev)):
                    #   get the index of the first non-zero ibound cell
                    try:
                        top_cell_lay_idx = self.ibound_arr[:, 0, a].tolist().index(1)
                        topo_val = self.top_elev[a]
                        if topo_val < sp_sealevel:
                            if sp_bc_off_head == sp_sealevel:
                                if sp_bc_off_cond < 0:
                                    self.ghb_input_lst.append([top_cell_lay_idx, 0, a, sp_sealevel, self.cond_val[top_cell_lay_idx, 0, a]])
                                else:
                                    self.ghb_input_lst.append([top_cell_lay_idx, 0, a, sp_sealevel, sp_bc_off_cond])
                            else:
                                self.ghb_input_lst.append([top_cell_lay_idx, 0, a, sp_sealevel, sp_bc_off_cond])    
                                #self.ghb_input_lst.append([top_cell_lay_idx, 0, a, sp_bc_off_head, sp_bc_off_cond])
                            self.ghb_arr[top_cell_lay_idx, 0, a] = -1
                            self.ssmdata.append([top_cell_lay_idx, 0, a, sp_bc_off_conc, itype['GHB']])
                    except ValueError:
                        pass

                #   add GHB cells in the last offshore model column 
                for c in range(self.ibound_arr.shape[0]):
                    if self.ibound_arr[c, 0, self.ibound_arr.shape[-1] - 1] == 1:
                        if sp_bc_off_head == sp_sealevel:
                            if sp_bc_off_cond < 0:
                                self.ghb_input_lst.append([c, 0, self.ibound_arr.shape[-1] - 1, sp_sealevel, self.cond_val[top_cell_lay_idx, 0, a]])
                            else:
                                self.ghb_input_lst.append([c, 0, self.ibound_arr.shape[-1] - 1, sp_sealevel, sp_bc_off_cond])
                        else:
                            self.ghb_input_lst.append([c, 0, self.ibound_arr.shape[-1] - 1, sp_bc_off_head, sp_bc_off_cond])                          
                        self.ghb_arr[c, 0, self.ibound_arr.shape[-1] - 1] = -1
                        self.ssmdata.append([c, 0, self.ibound_arr.shape[-1] - 1, sp_bc_off_conc, itype['GHB']])     
                 
            #   write the final output dictionary, inlcude each stress period
            self.ghb_arr_in = {}
            self.ghb_arr_in[0] = self.ghb_input_lst
            #   finally write the GHB input package
            self.ghb = flopy.modflow.ModflowGhb(self.mswt, ipakcb = self.GHB_CHB_ipakcb, stress_period_data = self.ghb_arr_in)  #   ipakcb - write output in cbc file    

        #   6) write the RCH package, create initial arrays
        self.rch_arr = np.array([[0.0] * 1 * self.ibound_arr.shape[-1]], dtype = np.float32)
        self.irch_arr = np.zeros((1, 1, self.ibound_arr.shape[-1]), dtype=np.float)
        #       if the rch type is randomized then for every cell that is above the sea level implement rch cell with random (mean and std) recharge value
        if sp_rch_type == 'Randomized':
            #   loop through the top_elev_lst and assign the precipitation value to cells above sea level
            for a in range(len(self.top_elev)):
                ibound_act_lay_idxs = [k for k, x in enumerate(self.ibound_arr[:, 0, a].tolist()) if x == 1]    
                if self.top_elev[a] >= sp_sealevel:
                    #   get random rch value
                    while round(abs(np.random.normal(sp_rch_mean, sp_rch_std)), 4) > 0.:
                        rch_val = round(abs(np.random.normal(sp_rch_mean, sp_rch_std)), 4)
                        break
                    try:
                        #   compare with the VK array - which acts as a RCH value limit
                        self.rch_arr[0][a] = min(self.vk_arr[ibound_act_lay_idxs[0], 0, a], rch_val)
                        self.irch_arr[0][0][a] = min(self.vk_arr[ibound_act_lay_idxs[0], 0, a], rch_val)
                        self.rch_extent = a
                    except IndexError:
                        self.rch_arr[0][a] = rch_val
                        self.irch_arr[0][0][a] = rch_val
                        self.rch_extent = a             
        #       if the type is Datasource, read 
        elif sp_rch_type == 'Datasource':
            rch_lst = loadRCHvalsFromDir(self.cs_points_dir, os.path.join(self.rch_foldrer_dir, sp_rch_dir))
            rch_mean = np.nanmean(rch_lst)
            rch_std = np.nanstd(rch_lst)
            #   loop through the list and assign values to the rch array, in case there are nan values then assign mean + stdev
            for a in range(len(self.top_elev)):
                ibound_act_lay_idxs = [k for k, x in enumerate(self.ibound_arr[:, 0, a].tolist()) if x == 1]    
                if self.top_elev[a] >= sp_sealevel:
                    if not math.isnan(rch_lst[a]):
                        rch_val = rch_lst[a]
                    else:
                        while round(abs(np.random.normal(rch_mean, rch_std)), 5) > 0.:
                            rch_val = round(abs(np.random.normal(rch_mean, rch_std)), 5)
                            break
                    try:
                        #   compare with the VK array - which acts as a RCH value limit
                        self.rch_arr[0][a] = min(self.vk_arr[ibound_act_lay_idxs[0], 0, a], rch_val)
                        self.irch_arr[0][0][a] = min(self.vk_arr[ibound_act_lay_idxs[0], 0, a], rch_val)
                        self.rch_extent = a
                    except IndexError:
                        self.rch_arr[0][a] = rch_val
                        self.irch_arr[0][0][a] = rch_val
                        self.rch_extent = a              
        #   write the recharge package 
        self.rch = flopy.modflow.ModflowRch(self.mswt, nrchop = self.RCH_nrchop, ipakcb = self.RCH_ipakcb, rech = self.rch_arr, irch = self.irch_arr)
            
        #   7) write the DRN package
        if sp_drn_typ == 'Constant':
            drn_input_lst = []
            for i in range(len(self.top_elev)):
                ibound_col_lst = self.ibound_arr[:, 0, i].tolist()
                #   check the 1st column with ibound_val = 1 (active cell)
                try:
                    drn_lay = ibound_col_lst.index(1)
                    ##  now check if the elevation is below sea level, if so assign the cell to ghb list
                    if self.top_elev[i] >= sp_sealevel:
                        if sp_drn_cond > 0:
                            drn_input_lst.append([drn_lay, 0, i, self.top_elev[i] + sp_drn_elev, sp_drn_cond])   # + sp_drn_elev because it is usually a negative value
                        else:
                            drn_input_lst.append([drn_lay, 0, i, self.top_elev[i] + sp_drn_elev, self.cond_val[drn_lay][0][i]])
                    else:
                        pass
                except ValueError:
                    pass
            #   write the final output dictionary, inlcude each stress period
            self.drn_arr_in = {}
            self.drn_arr_in[0] = drn_input_lst        
            if len(self.drn_arr_in[0]) != 0:
                self.drn = flopy.modflow.ModflowDrn(self.mswt, ipakcb = self.DRN_ipakcb, stress_period_data = self.drn_arr_in)            

        #   9) write the PCG package
        self.pcg = flopy.modflow.ModflowPcg(self.mswt, hclose = self.PCG_hclose, rclose = self.PCG_rclose)
        
        #   10) write the OC package
        self.ihedfm = 1          # a code for the format in which heads will be printed.
        self.iddnfm = 0          # a code for the format in which drawdowns will be printed.
        self.extension = ['oc','hds','ddn','cbc']
        self.unitnumber = [14, 30, 0, 50]
        #   create the dictionary that defines how to write the output file
        self.spd = {(0, 0): ['save head', 'save budget']}
        for t in range(0, self.nper):
            per = t #+ 1
            #   xrange allows to iterate through the list with specified step size - 25
            #   to save space on disk, every 10th timestep is saved
            for g in range(0, sp_nstp + 1, self.OC_ts_freq):
                self.spd[(per, int(g))] = ['save head', 'save budget']
                self.spd[(per, int(g) + 1)] = []
            self.spd[(per, int(g) + 1)] = ['save head', 'save budget']
            self.spd[(per, int(g) - 1)] = ['save head', 'save budget']
        self.oc = flopy.modflow.ModflowOc(self.mswt, stress_period_data = self.spd, compact = True)

        #   11) write the BTN package
        th_btn_time_step_val = int((sp_perlen / (25 * 365.25)) + 1)
        timprs_sp1 = np.linspace(1., sp_perlen, th_btn_time_step_val, endpoint = True)
        if self.nper > 1:
            timprs_sp2 = np.linspace(0, sp_perlen, self.BTN_ts_freq, endpoint = True)
            self.timprs = np.concatenate((timprs_sp1, timprs_sp2[1:]), axis = 0)
        else:
            self.timprs = timprs_sp1        
        self.btn = flopy.mt3d.Mt3dBtn(self.mswt, nprs = self.BTN_nprs, timprs = self.timprs, prsity = self.BTN_porosity, sconc = self.sconc_arr, ifmtcn = self.BTN_ifmtcn,
                             chkmas = self.BTN_chkmas, nprobs = self.BTN_nprobs, nprmas = self.BTN_nprmas, dt0 = self.BTN_dt0)
        
        #   12) write the ADV package
        """
        self.adv = flopy.mt3d.Mt3dAdv(self.mswt, mixelm = self.ADV_mixelm, mxpart = self.ADV_mxpart, itrack = self.ADV_itrack, wd = self.ADV_wd, dceps = self.ADV_dceps,\
                                      nplane = self.ADV_nplane, npl = self.ADV_npl, nph = self.ADV_nph, npmin = self.ADV_npmin, npmax = self.ADV_npmax,\
                                      interp = self.ADV_interp, nlsink = self.ADV_nlsink, npsink = self.ADV_npsink, dchmoc = self.ADV_dchmoc)
        """
        if self.ADV_mixelm == 0:
            self.adv = flopy.mt3d.Mt3dAdv(self.mswt, mixelm = self.ADV_mixelm, mxpart = self.ADV_mxpart)
        elif self.ADV_mixelm == 1:
            self.adv = flopy.mt3d.Mt3dAdv(self.mswt, mixelm = self.ADV_mixelm, mxpart = self.ADV_mxpart, itrack = self.ADV_itrack, wd = self.ADV_wd, dceps = self.ADV_dceps,\
                                          nplane = self.ADV_nplane, npl = self.ADV_npl, nph = self.ADV_nph, npmin = self.ADV_npmin, npmax = self.ADV_npmax)
        elif self.ADV_mixelm == 2:
            self.adv = flopy.mt3d.Mt3dAdv(self.mswt, mixelm = self.ADV_mixelm, itrack = self.ADV_itrack, wd = self.ADV_wd, interp = self.ADV_interp,\
                                          nlsink = self.ADV_nlsink, npsink = self.ADV_npsink)
        elif self.ADV_mixelm == 3:
            self.adv = flopy.mt3d.Mt3dAdv(self.mswt, mixelm = self.ADV_mixelm, mxpart = self.ADV_mxpart, itrack = self.ADV_itrack, wd = self.ADV_wd, dceps = self.ADV_dceps,\
                                          nplane = self.ADV_nplane, npl = self.ADV_npl, nph = self.ADV_nph, npmin = self.ADV_npmin, npmax = self.ADV_npmax,\
                                          interp = self.ADV_interp, nlsink = self.ADV_nlsink, npsink = self.ADV_npsink, dchmoc = self.ADV_dchmoc)
        elif self.ADV_mixelm == -1:
            self.adv = flopy.mt3d.Mt3dAdv(self.mswt, mixelm = self.ADV_mixelm)
    
        
        #   13) write the ADV package
        self.dsp = flopy.mt3d.Mt3dDsp(self.mswt, al = self.DSP_al, trpt = self.DSP_trpt, trpv = self.DSP_trpv, dmcoef = self.DSP_dmcoef)
        
        #   14) write the ADV package
        self.gcg = flopy.mt3d.Mt3dGcg(self.mswt, iter1 = self.GCG_iter1, mxiter = self.GCG_mxiter, isolve = self.GCG_isolve, cclose = self.GCG_cclose)
        
        #   15) write the ADV package
        self.vdf = flopy.seawat.SeawatVdf(self.mswt, iwtable = self.VDF_iwtable, densemin = self.VDF_densemin, densemax = self.VDF_densemax,\
                                          denseref = self.VDF_denseref, denseslp = self.VDF_denseslp, firstdt = self.VDF_firstdt)
        
        #   16) write the SSM package
        self.ssm_rch_in = np.array([[0.0] * 1 * self.ncol], dtype = np.float32)
        self.ssmdata_dict = {0: self.ssmdata,\
                             1: self.ssmdata}
        self.ssm = flopy.mt3d.Mt3dSsm(self.mswt, crch = self.ssm_rch_in, stress_period_data = self.ssmdata_dict)
        
        #   write all the SEAWAT package files
        self.mswt.write_input()
    
    #   Run the model and measure the run time
    def run_model(self):
        t0 = time()
        #   run the model
        v = self.mswt.run_model(silent = False, report = True)
        for idx in range(-3, 0):
            print(v[1][idx])
        #   stop measuring time and calculate total run time
        t1 = time()
        self.run_time = t1 - t0  
        #   then open the list file and check if the run converged
        self.converged = v[0]
        self.runtime = self.run_time / 60.

        """
        runtime_str = [i for i in v[1] if 'Elapsed run time:  ' in i]
        if len(runtime_str) > 0:
            if 'Minutes' in runtime_str[0]:
                mins = float('Elapsed run time:  1 Minutes,  0.125 Seconds'.split('Elapsed run time:  ')[1].split(' Minutes,  ')[1].split(' Seconds')[0])
                secs = float(runtime_str[0].split('Elapsed run time:  ')[1].split(' Minutes,  ')[1].split(' Seconds')[0]) / 60.
                self.runtime = str(round(mins + secs, 2))
            else:
                self.runtime = str(float(runtime_str[0].split('Elapsed run time:  ')[-1].split(' Seconds')[0]) / 60.)
        else:
            self.runtime = np.nan
        """
        
    #   check one by one if all the package files exist, and if the model converged
    def check_convergence(self):
        #   first check the packages
        dis_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.dis'))
        bas_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.bas'))
        lpf_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.lpf'))
        if os.path.exists(os.path.join(self.sp_out_dir, self.name + '.ghb')) or os.path.exists(os.path.join(self.sp_out_dir, self.name + '.chd')):
            ghb_chd_file = True
        else:
            ghb_chd_file = False
        rch_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.rch'))
        drn_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.drn'))
        pcg_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.pcg'))
        oc_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.oc'))
        btn_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.btn'))
        adv_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.adv'))
        dsp_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.dsp'))
        gcg_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.gcg'))
        vdf_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.vdf'))
        ssm_file = os.path.exists(os.path.join(self.sp_out_dir, self.name + '.ssm'))
        #   return all the booleans
        return [dis_file, bas_file, lpf_file, ghb_chd_file, rch_file, drn_file, pcg_file, oc_file, btn_file, adv_file, dsp_file, gcg_file, vdf_file, ssm_file]
        
    #   create output netcdf files and return the last sconc and strt arrays
    def create_model_output(self):
        #   read the UCN file and get all timesteps      
        self.hdsobj = bf.HeadFile(os.path.join(self.sp_out_dir, self.name + '.hds'))#, model = self.ml_results)#, precision = 'double')
        self.h = self.hdsobj.get_alldata()
        self.ucnobj = bf.UcnFile(os.path.join(self.sp_out_dir, 'MT3D001.UCN'))#, precision = 'double')
        self.time_steps = self.ucnobj.get_times()
        self.cbbobj = bf.CellBudgetFile(os.path.join(self.sp_out_dir, self.name + '.cbc'))#, precision = 'double')
        self.times_heads = self.cbbobj.get_times()        
        conc_ts_lst_yrs = [round(k / 365.25, 0) for k in self.time_steps]
        heads_ts_lst_yrs = [round(k / 365.25, 0) for k in self.times_heads]
        #   create directory
        if not os.path.exists(self.nc_dir):
            os.makedirs(self.nc_dir)
            
        #   if its SP0 then also save the initial concentration at time = 0
        if 'SP_0' in self.name:
            conc_arr = self.sconc_arr.astype(dtype = np.float64)
            head_arr = self.strt_arr.astype(dtype = np.float64)            
            #   create a mas of non active cells that will be applied to the starting arrays to set inactive cells not to be plotted later
            for i in range(self.ibound_arr.shape[0]):
                for j in range(self.ibound_arr.shape[-1]):
                    if self.ibound_arr[i, 0, j] == 0:
                        conc_arr[i, 0, j] = 1000
                        head_arr[i, 0, j] = 1000
            qx_in = self.cbbobj.get_data(text='flow right face', totim = self.times_heads[0])[0]
            qz_in = self.cbbobj.get_data(text = 'flow lower face', totim = self.times_heads[0])[0]   
            #   for the concentration, heads and cbc create a netcdf file (to save memory)
            xa_sum = xr.Dataset(data_vars = {'solute concentration' : (('y', 'x'), conc_arr[:, 0, :]),
                                             'heads' : (('y', 'x'),head_arr[:, 0, :]),
                                             'cbc Q_right' : (('y', 'x'), qx_in[:, 0, :]),
                                             'cbc Q_bottom' : (('y', 'x'), qz_in[:, 0, :])},
                                coords = {'x' : self.x_coord,
                                          'y' : self.y_coord})
            #   create a nc output folder
            xa_sum = xa_sum.assign_coords(time = 0)
            xa_name = self.name + '_0.nc'
            if os.path.exists(os.path.join(self.nc_dir, xa_name)):
                os.remove(os.path.join(self.nc_dir, xa_name))
            xa_sum.to_netcdf(os.path.join(self.nc_dir, xa_name))  
            
        #   loop through the list and plot the heads and concentration profiles for each of the chosen time steps
        for z in range(1, len(conc_ts_lst_yrs)):
            #   select the index of the time step to be plotted and get the concentration and heads arrays
            ts_plt_conc = min(conc_ts_lst_yrs, key = lambda x:abs(x - conc_ts_lst_yrs[z]))
            ts_plt_head = min(heads_ts_lst_yrs, key = lambda x:abs(x - conc_ts_lst_yrs[z]))
            conc_arr = self.ucnobj.get_data(totim = self.time_steps[conc_ts_lst_yrs.index(ts_plt_conc)]).astype(dtype = np.float64)
            head_arr = self.hdsobj.get_data(totim = self.times_heads[heads_ts_lst_yrs.index(ts_plt_head)]).astype(dtype = np.float64)
            qx_in = self.cbbobj.get_data(text='flow right face', totim = self.times_heads[heads_ts_lst_yrs.index(ts_plt_head)])[0]
            qz_in = self.cbbobj.get_data(text = 'flow lower face', totim = self.times_heads[heads_ts_lst_yrs.index(ts_plt_head)])[0]               
            #   for the concentration, heads and cbc create a netcdf file (to save memory)
            xa_sum = xr.Dataset(data_vars = {'solute concentration' : (('y', 'x'), conc_arr[:, 0, :]),
                                             'heads' : (('y', 'x'),head_arr[:, 0, :]),
                                             'cbc Q_right' : (('y', 'x'), qx_in[:, 0, :]),
                                             'cbc Q_bottom' : (('y', 'x'), qz_in[:, 0, :])},
                                coords = {'x' : self.x_coord,
                                          'y' : self.y_coord})
            #   create a nc output folder
            xa_sum = xa_sum.assign_coords(time = ts_plt_conc)
            xa_name = self.name + '_' + str(int(ts_plt_conc)) + '.nc'
            if os.path.exists(os.path.join(self.nc_dir, xa_name)):
                os.remove(os.path.join(self.nc_dir, xa_name))
            xa_sum.to_netcdf(os.path.join(self.nc_dir, xa_name))                         
        #   return the last conc_arr and head_arr that will act as input into the next stress period                    
        self.next_sconc_arr = conc_arr
        self.next_head_arr = head_arr

    #   run the model but in the background using a QGIS task, check https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/tasks.html
    def run_model_TASK(self, task):
        """
        Raises an exception to abort the task.
        Returns a result if success.
        The result will be passed, together with the exception (None in
        the case of success), to the on_finished method.
        If there is an exception, there will be no result.
        """
        QgsMessageLog.logMessage('Started task {}'.format(task.description()), MESSAGE_CATEGORY, Qgis.Info)
               
        #   run the model
        t0 = time()
        #   run the model
        v = self.mswt.run_model(silent = False, report = True)
        for idx in range(-3, 0):
            print(v[1][idx])
        #   stop measuring time and calculate total run time
        t1 = time()
        self.run_time = t1 - t0  
        #   then open the list file and check if the run converged
        self.converged = v[0]
        self.runtime = self.run_time / 60.

        if task.isCanceled():
            self.stopped(task)
  
    def stopped(self, task):
        QgsMessageLog.logMessage('Task "{name}" was canceled'.format(name=task.description()), MESSAGE_CATEGORY, Qgis.Info)
            
    def completed(self, exception, result = None):
        """This is called when doSomething is finished.
        Exception is not None if doSomething raises an exception.
        result is the return value of doSomething."""
        if exception is None:
            if result is None:
                QgsMessageLog.logMessage('Completed with no exception and no result probably manually canceled by the user)', MESSAGE_CATEGORY, Qgis.Warning)
                self.success = False
            else:
                QgsMessageLog.logMessage('Model completed\n'
                    'Total: {total}'.format(total = self.runtime), MESSAGE_CATEGORY, Qgis.Info)
                self.success = True
        else:
            QgsMessageLog.logMessage("Exception: {}".format(exception), MESSAGE_CATEGORY, Qgis.Critical)
            self.success = False
            raise exception




#   define the QgsTask that will run all the SPs of the SEAWAT model in one go
def run_model_QgSTask(task, foldername, swat_exe_dir, NC_input_dir, SP_tb_dir, SEAWAT_params_dir, cs_points_dir, rch_foldrer_dir):
    
    #   loop through the stress period csv and run model for each period, extracting the SCONC and STRT arrays at the end of each stress period
    df_in = pd.read_csv(SP_tb_dir)# SP_tb_dir = r'g:\_modelbuilder\swampy\data\test_model\temp_files\test_model_SP_input.csv'
    for a in range(df_in[df_in.columns[0]].count()):
        #   create the SEAWAT model object
        SEAWAT_model_obj = SEAWAT_model(foldername, swat_exe_dir, NC_input_dir, SP_tb_dir, SEAWAT_params_dir, cs_points_dir, rch_foldrer_dir)
        #   write the packages and run the model, first check if its first SP, if yes then use the predefined initial conditions
        if a == 0:
            strt = SEAWAT_model_obj.ic_head_arr
            sconc = SEAWAT_model_obj.ic_salinity_arr
        SEAWAT_model_obj.write_SEAWAT_packages(a, strt, sconc, test = False)
        
        SEAWAT_model_obj.run_model()
        SEAWAT_model_obj.create_model_output()
        sconc = SEAWAT_model_obj.next_sconc_arr
        strt = SEAWAT_model_obj.next_head_arr
        #   check if the packages were written succesfully and insert a row into the table in the SEAWAT tab
        pckg_success = SEAWAT_model_obj.check_convergence()
        pckg_success.insert(0, str(a))
        pckg_success = pckg_success + [SEAWAT_model_obj.converged , SEAWAT_model_obj.runtime]

    #   merge the netcdf files
    merge_SEAWAT_netcdf_files(foldername, cleanup = False)   

def stopped(task):
    QgsMessageLog.logMessage('Task "{name}" was canceled'.format(name=task.description()), MESSAGE_CATEGORY, Qgis.Info)
        
def run_model_QgSTask_completed(exception, result = None):
    """This is called when doSomething is finished.
    Exception is not None if doSomething raises an exception.
    result is the return value of doSomething."""
    if exception is None:
        if result is None:
            QgsMessageLog.logMessage('Completed with no exception and no result probably manually canceled by the user)', MESSAGE_CATEGORY, Qgis.Warning)
        else:
            QgsMessageLog.logMessage('Model completed', MESSAGE_CATEGORY, Qgis.Info)
    else:
        QgsMessageLog.logMessage("Exception: {}".format(exception), MESSAGE_CATEGORY, Qgis.Critical)
        raise exception    

"""
#   run the model and create the output, do it in the background so the UI is still responsive
task_run_model = QgsTask.fromFunction('Model for stress period ' + str(a)  + ' out of ' + str(df_in[df_in.columns[0]].count()), self.SEAWAT_model_obj.run_model_TASK, on_finished = self.SEAWAT_model_obj.completed)
QgsApplication.taskManager().addTask(task_run_model)

#   if the task finished successfully then process the results 
if task_run_model.finished():
"""


#   define the QgsTask that will run all the SPs of the SEAWAT model in one go
class run_model_QgSTask_2(QgsTask):
    def __init__(self, description, foldername, swat_exe_dir, NC_input_dir, SP_tb_dir, SEAWAT_params_dir, cs_points_dir, rch_foldrer_dir):
        super().__init__(description, QgsTask.CanCancel)
        self.foldername = foldername
        self.swat_exe_dir = swat_exe_dir
        self.NC_input_dir = NC_input_dir
        self.SP_tb_dir = SP_tb_dir
        self.SEAWAT_params_dir = SEAWAT_params_dir
        self.cs_points_dir = cs_points_dir
        self.rch_foldrer_dir = rch_foldrer_dir

    def run(self):
        #   loop through the stress period csv and run model for each period, extracting the SCONC and STRT arrays at the end of each stress period
        df_in = pd.read_csv(self.SP_tb_dir)# SP_tb_dir = r'g:\_modelbuilder\swampy\data\test_model\temp_files\test_model_SP_input.csv'
        for a in range(df_in[df_in.columns[0]].count()):

            # check isCanceled() to handle cancellation
            if self.isCanceled():
                return False

            #   create the SEAWAT model object
            SEAWAT_model_obj = SEAWAT_model(self.foldername, self.swat_exe_dir, self.NC_input_dir, self.SP_tb_dir, self.SEAWAT_params_dir, self.cs_points_dir, self.rch_foldrer_dir)
            #   write the packages and run the model, first check if its first SP, if yes then use the predefined initial conditions
            if a == 0:
                strt = SEAWAT_model_obj.ic_head_arr
                sconc = SEAWAT_model_obj.ic_salinity_arr
            SEAWAT_model_obj.write_SEAWAT_packages(a, strt, sconc, test = False)
            SEAWAT_model_obj.run_model()
            SEAWAT_model_obj.create_model_output()
            sconc = SEAWAT_model_obj.next_sconc_arr
            strt = SEAWAT_model_obj.next_head_arr
            #   check if the packages were written succesfully and insert a row into the table in the SEAWAT tab
            pckg_success = SEAWAT_model_obj.check_convergence()
            pckg_success.insert(0, str(a))
            pckg_success = pckg_success + [SEAWAT_model_obj.converged , SEAWAT_model_obj.runtime]        
            return True

    def finished(self, result):
        """This method is automatically called when self.run returns.
        result is the return value from self.run.
        This function is automatically called when the task has completed (
        successfully or otherwise). You just implement finished() to do 
        whatever
        follow up stuff should happen after the task is complete. finished is
        always called from the main thread, so it's safe to do GUI
        operations and raise Python exceptions here.
        """
        if result:
            QgsMessageLog.logMessage(
                'Task "{name}" completed\n' \
                'Total: {total} ( with {iterations} iterations)'.format(
                    name=self.description(),
                    total=self.total,
                    iterations=self.iterations),
                MESSAGE_CATEGORY, Qgis.Success)
        else:
            if self.exception is None:
                QgsMessageLog.logMessage(
                    'Task "{name}" not successful but without exception ' \
                    '(probably the task was manually canceled by the '
                    'user)'.format(
                        name=self.description()),
                    MESSAGE_CATEGORY, Qgis.Warning)
            else:
                QgsMessageLog.logMessage(
                    'Task "{name}" Exception: {exception}'.format(
                        name=self.description(), exception=self.exception),
                    MESSAGE_CATEGORY, Qgis.Critical)
                raise self.exception

    def cancel(self):
        QgsMessageLog.logMessage(
            'Task "{name}" was cancelled'.format(name=self.description()),
            MESSAGE_CATEGORY, Qgis.Info)
        super().cancel()





def interpolate_salinity_arr_QgSTask(task, in_salinity, out_salinity_arr, x_cells, y_cells, out_dir):
    interp_arr = out_salinity_arr
    for a in range(out_salinity_arr.shape[0]):
        ibound_act_lay_idxs = [k for k, x in enumerate(out_salinity_arr[a, :].tolist()) if x == 1]    

        if len(ibound_act_lay_idxs) > 0:
            try:
                y_coord = y_cells[a]
                #   get active cells in the layer
                ibound_act_lay_idxs = [k for k, x in enumerate(out_salinity_arr[a, :].tolist()) if x == 1]    
                y_coord_coscat = min(in_salinity['y'].values.tolist(), key = lambda x : abs(x - y_coord))
                ibound_act_lay_idxs_coscat = [k for k, x in enumerate(in_salinity.sel(y = y_coord_coscat).values.tolist()[0]) if x < 3500. if x >= 0.] 
                #   select the y coordinates of the concentration array for the SRM and COSCAT arrays
                conc_val_coscat = [in_salinity.sel(y = y_coord_coscat).values.tolist()[0][i] / 100. for i in ibound_act_lay_idxs_coscat]
                #conc_val_coscat = [i / 100. for i in in_salinity.sel(y = y_coord_coscat).values.tolist()[0][ibound_act_lay_idxs_coscat[0] : ibound_act_lay_idxs_coscat[0] + len(ibound_act_lay_idxs_coscat)]]
                #   resize the concentration into the size of the SRM column 
                #size = conc_val_coscat.shape[-1]#len(ibound_act_lay_idxs_coscat)
                if len(conc_val_coscat) > 0:
                    size = len(conc_val_coscat)
                    xloc = np.arange(size)
                    newsize = len(ibound_act_lay_idxs)
                    new_xloc = np.linspace(0, size, newsize)
                    srm_conc_vals = np.interp(new_xloc, xloc, conc_val_coscat)                                                                  
                    for b in range(len(ibound_act_lay_idxs)):
                        interp_arr[a, ibound_act_lay_idxs[b]] = srm_conc_vals[b]                                                      
            except (IndexError, KeyError): # if the average COSCAT array is smaller than the actual HYBAS IBOUND
                #   if the x_coord is further from coastline than 20km then assign fresh water concentration and head value of the topography
                for b in range(len(ibound_act_lay_idxs)):
                    if y_coord < in_salinity.y.values[0]:
                        conc_val = in_salinity.sel(y = in_salinity.y.values[0]).values[ibound_act_lay_idxs[b]] / 100.
                    else:
                        conc_val = 999.99
                    interp_arr[a, ibound_act_lay_idxs[b]] = conc_val                                           
            #   fill in random values in place of nan values
            for j in range(len(ibound_act_lay_idxs)):
                if out_salinity_arr[a, ibound_act_lay_idxs[j]] == 999.99:
                    try:
                        mean_conc_val = statistics.mean([k for k in out_salinity_arr[a, ibound_act_lay_idxs[0] : ibound_act_lay_idxs[-1]].tolist() / 100. if k > 0.0 and k < 100.0])
                        std_conc_val = statistics.stdev([k for k in out_salinity_arr[a, ibound_act_lay_idxs[0] : ibound_act_lay_idxs[-1]].tolist() / 100. if k > 0.0 and k < 100.0])
                    except StatisticsError:
                        mean_conc_val, std_conc_val = 0.0, 0.0
                    interp_arr[a, ibound_act_lay_idxs[j]] = round(abs(np.random.normal(mean_conc_val, std_conc_val)), 2)
    
    xa = xr.Dataset(data_vars = {'solute concentration' : (('y', 'x'), interp_arr[:, :])},
                        coords = {'x' : x_cells,
                                  'y' : y_cells})
    xa_name = 'IC_salinity_interpolated.nc'
    if os.path.exists(os.path.join(out_dir, xa_name)):
        os.remove(os.path.join(out_dir, xa_name))
    xa.to_netcdf(os.path.join(out_dir, xa_name))      
    
    
def interpolate_salinity_arr_QgSTask_completed(exception, result = None):
    """This is called when doSomething is finished.
    Exception is not None if doSomething raises an exception.
    result is the return value of doSomething."""
    if exception is None:
        if result is None:
            QgsMessageLog.logMessage('Completed with no exception and no result probably manually canceled by the user)', MESSAGE_CATEGORY, Qgis.Warning)
        else:
            QgsMessageLog.logMessage('Model completed', MESSAGE_CATEGORY, Qgis.Info)
    else:
        QgsMessageLog.logMessage("Exception: {}".format(exception), MESSAGE_CATEGORY, Qgis.Critical)
        raise exception    



#   function that will merge all the netcdf files into one final netcdf and also delete all the individual netcdfs
def merge_SEAWAT_netcdf_files(model_dir, cleanup = True):
    #model_dir = r'g:\_modelbuilder\swampy\data\test_model4\SEAWAT_model'
    #   loop through the directory
    subdirs = os.listdir(model_dir)
    #   create a list of netcdf files, ordered from start of simulation to the end
    nc_files_lst = []
    for i in range(len(subdirs)):
        #   organize the files by date of creation
        os.chdir(os.path.join(model_dir, subdirs[i], 'NC_files'))
        files = filter(os.path.isfile, os.listdir(os.path.join(model_dir, subdirs[i], 'NC_files')))
        files = [os.path.join(os.path.join(model_dir, subdirs[i], 'NC_files'), f) for f in files] # add path to each file
        files.sort(key=lambda x: os.path.getmtime(x))
        for file in files:
            nc_files_lst.append(file)
    #   start time at first time step
    time_model = int(os.path.basename(nc_files_lst[0]).split('.nc')[0].split('_')[-1])         
    ts = int(os.path.basename(nc_files_lst[1]).split('.nc')[0].split('_')[-1]) - time_model  
    #   first loop through all the files and add the time coordinate
    for file in nc_files_lst:
        with xr.open_dataset(file) as da:
            data = da.load()
            data_newcoord = data.assign_coords(time = time_model)
            data_expanded = data_newcoord.expand_dims('time')            
            da.close()
            data_expanded.to_netcdf(file)
            time_model = time_model + ts
    #   merge into one netcdf file
    ds = xr.open_mfdataset(nc_files_lst, concat_dim = 'time', combine='by_coords', autoclose = True)
    #   save the dataset
    merge_name = os.path.join(model_dir, 'final_SEAWAT_output.nc')
    ds.to_netcdf(merge_name)
    del ds    
    #   remove the individual netcdf files
    if cleanup:
        for subdir in subdirs:
            shutil.rmtree(os.path.join(model_dir, subdir))
         
    
#   function that interpolates salinity from a NC file into another IBOUND shape (model)
#   in_salinity_arr = array that we extract salinity from
#   out_salinity_arr = the array to which we interpolate the salinity
def interpolate_salinity_arr(in_salinity, out_salinity_arr, x_cells, y_cells):
    interp_arr = out_salinity_arr
    for a in range(out_salinity_arr.shape[0]):
        ibound_act_lay_idxs = [k for k, x in enumerate(out_salinity_arr[a, :].tolist()) if x == 1]    
        if len(ibound_act_lay_idxs) > 0:
            try:
                y_coord = y_cells[a]
                #   get active cells in the layer
                ibound_act_lay_idxs = [k for k, x in enumerate(out_salinity_arr[a, :].tolist()) if x == 1]    
                y_coord_coscat = min(in_salinity['y'].values.tolist(), key = lambda x : abs(x - y_coord))
                ibound_act_lay_idxs_coscat = [k for k, x in enumerate(in_salinity.sel(y = y_coord_coscat).values.tolist()[0]) if x < 3500. if x >= 0.] 
                #   select the y coordinates of the concentration array for the SRM and COSCAT arrays
                conc_val_coscat = [in_salinity.sel(y = y_coord_coscat).values.tolist()[0][i] / 100. for i in ibound_act_lay_idxs_coscat]
                #conc_val_coscat = [i / 100. for i in in_salinity.sel(y = y_coord_coscat).values.tolist()[0][ibound_act_lay_idxs_coscat[0] : ibound_act_lay_idxs_coscat[0] + len(ibound_act_lay_idxs_coscat)]]
                #   resize the concentration into the size of the SRM column 
                #size = conc_val_coscat.shape[-1]#len(ibound_act_lay_idxs_coscat)
                if len(conc_val_coscat) > 0:
                    size = len(conc_val_coscat)
                    xloc = np.arange(size)
                    newsize = len(ibound_act_lay_idxs)
                    new_xloc = np.linspace(0, size, newsize)
                    srm_conc_vals = np.interp(new_xloc, xloc, conc_val_coscat)                                                                  
                    for b in range(len(ibound_act_lay_idxs)):
                        interp_arr[a, ibound_act_lay_idxs[b]] = srm_conc_vals[b]                                                      
            except (IndexError, KeyError): # if the average COSCAT array is smaller than the actual HYBAS IBOUND
                #   if the x_coord is further from coastline than 20km then assign fresh water concentration and head value of the topography
                for b in range(len(ibound_act_lay_idxs)):
                    if y_coord < in_salinity.y.values[0]:
                        conc_val = in_salinity.sel(y = in_salinity.y.values[0]).values[ibound_act_lay_idxs[b]] / 100.
                    else:
                        conc_val = 999.99
                    interp_arr[a, ibound_act_lay_idxs[b]] = conc_val                                           
            #   fill in random values in place of nan values
            for j in range(len(ibound_act_lay_idxs)):
                if out_salinity_arr[a, ibound_act_lay_idxs[j]] == 999.99:
                    try:
                        mean_conc_val = statistics.mean([k for k in out_salinity_arr[a, ibound_act_lay_idxs[0] : ibound_act_lay_idxs[-1]].tolist() / 100. if k > 0.0 and k < 100.0])
                        std_conc_val = statistics.stdev([k for k in out_salinity_arr[a, ibound_act_lay_idxs[0] : ibound_act_lay_idxs[-1]].tolist() / 100. if k > 0.0 and k < 100.0])
                    except StatisticsError:
                        mean_conc_val, std_conc_val = 0.0, 0.0
                    interp_arr[a, ibound_act_lay_idxs[j]] = round(abs(np.random.normal(mean_conc_val, std_conc_val)), 2)
    return interp_arr
         
#   function to save/overwrite the numpy dictionary with SEAWAT package values
def save_SEAWAT_pckg_dictionaries(seawat_dict_dir, DIS_laycbd, LPF_laytyp, LPF_ipakcb, GHB_CHB_ipakcb, RCH_nrchop, RCH_conc, RCH_ipakcb, DRN_ipakcb,\
                                  OC_ts_freq, BTN_porosity, BTN_dt0, BTN_nprs, BTN_ts_freq, BTN_ifmtcn, BTN_chkmas, BTN_nprmas, BTN_nprobs, DSP_al,\
                                  DSP_trpt, DSP_trpv, DSP_dmcoef, ADV_mixelm, ADV_mxpart, ADV_itrack, ADV_wd, ADV_dceps, ADV_nplane, ADV_npl, ADV_nph,\
                                  ADV_npmin, ADV_npmax, ADV_interp, ADV_nlsink, ADV_npsink, ADV_dchmoc, VDF_iwtable, VDF_densemin, VDF_densemax, VDF_denseref,\
                                  VDF_denseslp, VDF_firstdt, GCG_iter1, GCG_mxiter, GCG_isolve, GCG_cclose, PCG_hclose, PCG_rclose):
    #   create dictionary with all the default values 
    dict_out = {}
    dict_out['DIS_laycbd'] = DIS_laycbd
    dict_out['LPF_laytyp'] = LPF_laytyp
    dict_out['LPF_ipakcb'] = LPF_ipakcb
    dict_out['GHB_CHB_ipakcb'] = GHB_CHB_ipakcb
    dict_out['RCH_nrchop'] = RCH_nrchop
    dict_out['RCH_conc'] = RCH_conc
    dict_out['RCH_ipakcb'] = RCH_ipakcb
    dict_out['DRN_ipakcb'] = DRN_ipakcb
    dict_out['OC_ts_freq'] = OC_ts_freq
    dict_out['BTN_porosity'] = BTN_porosity
    dict_out['BTN_dt0'] = BTN_dt0
    dict_out['BTN_nprs'] = BTN_nprs
    dict_out['BTN_ts_freq'] = BTN_ts_freq
    dict_out['BTN_ifmtcn'] = BTN_ifmtcn
    dict_out['BTN_chkmas'] = BTN_chkmas
    dict_out['BTN_nprmas'] = BTN_nprmas
    dict_out['BTN_nprobs'] = BTN_nprobs
    dict_out['DSP_al'] = DSP_al
    dict_out['DSP_trpt'] = DSP_trpt
    dict_out['DSP_trpv'] = DSP_trpv
    dict_out['DSP_dmcoef'] = DSP_dmcoef
    dict_out['ADV_mixelm'] = ADV_mixelm
    dict_out['ADV_mxpart'] = ADV_mxpart
    dict_out['ADV_itrack'] = ADV_itrack
    dict_out['ADV_wd'] = ADV_wd
    dict_out['ADV_dceps'] = ADV_dceps
    dict_out['ADV_nplane'] = ADV_nplane
    dict_out['ADV_npl'] = ADV_npl
    dict_out['ADV_nph'] = ADV_nph
    dict_out['ADV_npmin'] = ADV_npmin
    dict_out['ADV_npmax'] = ADV_npmax
    dict_out['ADV_interp'] = ADV_interp
    dict_out['ADV_nlsink'] = ADV_nlsink
    dict_out['ADV_npsink'] = ADV_npsink
    dict_out['ADV_dchmoc'] = ADV_dchmoc
    dict_out['VDF_iwtable'] = VDF_iwtable
    dict_out['VDF_densemin'] = VDF_densemin
    dict_out['VDF_densemax'] = VDF_densemax
    dict_out['VDF_denseref'] = VDF_denseref
    dict_out['VDF_denseslp'] = VDF_denseslp
    dict_out['VDF_firstdt'] = VDF_firstdt
    dict_out['GCG_iter1'] = GCG_iter1
    dict_out['GCG_mxiter'] = GCG_mxiter
    dict_out['GCG_isolve'] = GCG_isolve
    dict_out['GCG_cclose'] = GCG_cclose
    dict_out['PCG_hclose'] = PCG_hclose
    dict_out['PCG_rclose'] = PCG_rclose
    np.save(seawat_dict_dir, dict_out)      


         
"""   
netcdf_dir = r'g:\Water_Nexus\_A4_models\_SLR_models_OUT_files\_avg_nc_files_MERGING\0016_SRM_14.nc'
in_salinity = in_salinity_arr
out_salinity_arr = nc_dataset[13]
x_cells = nc_dataset[0]
y_cells = nc_dataset[1]
"""
"""
class interpolate_salinity_arr(QgsTask):
    def __init__(self, in_salinity, out_salinity_arr, x_cells, y_cells):
        super(interpolate_salinity_arr, self).__init__(QgsTask)
        self.in_salinity = in_salinity
        self.out_salinity_arr = out_salinity_arr
        self.x_cells = x_cells
        self.y_cells = y_cells
    def run(self):
        self.interp_arr = self.out_salinity_arr
        for a in range(self.out_salinity_arr.shape[0]):
            ibound_act_lay_idxs = [k for k, x in enumerate(self.out_salinity_arr[a, :].tolist()) if x == 1]    
            if len(ibound_act_lay_idxs) > 0:
                try:
                    y_coord = self.y_cells[a]
                    #   get active cells in the layer
                    ibound_act_lay_idxs = [k for k, x in enumerate(self.out_salinity_arr[a, :].tolist()) if x == 1]    
                    y_coord_coscat = min(self.in_salinity['y'].values.tolist(), key = lambda x : abs(x - y_coord))
                    ibound_act_lay_idxs_coscat = [k for k, x in enumerate(self.in_salinity.sel(y = y_coord_coscat).values.tolist()[0]) if x < 3500. if x >= 0.] 
                    #   select the y coordinates of the concentration array for the SRM and COSCAT arrays
                    conc_val_coscat = [self.in_salinity.sel(y = y_coord_coscat).values.tolist()[0][i] / 100. for i in ibound_act_lay_idxs_coscat]
                    #conc_val_coscat = [i / 100. for i in in_salinity.sel(y = y_coord_coscat).values.tolist()[0][ibound_act_lay_idxs_coscat[0] : ibound_act_lay_idxs_coscat[0] + len(ibound_act_lay_idxs_coscat)]]
                    #   resize the concentration into the size of the SRM column 
                    #size = conc_val_coscat.shape[-1]#len(ibound_act_lay_idxs_coscat)
                    if len(conc_val_coscat) > 0:
                        size = len(conc_val_coscat)
                        xloc = np.arange(size)
                        newsize = len(ibound_act_lay_idxs)
                        new_xloc = np.linspace(0, size, newsize)
                        srm_conc_vals = np.interp(new_xloc, xloc, conc_val_coscat)                                                                  
                        for b in range(len(ibound_act_lay_idxs)):
                            self.interp_arr[a, ibound_act_lay_idxs[b]] = srm_conc_vals[b]                                                      
                except (IndexError, KeyError): # if the average COSCAT array is smaller than the actual HYBAS IBOUND
                    #   if the x_coord is further from coastline than 20km then assign fresh water concentration and head value of the topography
                    for b in range(len(ibound_act_lay_idxs)):
                        if y_coord < self.in_salinity.y.values[0]:
                            conc_val = self.in_salinity.sel(y = self.in_salinity.y.values[0]).values[ibound_act_lay_idxs[b]] / 100.
                        else:
                            conc_val = 999.99
                        self.interp_arr[a, ibound_act_lay_idxs[b]] = conc_val                                           
                #   fill in random values in place of nan values
                for j in range(len(ibound_act_lay_idxs)):
                    if self.out_salinity_arr[a, ibound_act_lay_idxs[j]] == 999.99:
                        try:
                            mean_conc_val = statistics.mean([k for k in self.out_salinity_arr[a, ibound_act_lay_idxs[0] : ibound_act_lay_idxs[-1]].tolist() / 100. if k > 0.0 and k < 100.0])
                            std_conc_val = statistics.stdev([k for k in self.out_salinity_arr[a, ibound_act_lay_idxs[0] : ibound_act_lay_idxs[-1]].tolist() / 100. if k > 0.0 and k < 100.0])
                        except StatisticsError:
                            mean_conc_val, std_conc_val = 0.0, 0.0
                        self.interp_arr[a, ibound_act_lay_idxs[j]] = round(abs(np.random.normal(mean_conc_val, std_conc_val)), 2)


"""



        
"""

         csv_headers = ['ID', 'Name', 'Duration (ka)', 'TS total', 'TS duration (ka)', 'Sea-level (m)', 'RCH type', 'RCH mean (m/d)', 'RCH stdev (m/d)', 'RCH datasource',\
                       'DRN type', 'DRN elev.', 'DRN conductance', 'BC inland type', 'BC inland head elev.', 'BC inland conductance', 'BC inland conc.', 'BC offshore type',\
                       'BC offshore head elev.', 'BC offshore conductance', 'BC offshore conc.']
         df_out = pd.DataFrame([[0, '', 1, 10, 0.1, 0., 'Randomized', 0.001, 0.00025, '', 'Constant', -0.25, 100., 'GHB', 0., 100., 0., 'GHB', 0., 100., 35.]])
         csv_dir = r'g:\_modelbuilder\swampy\data\test.csv'
         df_out.to_csv(csv_dir, index = False, header = csv_headers, line_terminator='')    


        name = name
        csv_in = csv_in
        df = pd.read_csv(csv_in)
        top_elev = df['DEM'].values.tolist()        

        delcol = dC
        dellay = dL
        if x_st >= 0.:
            x_st = x_st * -1
        else:
            x_st = x_st
        y_bot_st = top_elev[0] - y_bot_st
        
        x_coast = 0.
        y_coast_top = top_elev[int(abs(x_st / delcol))]        
        y_bot_cst = y_coast_top - y_bot_cst
        
        #   then we need to find the continental shelf and foot of cont. slope X coordinates 
        sb_pt = find_shlf_edge(top_elev)
        x_shelf_edge = round(x_st + (sb_pt[0] * delcol), 3)
        y_shelf_top = top_elev[sb_pt[0]]
        y_bot_shlf = y_shelf_top - y_bot_shlf        
            
        #fos_pt = find_FOS(top_elev)[1]
        fos_pt = [len(top_elev) - 1, top_elev[-1]]
        x_fos = x_st + len(top_elev) * delcol
        y_fos_top = top_elev[-1]
        y_bot_fos = y_fos_top - y_bot_fos


        initial_model = create_ibound_arr(delcol, dellay, x_st, top_elev[0], y_bot_st, x_coast, y_coast_top, y_bot_cst,\
                                          x_shelf_edge, sb_pt[1], y_bot_shlf, x_fos, top_elev[-1], y_bot_fos, top_elev)
        ibound_arr, bot_elev, y_mid_cell_lst, x_mid_cell_lst, top_elev_lst = initial_model[0], initial_model[1], initial_model[2], initial_model[3], initial_model[4]


        #   save the ibound_array as csv
        #x_header = [round(i, 3) for i in np.arange((x_st) + delcol / 2., (x_st) + ibound_arr.shape[-1] * delcol + delcol / 2., delcol).tolist()]
        x_header = [round(i, 3) for i in np.arange((x_st) + delcol / 2., (x_st) + ibound_arr.shape[-1] * delcol, delcol).tolist()]
        x_header = [str(i) for i in x_header]
        ##y_index = np.arange(max(top_elev) - (dL * 1000.) / 2., min(bot_elev) + (dL * 1000.) / 2., dL).tolist()
        df = pd.DataFrame(ibound_arr[:, 0, :])
        df.columns = x_header
        df.insert(0, column = 'xy_midcell', value = y_mid_cell_lst)
        df.to_csv(os.path.join(os.path.dirname(csv_in), 'ibound_arr.csv'), index = 'xy_midcell')

        nc_path = r'g:\_modelbuilder\swampy\data\test_model106\temp_files\test_model106_INPUT.nc'
        topo_dict = xr.open_dataset(nc_path)
        
        topo_dict.close()
        
        xa_sum = xr.Dataset(data_vars = {'ibound_arr' : (('y', 'x'), ibound_arr[:, 0, :]),
                                         'top_elev_midcell' : (('x'), top_elev_lst),
                                         'bot_elev_midcell' : (('x'), bot_elev),
                                         'Hk_arr' : (('y', 'x'), ibound_arr[:, 0, :]),
                                         'Vk_arr' : (('y', 'x'), ibound_arr[:, 0, :]),
                                         'cst_pt' : ([y_bot_cst, x_coast])},
                            coords = {'y' : y_mid_cell_lst,
                                      'x' : x_mid_cell_lst})
        xa_sum.to_netcdf(nc_path)      




"""        


