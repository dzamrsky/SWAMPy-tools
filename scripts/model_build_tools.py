# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:49:55 2018

@author: lubna
"""
import os
import numpy as np
from .. import flopy
from scipy.signal import argrelextrema
import rdp

def create_ibound_arr(del_col, del_lay, x_start, y_start_top, y_start_bot, x_coast, y_coast_top, y_coast_bot, x_shelf_edge,
                      y_shelf_edge_top, y_shelf_edge_bot, x_foot_of_slope, y_foot_of_slope_top, y_foot_of_slope_bot):

        if x_start > 0.:
            print('Inland start of model domain is > coast')
            return  #   makes the function stop
    
        #   create the first version of the array based on the number of layers and columns
        #       1) get the number of columns based on the total extent of the domain in the x direction
        x_len_m = abs(x_start) + abs(x_foot_of_slope)         #   total distance in meters
        ncols = int(x_len_m / del_col)      #   total distance in columns (has to be an integer number so you can create a loop later!)
        #       2) calculate the number of layers based on the extent in the y direction
        y_len_m = abs(y_start_top) + abs(y_foot_of_slope_bot)
        nlays = int(y_len_m / del_lay)         
        #       3) create an array with the dimensions calculated above, fill with zeros first
        ibound_arr = np.zeros((nlays, 1, ncols), dtype = np.int16)
        
        #   define a list with mid_layer elevations
        mid_lay_elev = np.arange(y_start_top - (0.5 * del_lay), y_foot_of_slope_bot + (0.5 * del_lay), -del_lay)
        
        #   calculate the amount columns that span across the distance between the starting and ending point           
        n_cols_inland = int((abs(x_start) + 0.0) / del_col)
        
        #   calculate the drop in elevations for the top and bottom
        diff_y_inland_top = abs((y_start_top - y_coast_top)) / n_cols_inland     
        diff_y_inland_bot = abs((y_coast_bot - y_start_bot)) / n_cols_inland  
        
        #   create lists with elevation at each column between the inland boundary and the coast
        mid_col_top_elev_inland = np.linspace(y_start_top - (0.5 * diff_y_inland_top), y_coast_top + (0.5 * diff_y_inland_top), n_cols_inland).tolist() 
        mid_col_bot_elev_inland = np.linspace(y_start_bot - (0.5 * diff_y_inland_bot), y_coast_bot + (0.5 * diff_y_inland_bot), n_cols_inland).tolist() 

        #   round the numbers in the list to integers
        inland_top_elev_lst = [round(i, 1) for i in mid_col_top_elev_inland]        
        inland_bot_elev_lst = [round(i, 1) for i in mid_col_bot_elev_inland]       
        
        #   
        for a in range(0, n_cols_inland):   
            #   find the top and bottom elevation at the respective column
            top_elev_col = inland_top_elev_lst[a]
            bot_elev_col = inland_bot_elev_lst[a]
            #   loop through the layers 
            for b in range(len(mid_lay_elev)):
                
                #   check if the mid_lay_elev is between the top and bot elevation of the column
                mid_lay = mid_lay_elev[b]
                
                if mid_lay < top_elev_col and mid_lay > bot_elev_col:
                    #print top_elev_col, bot_elev_col, mid_lay, a, b 
                    #   change the cells to be active in the ibound_arr
                    ibound_arr[b, 0, a] = 1
       
        " ******** COAST PART ********* "
        #   calculate the amount columns that span across the distance between the starting and ending point           
        n_cols_coast = int(abs((x_coast + x_shelf_edge)) / del_col)        
        
        #   calculate the drop in elevations for the top and bottom
        diff_y_coast_top = abs((y_coast_top - y_shelf_edge_top)) / n_cols_coast   
        # y_shelf_edge_bot condition
        if y_shelf_edge_bot == None:
            diff_y_coast_bot = abs((y_coast_bot - y_foot_of_slope_bot)) / n_cols_coast
            #   you need to specify the value of the y_shelf_edge_bot because you use it in the next step and it cant calculate with None value
            y_shelf_edge_bot = y_coast_bot - (y_coast_bot - y_foot_of_slope_bot) / 2
        else:
            diff_y_coast_bot = abs((y_coast_bot - y_shelf_edge_bot)) / n_cols_coast        
        
        #   create lists with elevation at each column between the inland boundary and the coast
        mid_col_top_elev_coast = np.linspace(y_coast_top - (0.5 * diff_y_coast_top), y_shelf_edge_top + (0.5 * diff_y_coast_top), n_cols_coast).tolist() 
        mid_col_bot_elev_coast = np.linspace(y_coast_bot - (0.5 * diff_y_coast_bot), y_shelf_edge_bot + (0.5 * diff_y_coast_bot), n_cols_coast).tolist() 

        #   round the numbers in the list to integers
        coast_top_elev_lst = [round(i, 1) for i in mid_col_top_elev_coast]        
        coast_bot_elev_lst = [round(i, 1) for i in mid_col_bot_elev_coast]       
        
        #   
        for a in range(0, n_cols_coast):   
            #   find the top and bottom elevation at the respective column
            top_elev_col = coast_top_elev_lst[a]
            bot_elev_col = coast_bot_elev_lst[a]
            #   loop through the layers 
            for b in range(len(mid_lay_elev)):
                
                #   check if the mid_lay_elev is between the top and bot elevation of the column
                mid_lay = mid_lay_elev[b]
                
                if mid_lay < top_elev_col and mid_lay > bot_elev_col:
                    #print top_elev_col, bot_elev_col, mid_lay, a, b 
                    #   change the cells to be active in the ibound_arr
                    ibound_arr[b, 0, a + n_cols_inland] = 1        
       
    
        " ******** SHELF PART ********* "
        #   calculate the amount columns that span across the distance between the starting and ending point           
        n_cols_shelf = int((x_foot_of_slope - abs(x_shelf_edge)) / del_col)
        
        #   calculate the drop in elevations for the top and bottom
        diff_y_shelf_top = abs((y_shelf_edge_top - y_foot_of_slope_top)) / n_cols_shelf
        
        # y_shelf_edge_bot condition
        if y_shelf_edge_bot == None:
            diff_y_shelf_bot = abs((y_coast_bot - y_foot_of_slope_bot)) / n_cols_coast
        else:
            diff_y_shelf_bot = abs((y_shelf_edge_bot - y_foot_of_slope_bot)) / n_cols_coast            
        
        #   create lists with elevation at each column between the inland boundary and the coast
        mid_col_top_elev_shelf = np.linspace(y_shelf_edge_top - (0.5 * diff_y_shelf_top), y_foot_of_slope_top + (0.5 * diff_y_shelf_top), n_cols_shelf).tolist() 
        mid_col_bot_elev_shelf = np.linspace(y_shelf_edge_bot - (0.5 * diff_y_shelf_bot), y_foot_of_slope_bot + (0.5 * diff_y_shelf_bot), n_cols_shelf).tolist() 

        #   round the numbers in the list to integers
        shelf_top_elev_lst = [round(i, 1) for i in mid_col_top_elev_shelf]        
        shelf_bot_elev_lst = [round(i, 1) for i in mid_col_bot_elev_shelf]       
        
        #   
        for a in range(0, n_cols_shelf):   
            #   find the top and bottom elevation at the respective column
            top_elev_col = shelf_top_elev_lst[a]
            bot_elev_col = shelf_bot_elev_lst[a]
            #   loop through the layers 
            for b in range(len(mid_lay_elev)):
                
                #   check if the mid_lay_elev is between the top and bot elevation of the column
                mid_lay = mid_lay_elev[b]
                
                if mid_lay < top_elev_col and mid_lay > bot_elev_col:
                    #print top_elev_col, bot_elev_col, mid_lay, a, b 
                    #   change the cells to be active in the ibound_arr
                    ibound_arr[b, 0, a + n_cols_inland + n_cols_coast] = 1
        
        top_elev = inland_top_elev_lst + coast_top_elev_lst + shelf_top_elev_lst
        bot_elev = inland_bot_elev_lst + coast_bot_elev_lst + shelf_bot_elev_lst
        
        return ibound_arr, top_elev, bot_elev



def find_lonely_cells(ibound_array):

    col_n = ibound_array.shape[-1]
    row_n = ibound_array.shape[0]

    #   go through the ibound_arr cell by cell
    for j in range(1, col_n - 1):
        for i in range(1, row_n - 1):
            if j == 0:
                left = 0
            else:
                left = ibound_array[i-1,0,j]
            
            if j == (col_n - 1):
                right = 0
            else:
                right = ibound_array[i+1,0,j]
                
            if i == 0:
                up = 0
            else:
                up = ibound_array[i,0,j-1]
                
            if i == (row_n -1):
                down = 0
            else:
                down = ibound_array[i,0,j+1]

            if (left == 0 and right == 0 and up == 0 and down == 0):
                ibound_array[i,0,j] = 0
 
    return ibound_array


"""
ibound_arr_in = ibound_arr

hk_aqf = 10.                     #   HK value of the aquifer material
hk_aqt = 0.001                   #   HK value of the aquitard material
aqt_top = True                   #   Presence/absence of the top aquitard layer
aqt_layer = True                 #   Presence/absence of the aquitard layer(s) within the aquifer body
aqt_top_thk = 1                  #   the thickness of the top clay layer (in cell layers!!!)
aqt_top_col_start = 25           #   start of the top clay layer (in cell column!!)
aqt_top_col_end = 50             #   end of the top clay layer (in cell column!!)

aqt_lay_thk = [1, 2]             #   thickness of the aquitard layer (in cell layers)
aqt_lay_lay_top = [4, 7]         #   starting layer of the aquitard (layer cell below surface)
aqt_lay_col_start = [15, 35]     #   start of the aquitard layer (in cell column!!)
aqt_lay_col_end = [40, 67]       #   end of the aquitard layer (in cell column!!)
"""

def create_hk_arr(ibound_arr_in, hk_aqf, hk_aqt, anis_factor = 0.1, aqt_top = False, aqt_layer = False,
                  aqt_top_thk = None, aqt_top_col_start = None, aqt_top_col_end = None, aqt_lay_thk = None, aqt_lay_lay_top = None,
                  aqt_lay_col_start = None, aqt_lay_col_end = None):
    
    #   replicate the ibound array
    hk_arr = ibound_arr_in * hk_aqf

    #   if there an aquitard (clay) capping layer on top assign the hk_aqt to the corresponding cells
    if aqt_top is True:
        for i in range(aqt_top_col_start, aqt_top_col_end):
            #   now select the right amount of IBOUND cells (starting at the top of the model domain) based
            #   on the aqt_top_thk indicating the thickness of the clay layer in model layers
            aqt_top_top = ibound_arr_in[:, 0, i].tolist().index(1)
            aqt_top_bot = aqt_top_top + aqt_top_thk
            hk_arr[aqt_top_top : aqt_top_bot, 0, i] = hk_aqt
    
    #   if there is an aquitard (clay) layer present in the aquifer change the hk values of the corresponding cells
    if aqt_layer is True:
        #   loop through the input lists, there can be multiple clay layers
        for a in range(len(aqt_lay_thk)):
            for j in range(aqt_lay_col_start[a], aqt_lay_col_end[a]):
                #   now select the right amount of IBOUND cells (starting at the top of the model domain) based
                #   on the aqt_top_thk indicating the thickness of the clay layer in model layers
                aqt_lay_top = ibound_arr_in[:, 0, j].tolist().index(1) + aqt_lay_lay_top[a]
                aqt_lay_bot = aqt_lay_top + aqt_lay_thk[a]
                hk_arr[aqt_lay_top : aqt_lay_bot, 0, j] = hk_aqt            

    #   create the VK array based on the anisotropy factor
    vk_arr = hk_arr * anis_factor
    
    #   return the final arrays
    return hk_arr, vk_arr



"""
ibound_arr_in = ibound_arr

head_val = 1.                   #   starting head value (constant)
conc_scenario = 'saline'        #   starting concentration scenario
coast_idx = 50                  #   index of the coastline 
"""

def start_head_conc(ibound_arr_in, head_val, coast_idx, conc_scenario):
    
    #   replicate the ibound array to create the starting head and concentration arrays
    strt_arr = ibound_arr_in * head_val
    
    #   do the same for the concentration array, based on the scenario
    if conc_scenario == 'fresh':
        #   if the scenario is fresh then make all active cells fresh
        sconc_arr = ibound_arr_in * 0.0
    
    elif conc_scenario == 'saline': 
        #   if the scenario is saline then make all active cells saline
        sconc_arr = ibound_arr_in * 35.0

    elif conc_scenario == 'coastline':
        #   the last scenario starts as fresh cells in the inland part and saline in the offshore
        sconc_arr = np.concatenate((ibound_arr_in[:, :, :coast_idx] * 0.0, ibound_arr_in[:, :, coast_idx:] * 35.0), axis = 2)

    return strt_arr, sconc_arr



"""  General-Head Boundary Package (GHB)  

hk_array = hk_arr
del_layer = del_lay
del_column = del_col
sea_lvl = 0.
top_elev_lst = top_elev
perlen_lst = perlen

"""
def create_ghb_input(ibound_arr_in, hk_array, del_layer, del_column, sea_lvl, top_elev_lst, perlen_lst):

    #   create a transmissivity array (m2/d)
    tran = np.dot(hk_array, del_column)
    cond_val = tran / del_layer
    """
    Double check that the calculation above is correct!
    """
    
    #   define an empty list for the SSM and GHB packages
    ssmdata = []
    ghb_input_lst = []
    itype = flopy.mt3d.Mt3dSsm.itype_dict()
    
    #    in the inland part, assign the GHB boundary to the firs column only
    for b in range(hk_array.shape[0]):
        #   check that the value is different than 0 = assign values to active cells only
        if hk_array[b, 0, 0] != 0.:
            ghb_input_lst.append([b, 0, 0, top_elev_lst[0], cond_val[b][0][0]])
            #   the concentration of these cells is always 0.0 (fresh)
            ssmdata.append([b, 0, 0, 0.0, itype['GHB']])
    
    #   offshore - only the top cell of each column (if top_elev_lst < sea level)
    for c in range(hk_array.shape[-1]):
        if top_elev_lst[c] < sea_lvl:
            #   if the top elevation of that column is below sea level find first active cell in the column
            try:
                lay_idx = ibound_arr_in[:, 0, c].tolist().index(1)
                #   head elevation is always equal to the sea level and concentration to 35.0 (saline water)
                ghb_input_lst.append([lay_idx, 0, c, sea_lvl, cond_val[lay_idx][0][c]])
                ssmdata.append([lay_idx, 0, c, 35.0, itype['GHB']])
            # this error happens when there are no active cells in the column
            except ValueError:
                pass

    # write the final output dictionaries, inlcude each stress period       
    ghb_arr_in = {}
    for d in range(len(perlen_lst)):
        ghb_arr_in[d] = ghb_input_lst        
    
    #ssm_arr_in = {}
    #for d in xrange(len(perlen_lst)):
    #    ssm_arr_in[d] = ssmdata    
    
    return ghb_arr_in, ssmdata, cond_val
                
                     
"""   Recharge Package (RCH)  

rch_rate = prec_val     #   constant recharge value 
ncol_val = ncol         #   number of columns in the model domain
sea_lvl = 0.            #   sea level elevation (compared to topography)
top_elev_lst = top_elev #   list of top elevations

"""

def create_rch_input(rch_rate, ncol_val, sea_lvl, top_elev_lst):

    #   create a recharge array, for every model column, filled with 0.0 value for now (no recharge over ocean)
    rch_arr = np.array([[0.0] * 1 * ncol_val], dtype = np.float32)
    
     #   loop through the top_elev_lst and assign the precipitation value to cells above sea level
    for a in range(ncol_val):
        if top_elev_lst[a] >= sea_lvl:
           rch_arr[0][a] = rch_rate
    
    ssm_rch_arr = np.array([[0.0] * 1 * ncol_val], dtype = np.float32)    
    
    return rch_arr, ssm_rch_arr
    
    

"""   Drainage Package (DRN)  

ncol_val = ncol         #   number of columns in the model domain
sea_lvl = 0.            #   sea level elevation (compared to topography)
top_elev_lst = top_elev #   list of top elevations
ibound_arr_in = ibound_arr 
cond_val_arr_in = cond_val
perlen_lst = perlen
"""

def create_drn_input(ibound_arr_in, cond_val_arr_in, ncol_val, sea_lvl, top_elev_lst, perlen_lst):

    #  drainage is assigned only to cells that receive recharge - cells with elev above sea level
    drn_input_lst = []
    for i in range(ncol_val):
        # check if the elevation is below sea level, if so assign the cell to ghb list
        if top_elev_lst[i] >= sea_lvl:
            try:
                #   check the 1st column with ibound_val = 1 (active cell)
                drn_lay = ibound_arr_in[:, 0, i].tolist().index(1)
                drn_input_lst.append([drn_lay, 0, i, top_elev_lst[i], cond_val_arr_in[drn_lay][0][i]])
            except ValueError:
                pass
        else:
            pass
  
    #   write the final output dictionary, inlcude each stress period
    drn_arr_in = {}
    for c in range(len(perlen_lst)):
        drn_arr_in[c] = drn_input_lst

    return drn_arr_in



"""     Basic transport package (BTN)
creates the timprs parameter

nper_val = nper
perlen_lst = perlen
th_ts = th_btn_time_step_val
end_pt = False
"""

def create_btn_input(nper_val, perlen_lst, th_ts, end_pt = False):
    #   the nprs parameter defines how many transport steps are going to be exported to the UCN file
    timprs_sp1 = np.linspace(1., perlen_lst[0], th_ts[0], endpoint = end_pt)
    #   depending on the number of stress periods create the output timprs parameter list
    if nper_val > 1:
        timprs_sp2 = np.linspace(perlen_lst[0], perlen_lst[0] + perlen_lst[1], th_ts[1], endpoint = True)
        timprs = np.concatenate((timprs_sp1, timprs_sp2[1:]), axis = 0)
    else:
        timprs = timprs_sp1

    return timprs

"""     Output control (OC)

nper_val = nper
nstp_val = nstp
th_ts = 1
"""

def create_oc_input(nper_val, nstp_val, th_ts):

    #   create the dictionary that defines how to write the output file
    spd = {(0, 0): ['save head', 'save budget']}
    for t in range(0, len(nstp_val)):
        per = t 
        #   xrange allows to iterate through the list with specified step size - 25
        #   to save space on disk, every 10th timestep is saved
        for g in range(0, nstp_val[t] + 1, th_ts):
            spd[(per, int(g))] = ['save head', 'save budget']
            spd[(per, int(g) + 1)] = []
    
        spd[(per, int(g) + 1)] = ['save head', 'save budget']
        spd[(per, int(g) - 1)] = ['save head', 'save budget']    

    return spd


"""         River package (RIV)
riv_soil = 'sandy_silt'       # 'silt_loam', 'clay_loam', 'silty_clay'    #  river bottom material
riv_x_st = -2000.             #  starting location relative to the coastline  
riv_w = 300.                  #  river width in meters
riv_be = -2.5                 #  river bottom elevation - relative to the elevation of the cell
riv_he = 0.0                  #  river head elevation (stage) relative to the cell elevation
x_st = x_start
d_col = del_col
d_lay = del_lay
t_elev_lst = top_elev
ib_arr = ibound_arr
v_thk = 1.                    #   vertical thickness of the sediment (m)
perlen_lst = perlen

#               k soil values from https://www.sciencedirect.com/science/article/pii/S0022169402000677
"""

def create_riv_input(x_st, v_thk, perlen_lst, d_col, d_lay, t_elev_lst, ib_arr, riv_x_st, riv_w, riv_be, riv_he, riv_soil):
    
    #   first calculate at what column the river starts and ends
    col_st = abs(int(round((x_st - riv_x_st) / d_col, 0)))
    col_end = col_st + int(round(riv_w / d_col))
    
    #   get the right K value based on soil type
    if riv_soil == 'sandy_silt':
        k_val = 0.3 # m/d
    elif riv_soil == 'silt_loam':
        k_val = 0.03
    elif riv_soil == 'clay_loam':
        k_val = 0.015
    elif riv_soil == 'silty_clay':
        k_val = 0.003
        
    riv_input_lst = []
    #   for the columns specified above assign the riv cell
    for col in range(col_st, col_end):
        cond = (k_val * d_col * 1) / v_thk  #   the 1 represents the width of the model row 
        stage = t_elev_lst[col] + riv_he
        rbot = t_elev_lst[col] + riv_be
        lay_idx = ib_arr[:, 0, col].tolist().index(1)            
        riv_input_lst.append([lay_idx, 0, col, stage, cond, rbot])

    # write the final output dictionaries, inlcude each stress period       
    riv_arr_in = {}
    for d in range(len(perlen_lst)):
        riv_arr_in[d] = riv_input_lst                 
    
    return riv_arr_in


#   function to find shelf
def find_shlf_edge(topo_lst, shlf_edge_depth = -175):
    #   find the first cell with elevation higher than the shlf_edge_depth from inversed list
    last_shelf = next(x[0] for x in enumerate(topo_lst[::-1]) if x[1] > shlf_edge_depth)
    shlf_idx = len(topo_lst) - last_shelf - 1
    return shlf_idx, topo_lst[shlf_idx]

#   Function that finds the FOS (foot of the contintental slope), based on a paper
def find_FOS(topo_lst, avg = True):
    
    #topo_lst = lst_topo_henry
    #topo_lst = lst_topo_other
    #topo_lst = final_topo
    #topo_lst = final_topo
    #out_dir = offshore_dir
    #cst_type = 'avg_avg'
    #avg= True
    
    #   clean the lists - if the values are = non_val then change to NaN not to screw up the calculation of averages
    arr_topo = np.array(topo_lst)
    #   distinguish between the cases when all the topographical profiles are supplied vs. when only one topo list is
    if avg and len(arr_topo.shape) > 1:
        avg_topo_lst = np.nanmean(arr_topo, axis = 0)
    elif avg:
        avg_topo_lst = arr_topo
    else:
        avg_topo_lst = topo_lst
    #x_axis = np.linspace(0.0, 200.0, 401) 
    #x_axis = np.linspace(0.0, 200.0, len(topo_lst)) 
    x_axis = np.linspace(0.0, 200.0, int((len(topo_lst) / 2 + 1))) 
    
    #   calculate the gradient and second derivative
    #grad = np.gradient(avg_topo_lst[400:])
    grad = np.gradient(avg_topo_lst[int((len(topo_lst) - 1) / 2) :])
    #y_spl = UnivariateSpline(x_axis, avg_topo[400:], s = 0, k = 4)
    #y_spl_2d = y_spl.derivative(n = 2)
    #x_range = np.linspace(x_axis[0], x_axis[-1], 401)
    grad2 = np.gradient(grad) 
    
    loc_max = argrelextrema(grad2, np.greater)    
    loc_min = argrelextrema(grad2, np.less)    
    
    loc_minmax = np.concatenate((loc_max[0], loc_min[0])) # we still need to do this unfortunatly.
    loc_minmax.sort()
    
    loc_max_plt, loc_min_plt, loc_minmax_plt1, loc_minmax_plt2 = [], [], [], []

    for a in range(loc_max[0].shape[0]):
        #loc_max_plt.append([x_axis[loc_max[0][a]], avg_topo_lst[400 + loc_max[0][a]]])
        loc_max_plt.append([x_axis[loc_max[0][a]], avg_topo_lst[int((len(topo_lst) - 1) / 2) + loc_max[0][a]]])
    
    for b in range(loc_min[0].shape[0]):
        #loc_min_plt.append([x_axis[loc_min[0][b]], avg_topo_lst[400 + loc_min[0][b]]])
        loc_min_plt.append([x_axis[loc_min[0][b]], avg_topo_lst[int((len(topo_lst) - 1) / 2) + loc_min[0][b]]])
        
    for c in range(loc_minmax.shape[0]):
        #loc_minmax_plt1.append([avg_topo[loc_minmax[c]], avg_topo[loc_minmax[c]]])
        #loc_minmax_plt1.append(avg_topo_lst[400 + loc_minmax[c]])
        #loc_minmax_plt2.append(avg_topo_lst[400 + loc_minmax[c]])
        loc_minmax_plt1.append(avg_topo_lst[int((len(topo_lst) - 1) / 2) + loc_minmax[c]])
        loc_minmax_plt2.append(avg_topo_lst[int((len(topo_lst) - 1) / 2) + loc_minmax[c]])
    
    rdp_in1, rdp_in2 = [], []
    for i in range(len(loc_minmax_plt1)):
        rdp_in1.append([loc_minmax_plt1[i], loc_minmax_plt1[i]])
        rdp_in2.append([x_axis[loc_minmax[i]], loc_minmax_plt1[i]])
    rdp_points = rdp.rdp(rdp_in2, epsilon = 2)
    
    for j in range(len(rdp_points)):
        #   if it is the first point in the list start with coastal point - 0.0, 0.0
        if j == 0:
            slope_left = (rdp_points[j][1]) / (rdp_points[j][0])
            slope_right = (rdp_points[j + 1][1] - rdp_points[j][1]) / (rdp_points[j + 1][0] - rdp_points[j][0])
        #   if it is the last point than take the last point as 200.0, avg_topo[-1] 
        elif j == len(rdp_points) - 1:
            slope_left = (rdp_points[j - 1][1] - rdp_points[j][1]) / (rdp_points[j - 1][0] - rdp_points[j][0])
            slope_right = (rdp_points[j][1] - avg_topo_lst[-1]) / (200.0 - rdp_points[j][0])            
        #   otherwise just take the elements before and after the given point
        else:
            slope_left = (rdp_points[j - 1][1] - rdp_points[j][1]) / (rdp_points[j - 1][0] - rdp_points[j][0])
            slope_right = (rdp_points[j + 1][1] - rdp_points[j][1]) / (rdp_points[j + 1][0] - rdp_points[j][0])
            
        rdp_points[j].append(slope_left)
        rdp_points[j].append(slope_right)
        
        #print j, rdp_points[j], round(((slope_left) + (slope_right)) / 2., 2) # slope_left, slope_right

    #   reset the sb_point and fos_point
    sb_point, fos_point = None, None

    #   loop through the points to identify the shelf break (sb_point) sand the foot of the slope (fos_point)
    #   skipping the first point in the list because that one is almost at the coastline..
    for g in range(1, len(rdp_points) - 1):
        #   get all the different values form the list
        pt_dist, pt_depth, pt_slope_left, pt_slope_right = rdp_points[g][0], rdp_points[g][1], rdp_points[g][2], rdp_points[g][3]
        print(g, pt_dist, pt_depth, pt_slope_left, pt_slope_right)
        #   first find the sb_point, it should be above -200m bsl. and the slope on the right from the point
        #   should be higher than the slope on the left from the point
        if pt_depth > -200.0 and abs(pt_slope_left) < abs(pt_slope_right):
            sb_point = [pt_dist, pt_depth]
        #   the fos_point should be deeper than -200m bsl, the absolute slope gradient on the left should be higher
        #   than on the right, and the slope on the right is either positive or less than 25% than the absolute value
        #   of the slope on the left
        elif pt_depth < -1000.0 and abs(pt_slope_left) > abs(pt_slope_right):
            if pt_slope_right > 0 or round(abs(pt_slope_right)) <= round((75 * abs(pt_slope_left)) / 100.):
                #   check if it is just one peak (concave hull) in the profile, do this by checking the slope and 
                #   deptg(elev) of the following part of the topographical profile
                if rdp_points[g + 1][1] > pt_depth and abs(50 * rdp_points[g + 1][3]) / 100. > abs(pt_slope_left):
                    continue
                else:
                    fos_point = [pt_dist, pt_depth]
                    break
            else:
                pass
            
    #   return the points - if found
    try:
        return sb_point, fos_point
    except UnboundLocalError:
        return [None, None]


