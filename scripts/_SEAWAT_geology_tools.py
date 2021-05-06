# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 09:39:55 2018

@author: daniel
"""

import random
import numpy as np
import math
import itertools



def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

"""
line1 = (line_A_lst[j], line_B_lst[j])
line2 = line_thk_end
"""

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

""" 
    Function that calculates mean and stdev Hk values based on dataset input (GLHYMPS 1 and 2)
""" 
def get_avg_hk_vals(hk_aqf_lst, hk_aqt_lst, k_soil_lst):
    #   check that the glhymps 2 layer is composed of non-nan values
    glh_2_check = [i for i in hk_aqf_lst if not math.isnan(i)]
    #   if there are only non values take the k soil as values for aquifer     
    if glh_2_check == []:
        hk_aqf_lst_out = [round(i, 4) for i in k_soil_lst]
    else:
        hk_aqf_lst_out = [round(i * 3600 *24, 4) for i in hk_aqf_lst]
    #   recalculate the values based on the GLHYMPS 2 formula
    hk_aqt_lst_out = [round(i, 4) for i in hk_aqt_lst]
    #   replace the negative values - sometimes they are there for some reason, also the totaly unrealistic huge values, limit to 100. m/d
    for x in range(len(hk_aqf_lst_out)):
        if hk_aqf_lst_out[x] < 0.:
           hk_aqf_lst_out[x] = np.mean([i for i in hk_aqf_lst_out if i > 0])
        try:
            if hk_aqt_lst_out[x] < 0.:
               hk_aqt_lst_out[x] = np.mean([i for i in hk_aqt_lst_out if i > 0])
        #   happens when the hk_vals_bot list is shorter than the top one, not sure why it happens sometimes
        #   if it happens then just skip the index in the loop, go to the next one 
        except IndexError:
            pass
    #   calculate the statistics for both top and bottom aquifer layers
    hk_aqf_mean = np.nanmean(hk_aqf_lst_out, axis=0)
    hk_aqf_std = np.nanstd(hk_aqf_lst_out, axis=0)
    hk_aqt_mean = np.nanmean(hk_aqt_lst_out, axis=0)
    hk_aqt_std = np.nanstd(hk_aqt_lst_out, axis=0)    
    #   limit the stdev value to 50% of the mean?
    if hk_aqt_std > abs(hk_aqt_mean):
        hk_aqt_std = abs(hk_aqt_mean) / 2.
    if hk_aqf_std > abs(hk_aqf_mean):
        hk_aqf_std = abs(hk_aqf_mean) / 2.    
    #   return the final values
    return hk_aqf_mean, hk_aqf_std, hk_aqt_mean, hk_aqt_std
    
    
    
    
    
    
    
    
    
    
    

"""
model_obj = model
inland_aqf_lrs = inland_aqf_lrs
sed_type = 'small'
#sed_flux = 'medium'
save_dir = topo_sc_dir
summary_save_dir = geo_summary_csv_dir

glh_1_mu, glh_1_std = glhymps_bot_param_ln[0][0], glhymps_bot_param_ln[1][0]
glh_2_mu, glh_2_std = glhymps_top_param_ln[0][0], glhymps_top_param_ln[1][0]
rand_seed_in = rand_seed

const_geo_hk_vals = False
glh_1_val = 0.1
glh_2_val = 10.
clay_val = 0.0001
rand_seed_in = 1

p_fact = 0.5
off_lay_thk_ratio = mud_pct
lay_pres_y1 = y_val           
clay_cap_shelf = True       #   insert the clay capping layer on top of the continental shelf/slope, the same reworking
clay_cap_slope = True       #   parameter lay_pres_y1 will apply to these layers as well
clay_cap_thk = 20.    


model, rand_seed, inland_aqf_lrs, p_fact, off_lay_thk_ratio, sed_type, lay_pres_y1, clay_cap_thk,\
                                                           off_lay_start, 'low', topo_sc_dir, const_geo_hk_vals, clay_cap_shelf, clay_cap_slope

mud_shlf_pct, mud_slp_pct
"""
 
#def create_geology_profile(model_obj, rand_seed_in, inland_aqf_lrs, p_fact, off_lay_thk_ratio, sed_type, lay_pres_y1, clay_cap_shelf_thk, clay_cap_slope_thk,\
#                           off_lay_start, sed_flux, save_dir, summary_save_dir, const_geo_hk_vals, clay_cap_shelf, clay_cap_slope, figname):           

"""
model_obj = model
rand_seed_in = rand_seed
inland_aqf_lrs
p_fact
off_lay_thk_ratio = mud_pct
sed_type
lay_pres_y1
clay_cap_shelf_thk
clay_cap_slope_thk
off_lay_start
sed_flux
save_dir = topo_sc_dir
summary_save_dir = geo_summary_csv_dir
const_geo_hk_vals
clay_cap_shelf
clay_cap_slope
figname



#   define all variables
#rand_seed = 1                   #   randomization seed, for reproductive purposes
laytyp_val = 0                  #   MODFLOW layer type, 0 = confined
sand_pct = 70                   #   sand percentage of the sediment volume (Maria)
clay_pct = 30                   #   clay (mud) percentage of the sediment volume (Maria)
n_aqt_in = 2                    #   number of aquitard layers in the inland part of the domain
aqt_in_x1 = -1.0                #   
n_aqt_off = 3                   #   number of aquitard layers in the offshore part of the domain
aqt_off_x0 = 1.0
soil_thk = 10.0                 #   
aq_vals_smooth_hor = False      #
top_soil = False                #   presence of top soil layer in the upper part of the inland domain
top_offshore = True             #   presence of top aquitard clay (mud) layer on top of the offshore part of the domain
rand_aqt_inland = True          #   
rand_aqt_offshore = True        #
end_at_cst = False              #
fos_point = None            
"""





"""
rand_seed_in = 837
n_aqf_aqt_lrs = 2
p_fact = 0.5
off_lay_thk_ratio = 50
mud_pct = 50
erosion_fact = 25.
clay_cap_shelf_thk = 10.
clay_cap_slope_thk = 10.
off_lay_start = 0.
sed_flux = 'low'
const_geo_hk_vals = False
clay_cap_shelf = True
clay_cap_slope = True
hk_aqf_mean = 10.
hk_aqf_std = 2.5
hk_aqt_mean = 0.01
hk_aqt_std = 0.005
hk_clay_mean = 0.0001
hk_clay_std = 0.

import xarray as xr
xr_data = xr.open_mfdataset(r'g:\_modelbuilder\swampy\data\test_model10\temp_files\test_model10_INPUT.nc')
x_coord, y_coord = xr_data['x'].values.tolist(), xr_data['y'].values.tolist()
ibound_arr = xr_data['ibound_arr'].values
hk_arr = xr_data['Hk_arr'].values
top_elev = xr_data['top_elev_midcell'].values.tolist()
bot_elev = xr_data['bot_elev_midcell'].values.tolist()

cont_shlf_edge = xr_data['cont_shlf_edg'].values.tolist()

xr_data.close()

cont_shlf_edge = sb_pt

x_lst = x_coord
y_lst = y_coord
x_shlf = x_shelf_edge

mid_y_elev = y_lst

#   soilgrids soil thickness - set to None, so it will create a list of zeros
soilgrids_thk = None


dc = 100.
dl = 10.
"""

def create_geology_profile(rand_seed_in, n_aqf_aqt_lrs, p_fact, off_lay_thk_ratio, mud_pct, erosion_fact, clay_cap_shelf_thk, clay_cap_slope_thk,\
                           off_lay_start, sed_flux, const_geo_hk_vals, clay_cap_shelf, clay_cap_slope, hk_aqf_mean, hk_aqf_std, hk_aqt_mean, hk_aqt_std,\
                           cont_shlf_edge, hk_clay_mean, hk_clay_std, x_lst, y_lst, ibound_arr, top_elev, bot_elev, x_shlf, soilgrids_thk, dc, dl):  
           
    #   set the random seed - to be able to reproduce the same randomized geology
    random.seed(rand_seed_in)             

    #   create random thickness of AQF and AQT layers
    #       loop through the number of sand layers and calculate a thickness of each sand layer
    #       define the total fractions of the model domain based on mud percentage
    pct_glh_1 = int(round(mud_pct))
    pct_glh_2 = 100 - pct_glh_1
    #   split the percentages into the number of layers
    def chunk_lst(tot_sum, n_lays):
        dividers = sorted(random.sample(range(1, tot_sum), n_lays - 1))
        return [a - b for a, b in zip(dividers + [tot_sum], [0] + dividers)]
    #   get the individual layer fractions
    glh_1_layers = chunk_lst(pct_glh_1, n_aqf_aqt_lrs)
    glh_2_layers = chunk_lst(pct_glh_2, n_aqf_aqt_lrs)
    #   reset the list
    inland_aqf_lrs = []
    #   combine those into the final list
    for a in range(n_aqf_aqt_lrs):
        inland_aqf_lrs.append([int(glh_1_layers[a]), 'aqt'])
        inland_aqf_lrs.append([int(glh_2_layers[a]), 'aqf'])      
    #   sometimes a layer can have a negative thickness, if that is the case then remove 1% of each layer above and check if the 
    #   previously negative thickness layer is now positive, if not then repeat. Also include the case when the layer has 0 thickness
    for b in range(len(inland_aqf_lrs)):
        if inland_aqf_lrs[b][0] <= 0:
            while inland_aqf_lrs[b][0] <= 0:
                for c in range(b):
                    inland_aqf_lrs[c][0] -= 1
                    inland_aqf_lrs[b][0] += 1
    #   get the pct of the layers into the table and string
    lay_thk_pct = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    for h in range(len(inland_aqf_lrs)):
        lay_thk_pct[h] = inland_aqf_lrs[h][0]
    lay_thk_str = ''
    for g in range(len(lay_thk_pct)):
        lay_thk_str += str(lay_thk_pct[g]) + ','
    
    #   find the coastal index - that will determine the limit between the inland and offshore domain
    x_st = round(x_lst[0] + (x_lst[0] - x_lst[1]) / 2., 3)
    x_end = round(x_lst[-1] + (x_lst[0] - x_lst[1]) / 2., 3)
    cst_offset_idx_clip =  int(abs(round(x_st, 1) * 1000. / dc) - 1)  
    cst_offset_val = next((x for x in top_elev[:int(abs(x_st) * 10)] if x < -10.0), None)
    if not cst_offset_val:
        cst_offset_val = next((x for x in top_elev[int(abs(x_st) * 10):] if x < 0.0), None)
        cst_offset_idx = int(abs(x_st) * 10) + top_elev[int(abs(x_st) * 10):].index(cst_offset_val) - 1
    else:
        cst_offset_idx = top_elev.index(cst_offset_val) - 1
    cst_offset_plot = round(x_st + cst_offset_idx / (1000. / dc), 2) #  - to get the distance in km    

    #   take into account the soil layer - if there is soil layer thickness from SOILGRIDS, default is None and assigns 0m thickness
    if soilgrids_thk is None:
        soilgrids_thk_lst = [0.] * cst_offset_idx_clip
        soilgrids_thk_mean = 0.
    else:
        #   trim the soilgrids and replace all potential NaN values with the mean value of all the other thicknesses
        soilgrids_thk = soilgrids_thk[: cst_offset_idx_clip]  
        soilgrids_thk_lst = [round(p / 100., 0) for p in soilgrids_thk if p > -1]
        soilgrids_thk_mean = round(np.nanmean(soilgrids_thk_lst, axis=0), 0)
        soilgrids_thk_lst = [soilgrids_thk_mean if math.isnan(p) else p for p in soilgrids_thk_lst]            
    
    #   assign constant HK values
    glh_1_val = round(hk_aqt_mean, 4)
    glh_2_val = round(hk_aqf_mean, 4)
    clay_val = hk_clay_mean
    
    #   create the HK_ARR array that will be filled in the next steps
    hk_arr = np.zeros([ibound_arr.shape[0], 1, ibound_arr.shape[-1]])
    
    """                Next, create the inland part of the domain                   """       
    #   There are two different scenarios on how to create the stratigraphy in the inland part of the domain
    #       1) Based on the SOILGRIDS thickness (soilgrids_thk_lst) fill the upper part of the model_obj domnain
    #          with the GLHYMPS 2.0 values and the whole rest of the model_obj domain located below thet upper 
    #          part gets the HK values based on GLHMYPS 1.0 dataset. 
    #       2) This scenario is based on the assumption that more permeable layers are interlayed with less
    #          permeable ones - as mentiond in GLHYMPS 2.0 article, its values are usually 10 times higher than
    #          GLHYMPS 1.0 - that is why the SOILGRIDS thickness is repeated through the model_obj domain and assigned
    #          the GLHYMPS 1.0 or 2.0 values creating a sort of a zebra pattern.
    #       3) The last scenario takes into account the SOILGRIDS thickness and fills the upper aquifer part 
    #          with GLHYMPS 2.0 values. However, the rest of the aquifer is split into layers with either 
    #          GLHYMPS 1.0 or 2.0 based on an input list that defines the thickness and GLHYMPS type of each layer

    #   create a list where the glhymps 1 and glhymps 2 layers and respective layer indexes for each column will be stored
    #   only for inland part of the model domain.
    inland_lrs_lst = []

    #   calculate the index of the coastline based on the x start and the offset of the topography
    cst_idx_geo = int(abs(x_st - cst_offset_plot) * 10)
    
    for r in range(cst_idx_geo):
        try:
            #   calculate the number of layers that are part of the upper aquifer
            col_top_lay_n = int(round(soilgrids_thk_lst[r] / dl))
        #   might be that the soilgrids list is shorter than the coastline, in that case take the last value
        except IndexError:
            col_top_lay_n = int(round(soilgrids_thk_lst[-1] / dl))
        #   fill the upper part of the aquifer in any case (same for both scenarios), the values are randomized
        #   based on the statistics of the GLHYMPS 2.0 list calculated in the previous step
        ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, r].tolist()) if x == 1]
        lrs_lst = []
                
        if abs(glh_2_val) > abs(glh_1_val):
            lrs_lst.append(['aqf', ibound_act_lay_idxs[: col_top_lay_n]])
        else:            
            lrs_lst.append(['aqt', ibound_act_lay_idxs[: col_top_lay_n]]) 
            
        for s in range(col_top_lay_n):
            #   try except - catch the IndexError in case the SOILGRIDS thickness is larger than the active thickness of the IBOUND array in the current column
            try:
                if not const_geo_hk_vals:
                    hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                    if hk_cell_val > hk_aqf_mean / 10.:
                        hk_arr[ibound_act_lay_idxs[s], 0, r] = hk_cell_val
                    else:
                        hk_arr[ibound_act_lay_idxs[s], 0, r] = hk_aqf_mean / 10.
                else:
                    hk_arr[ibound_act_lay_idxs[s], 0, r] = glh_2_val
            except IndexError:
                pass
    
        #   check if there are any ibound_act_lay_idxs indexes below the upper aquifer part
        if len(ibound_act_lay_idxs) > col_top_lay_n:
            #   split the active ibound cells in the column list into groups (geological layers) based on the % from the input list
            tot_cells = len(ibound_act_lay_idxs) - col_top_lay_n
            cell_cnt_lst = []
            geo_lay_thk_start = col_top_lay_n
            #   get the pct of thickness per model layer
            lay_thk_pct = round(1 / float(tot_cells) * 100., 0)
            #   decide the layer types in the rest of the model layers based on the majority of glhymps type in that layer
            lay_glh_lst = []

            # in very deep systems the pct can be equal to 0, in that case round to one more decimal
            if lay_thk_pct != 0.0:
                lay_thk_pct = math.ceil(1 / float(tot_cells) * 100.)
                #   create a list of 100 values (1 or 2 depending on the glhymps type)
                for i in range(len(inland_aqf_lrs)):
                    glh_pct = inland_aqf_lrs[i][0]
                    glh_type = inland_aqf_lrs[i][1]   
                    #   adapt the number of cells to be assigned
                    new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))
                    for j in range(new_cell_n):
                        if glh_type == 'aqt':
                            lay_glh_lst.append(1)
                        else:
                            lay_glh_lst.append(2)                        
                lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]                    
            else:                
                #   create a list of 100 values (1 or 2 depending on the glhymps type)
                for i in range(len(inland_aqf_lrs)):
                    glh_pct = inland_aqf_lrs[i][0]
                    glh_type = inland_aqf_lrs[i][1]  
                    #   adapt the number of cells to be assigned
                    new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))                    
                    for j in range(new_cell_n):
                        if glh_type == 'aqt':
                            lay_glh_lst.append(1)
                        else:
                            lay_glh_lst.append(2)                                       
                lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]
                
            if lay_glhymps_lst != [[]] and [i for i in lay_glhymps_lst if i != []] != []:
                last_non_empty = [z[0] for z in lay_glhymps_lst if z != []][-1]                
            else:
                glh_type_tofill = max([sublist for sublist in inland_aqf_lrs])[1]
                if glh_type_tofill == 'aqf': 
                    last_non_empty = 2
                elif glh_type_tofill == 'aqt': 
                    last_non_empty = 1            
            
            for x in range(len(lay_glhymps_lst)):
                if lay_glhymps_lst[x] == []:
                    lay_glhymps_lst[x] = [last_non_empty]
            
            #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
            if len(lay_glhymps_lst) > tot_cells:
                lay_glhymps_lst = lay_glhymps_lst[:-1]

            #   go through the list and depending on the majority of values in each sublist decide which GLHYMPS value will be filled in
            for l in range(len(lay_glhymps_lst)):
                lay = lay_glhymps_lst[l]
                glh_typ = max(set(lay), key = lay.count)
                
                #   due to rounding of the  tot_cells above sometimes the ibound_act_lay is one index shorter
                try:
                    if not const_geo_hk_vals: 
                        if glh_typ == 1:
                            hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, r] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                        elif glh_typ == 2:
                            hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                            if hk_cell_val > hk_aqf_mean / 10.:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, r] = hk_cell_val
                            else:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, r] = hk_aqf_mean / 10.
                    else:
                        if glh_typ == 1:
                            hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, r] = glh_1_val
                        elif glh_typ == 2:
                            hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, r] = glh_2_val         
                
                except IndexError:
                    try:
                        if not const_geo_hk_vals: 
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l - 1], 0, r] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                            elif glh_typ == 2:
                                hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                if hk_cell_val > hk_aqt_mean / 10.:
                                    hk_arr[ibound_act_lay_idxs[col_top_lay_n + l - 1], 0, r] = hk_cell_val
                                else:
                                    hk_arr[ibound_act_lay_idxs[col_top_lay_n + l - 1], 0, r] = hk_aqt_mean / 10.
                        else:
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l - 1], 0, r] = glh_1_val
                            elif glh_typ == 2:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l - 1], 0, r] = glh_2_val                
                    except IndexError:
                        if not const_geo_hk_vals: 
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[-1], 0, r] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                            elif glh_typ == 2:
                                #hk_arr[ibound_act_lay_idxs[-1], 0, r] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)
                                hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                if hk_cell_val > hk_aqf_mean / 10.:
                                    hk_arr[ibound_act_lay_idxs[-1], 0, r] = hk_cell_val
                                else:
                                    hk_arr[ibound_act_lay_idxs[-1], 0, r] = hk_aqf_mean / 10.
                        else:
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[-1], 0, r] = glh_1_val
                            elif glh_typ == 2:
                                hk_arr[ibound_act_lay_idxs[-1], 0, r] = glh_2_val    

            glh_nr = 1
            glh_tx = 'aqt'
            to_lst = []

            #   go through the list and depending on the majority of values in each sublist decide which GLHYMPS value will be filled in
            for l in range(tot_cells):
                if glh_typ == glh_nr:
                    to_lst.append(ibound_act_lay_idxs[col_top_lay_n + l])
                else:
                    lrs_lst.append([glh_tx, to_lst])
                    if glh_nr == 1:
                        glh_nr = 2
                        glh_tx = 'aqf'
                    else:
                        glh_nr = 1
                        glh_tx = 'aqt'                    
                    to_lst = []

        inland_lrs_lst.append([r, lrs_lst])
                    
    #   define input into the next step for creating the offshore part of the model_obj domain
    try:
        tot_cells_cst = len([i for i, x in enumerate(ibound_arr[:, r].tolist()) if x == 1])
    except UnboundLocalError:
        r = cst_idx_geo - 1
        tot_cells_cst = len([i for i, x in enumerate(ibound_arr[:, r].tolist()) if x == 1])
        
    #lay_num = len(inland_aqf_lrs) + 1
    lay_thk = []
    #   lay_thk_1 is going to be repeated for all the layers because they all have same thickness
    try:
        lay_thk_1 = round((int(soilgrids_thk_lst[r] / dl) / float(tot_cells_cst)) * 100., 2) 
    except IndexError:
        lay_thk_1 = round((int(soilgrids_thk_lst[-1] / dl) / float(tot_cells_cst)) * 100., 2) 
    lay_thk.append([lay_thk_1, 'aqf'])
    for w in range(len(inland_aqf_lrs)):
        try:
            geo_lay_thk = (inland_aqf_lrs[w][0] * tot_cells) / 100
            lay_thk.append([round((geo_lay_thk / float(tot_cells_cst)) * 100., 2), inland_aqf_lrs[w][1]])
        # NameError: name 'tot_cells' is not defined
        except NameError: 
            lay_thk.append([0.0, inland_aqf_lrs[w][1]])
    lay_thk[-1][0] += round(100. - sum(i[0] for i in lay_thk), 2)

    """       Once the inland part of the domain is filled, create the offshore part         """
    #   The offshore part of the model_obj domain is conceptualized as the unconsolidated sediment part of the contintental
    #   shelf. Since there are quite a lot of differences around the coastline and it is impossible to model_obj each individual
    #   case. Therefore the classification by Maria is used to classify the global continental shelf types based on the 
    #   sediment grain size deposited and the shape of the chronologically and stratigraphically different layers. There can
    #   be a present/absent clay aquitard layer in between these individual layers. This is all specified by various classes 
    #   that are specified below.
    #
    #       1) The first class to be specified is the sediment type based on Marias research. This will have effect on the
    #          upper composition of each layer. The following types are specified for variable sed_type 
    #           
    #          a) 'large' sediment size means that the top of each layer will have a coarder (*10) HK values than supposed
    #          b) 'medium' sediment size means that there is no change in permeabilities
    #          c) 'small' sediment size means that there is a clay (mud) layer on top of each layer
    #
    #          The other vairable specified in this class is the sed_num_cells which determines the number of cells (in %) for 
    #          each layer that gets the sediment type property specified above. 
    #          
    #       2) Number of different sediment layers off_lay_num (specified as GLHYMPS 1.0 or 2.0) can:
    #
    #          a) be specified based on the inland_aqf_scenario is use_inland_aqf_scenario - True/False
    #               - if True then just follow the num_layers from the inland part of the domain
    #               - if False then specify the off_lay_num manually 
    #          b) another variable here is the thickness of each layer (in % of the verical domain space) off_aqf_lrs which can
    #             either be:
    #               - equal to inland_aqf_lrs
    #               - specified manuall (but then the connection at the coastline will look a bit strange..)
    #
    #       3) The last class of variables to be specified concerns the sediment flux type, which can be
    #
    #          a) 'high' sediment flux means that the different layers are more extended in the horizontal direction, meaning that
    #             the axis_angle is larger 
    #          b) 'medium' sediment flux translates into an average axis_angle that is lower than in case of the previous type
    #          c) 'low' sediment flux means that the layers are shorter and shorter in the x direction the younger they are. No
    #             angle is necessary here as the layers do not extend and 'slide down' around the continental slope
    #   
    #          axis_angle is the angle between the coastline (verical x = 0) and the line that passes through the end of the 
    #          continental shelf point and the ATE point at the x = 0 line. The intersection of this line and each of the 
    #          horizontal layers lines defines the inflection point of each sediment layer.
        
    #   get the shelf index in and also the last layer of the upper aquifer part in that column
    #shelf_edge_idx = top_elev.index(cont_shlf_edge[1]) 
    shelf_edge_idx = int(cont_shlf_edge[0])
    if shelf_edge_idx > ibound_arr.shape[-1]:
        shelf_edge_idx = ibound_arr.shape[-1] - 1
    shelf_edge_ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, shelf_edge_idx].tolist()) if x == 1]      

    #   create a list where the aqt and aqf layers and respective layer indexes for each column will be stored
    #   only for offshore part of the model domain, the inland part will or wont have its own clay layers formed later
    offshore_lrs_lst = []
    
    #   first create profiles if the sed influx is low low sediment influx means that the layers are going
    #   to be split based on the ratio specified above, and will spread till the continental slope
    if sed_flux == 'low': 
        #   loop through each model_obj domain column located in the offshore
        for a in range(cst_idx_geo, ibound_arr.shape[-1]):
            #   calculate the number of layers that are part of the upper aquifer, use the mean SOILGRIDS thickness value
            try:
                col_top_lay_n = int(soilgrids_thk_lst[r] / dl)
            except IndexError:
                col_top_lay_n = int(soilgrids_thk_lst[-1] / dl)             
            ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, a].tolist()) if x == 1]
            #   create a list for to store the layer indexes for each of the sediment layers
            lrs_lst = []               
            
            #   first check if there is a edge of the continental shelf point found
            #       if yes, then create a line between the point and the continental slope point
            if cont_shlf_edge is not None:                
                #   calculate the amount of layers at the continental shelf column
                tot_layers = len(inland_aqf_lrs)
                #   divide the area between the shelf edge and model_obj bottom into the tot_layers identical amount of cells
                #   find the limits for each of the geological layers 
                idx_1 = shelf_edge_ibound_act_lay_idxs[col_top_lay_n]                
                end_lay_idxs =  [idx_1]
                act_cell_lst = [i for i, x in enumerate(ibound_arr[idx_1, :].tolist()) if x == 1]
                end_col_idxs = [act_cell_lst[-1]]                
                for b in range(tot_layers - 1):                    
                    lay_idx = idx_1 + int(round((inland_aqf_lrs[b][0] * (ibound_arr.shape[0] - shelf_edge_ibound_act_lay_idxs[col_top_lay_n])) / 100, 0))              
                    end_lay_idxs.append(lay_idx)
                    act_cell_lst = [i for i, x in enumerate(ibound_arr[lay_idx - 1, :].tolist()) if x == 1]
                    if len(act_cell_lst) > 0:
                        end_col_idxs.append(act_cell_lst[-1])
                        idx_1 = lay_idx
                #   append the last active layer and column as the end of the last layer
                end_lay_idxs.append(ibound_arr.shape[0])
                end_col_idxs.append(int(ibound_arr.shape[-1]) - 1)
 
                #   check if there are any ibound_act_lay_idxs indexes below the upper aquifer part
                #   fill in the part till the shelf edge with the constant thickness, after that point fill in straight line
                if a < shelf_edge_idx:                    
                    #   the upper part of the offshore aquifer domain
                    ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, a].tolist()) if x == 1]
                    """     add if aqf (or glhymps_top in old script version) is true or false     """                    
                    lrs_lst.append(['aqf', ibound_act_lay_idxs[:col_top_lay_n]]) 
                    for s in range(col_top_lay_n):
                        #   try except - catch the IndexError in case the SOILGRIDS thickness is larger than the active thickness
                        #   of the IBOUND array in the current column
                        try:
                            if not const_geo_hk_vals: 
                                #hk_arr[ibound_act_lay_idxs[s], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)
                                hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                if hk_cell_val > hk_aqf_mean / 10.:
                                    hk_arr[ibound_act_lay_idxs[s], 0, a] = hk_cell_val
                                else:
                                    hk_arr[ibound_act_lay_idxs[s], 0, a] = hk_aqf_mean / 10.
                            else:
                                hk_arr[ibound_act_lay_idxs[s], 0, a] = glh_2_val
                        except IndexError:
                            pass    
                        
                    #   split the active ibound cells in the column list into groups (geological layers) based on the % from the input list
                    tot_cells = len(ibound_act_lay_idxs) - col_top_lay_n
                    #   get the pct of thickness per model layer
                    lay_thk_pct = round(1 / float(tot_cells) * 100., 0)
                    #   decide the layer types in the rest of the model layers based on the majority of glhymps type in that layer
                    lay_glh_lst = []

                    # in very deep systems the pct can be equal to 0, in that case round to one more decimal
                    if lay_thk_pct != 0.0:
                        lay_thk_pct = math.ceil(1 / float(tot_cells) * 100.)
                        #   create a list of 100 values (1 or 2 depending on the glhymps type)
                        for i in range(len(inland_aqf_lrs)):
                            glh_pct = inland_aqf_lrs[i][0]
                            glh_type = inland_aqf_lrs[i][1]        
                            #   adapt the number of cells to be assigned
                            new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))                            
                            for j in range(new_cell_n):
                                if glh_type == 'aqt':
                                    lay_glh_lst.append(1)
                                else:
                                    lay_glh_lst.append(2)                        
                        lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]                           
                    else:                        
                        #   create a list of 100 values (1 or 2 depending on the glhymps type)
                        for i in range(len(inland_aqf_lrs)):
                            glh_pct = inland_aqf_lrs[i][0]
                            glh_type = inland_aqf_lrs[i][1]  
                            #   adapt the number of cells to be assigned
                            new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))
                            for j in range(new_cell_n):
                                if glh_type == 'aqt':
                                    lay_glh_lst.append(1)
                                else:
                                    lay_glh_lst.append(2)                                                
                        lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]
                        
                    if lay_glhymps_lst != [[]] and [i for i in lay_glhymps_lst if i != []] != []:
                        last_non_empty = [z[0] for z in lay_glhymps_lst if z != []][-1]                        
                    else:
                        glh_type_tofill = max([sublist for sublist in inland_aqf_lrs])[1]
                        if glh_type_tofill == 'aqf': 
                            last_non_empty = 2
                        elif glh_type_tofill == 'aqt': 
                            last_non_empty = 1            
                    
                    for x in range(len(lay_glhymps_lst)):
                        if lay_glhymps_lst[x] == []:
                            lay_glhymps_lst[x] = [last_non_empty]
                    #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
                    if len(lay_glhymps_lst) > tot_cells:
                        lay_glhymps_lst = lay_glhymps_lst[:-1]
                    
                    #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
                    if len(lay_glhymps_lst) > tot_cells:
                        diff_idx = len(lay_glhymps_lst) - tot_cells
                        rem_idxs = len(lay_glhymps_lst) / (diff_idx + 1)
                        idx_rem = int(rem_idxs)
                        for y in range(diff_idx):
                            try:
                                del lay_glhymps_lst[idx_rem]
                                idx_rem += int(rem_idxs)
                            except IndexError:
                                pass
                            
                    #   if some layers are missing (e.g. % is 6.0) then find out how many and at equidistant locations in the list repeat the glhymps sublist
                    elif len(lay_glhymps_lst) < tot_cells:
                        miss_lay_n = tot_cells - len(lay_glhymps_lst)
                        ins_idxs = int(round(len(lay_glhymps_lst) / (miss_lay_n + 1)))
                        idx_ins = ins_idxs
                        for x in range(miss_lay_n):
                            lay_glhymps_lst.insert(idx_ins, lay_glhymps_lst[idx_ins])
                            idx_ins += ins_idxs

                    #   loop through the list and assign the values to the lrs_lst
                    glh_nr = 1
                    glh_tx = 'aqt'
                    to_lst = []

                    #   go through the list and depending on the majority of values in each sublist decide which GLHYMPS value will be filled in
                    for l in range(tot_cells):
                        lay = lay_glhymps_lst[l]
                        glh_typ = max(set(lay), key = lay.count)
                        #print glh_typ
                        if not const_geo_hk_vals: 
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                            elif glh_typ == 2:                     
                                hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                if hk_cell_val > hk_aqf_mean / 10.:
                                    hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, a] = hk_cell_val
                                else:
                                    hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, a] = hk_aqf_mean / 10.
                        else:
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, a] = glh_1_val
                            elif glh_typ == 2:
                                hk_arr[ibound_act_lay_idxs[col_top_lay_n + l], 0, a] = glh_2_val  
                           
                        if glh_typ == glh_nr:
                            to_lst.append(ibound_act_lay_idxs[col_top_lay_n + l])
                        else:
                            lrs_lst.append([glh_tx, to_lst])
                            if glh_nr == 1:
                                glh_nr = 2
                                glh_tx = 'aqf'
                            else:
                                glh_nr = 1
                                glh_tx = 'aqt'                    
                            to_lst = []

                #   another case
                elif a < end_col_idxs[0]:
                    #   find the upper part of the model column with glhymps 2 values
                    lay_idx_upp = ibound_act_lay_idxs.index(end_lay_idxs[0])
                    tot_cells = len(ibound_act_lay_idxs) - lay_idx_upp                  
                    #   get the pct of thickness per model layer
                    lay_thk_pct = round(1 / float(tot_cells) * 100., 0)
                    #   decide the layer types in the rest of the model layers based on the majority of glhymps type in that layer
                    lay_glh_lst = []                    

                    # in very deep systems the pct can be equal to 0, in that case round to one more decimal
                    if lay_thk_pct != 0.0:
                        lay_thk_pct = math.ceil(1 / float(tot_cells) * 100.)
                        #   create a list of 100 values (1 or 2 depending on the glhymps type)
                        for i in range(len(inland_aqf_lrs)):
                            glh_pct = inland_aqf_lrs[i][0]
                            glh_type = inland_aqf_lrs[i][1]   
                            #   adapt the number of cells to be assigned
                            new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))
                            for j in range(new_cell_n):
                                if glh_type == 'aqt':
                                    lay_glh_lst.append(1)
                                else:
                                    lay_glh_lst.append(2)                                                    
                        lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]
                    else:                      
                        #   create a list of 100 values (1 or 2 depending on the glhymps type)
                        for i in range(len(inland_aqf_lrs)):
                            glh_pct = inland_aqf_lrs[i][0]
                            glh_type = inland_aqf_lrs[i][1]  
                            #   adapt the number of cells to be assigned
                            new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))
                            for j in range(new_cell_n):
                                if glh_type == 'aqt':
                                    lay_glh_lst.append(1)
                                else:
                                    lay_glh_lst.append(2)                        
                        lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]
                        
                    if lay_glhymps_lst != [[]] and [i for i in lay_glhymps_lst if i != []] != []:
                        last_non_empty = [z[0] for z in lay_glhymps_lst if z != []][-1]
                    else:
                        glh_type_tofill = max([sublist for sublist in inland_aqf_lrs])[1]
                        if glh_type_tofill == 'aqf': 
                            last_non_empty = 2
                        elif glh_type_tofill == 'aqt': 
                            last_non_empty = 1            
                    
                    for x in range(len(lay_glhymps_lst)):
                        if lay_glhymps_lst[x] == []:
                            lay_glhymps_lst[x] = [last_non_empty]
                    
                    #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
                    if len(lay_glhymps_lst) > tot_cells:
                        lay_glhymps_lst = lay_glhymps_lst[:-1]
                    
                    #   if some layers are missing (e.g. % is 6.0) then find out how many and at equidistant locations in the list repeat the glhymps sublist
                    if len(lay_glh_lst) < 100 and len(lay_glh_lst) > 0:
                        lay_glh_lst = list(itertools.chain.from_iterable((itertools.repeat(i, 2) for i in lay_glh_lst)))  
                    
                    #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
                    if len(lay_glhymps_lst) > tot_cells:
                        diff_idx = len(lay_glhymps_lst) - tot_cells
                        rem_idxs = len(lay_glhymps_lst) / (diff_idx + 1)
                        idx_rem = int(rem_idxs)
                        for y in range(diff_idx):
                            try:
                                del lay_glhymps_lst[idx_rem]
                                idx_rem += int(rem_idxs)
                            except IndexError:
                                pass
                            
                    #   if some layers are missing (e.g. % is 6.0) then find out how many and at equidistant locations in the list repeat the glhymps sublist
                    elif len(lay_glhymps_lst) < tot_cells:
                        miss_lay_n = tot_cells - len(lay_glhymps_lst)
                        ins_idxs = len(lay_glhymps_lst) / (miss_lay_n + 1)
                        idx_ins = ins_idxs
                        try:
                            for x in range(miss_lay_n):
                                lay_glhymps_lst.insert(idx_ins, lay_glhymps_lst[idx_ins])
                                idx_ins += ins_idxs
                        # in case the miss_lay_n is not an integer (TypeError: list indices must be integers or slices, not float)
                        except TypeError:
                            for x in range(miss_lay_n):
                                lay_glhymps_lst.insert(round(idx_ins), lay_glhymps_lst[round(idx_ins)])
                                idx_ins += ins_idxs
                                
                    #   create a list of final glhymps values and change the first layers to glhymps 2 if necessary
                    final_glh_lst = [max(set(sublist), key = sublist.count) for sublist in lay_glhymps_lst]
                    for x in range(lay_idx_upp):
                        final_glh_lst.insert(0, 2)
                    
                    #   go through the list and depending on the majority of values in each sublist decide which GLHYMPS value will be filled in
                    try:
                        for l in range(len(final_glh_lst)):
                            glh_typ = final_glh_lst[l]
                            if not const_geo_hk_vals: 
                                if glh_typ == 1:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                                elif glh_typ == 2:
                                    #hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)                            
                                    hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                    if hk_cell_val > hk_aqf_mean / 10.:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_cell_val
                                    else:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_aqf_mean / 10.
                            else:
                                if glh_typ == 1:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_1_val
                                elif glh_typ == 2:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_2_val  
                    except IndexError:
                        for l in range(len(ibound_act_lay_idxs)):
                            glh_typ = final_glh_lst[l]
                            if not const_geo_hk_vals: 
                                if glh_typ == 1:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                                elif glh_typ == 2:
                                    #hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)      
                                    hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                    if hk_cell_val > hk_aqf_mean / 10.:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_cell_val
                                    else:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_aqf_mean / 10.                                    
                            else:
                                if glh_typ == 1:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_1_val
                                elif glh_typ == 2:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_2_val  

                    #   loop through the list and assign the values to the lrs_lst
                    glh_nr = final_glh_lst[0]
                    if glh_nr == 1:
                        glh_tx = 'aqt'
                    else:
                        glh_tx = 'aqf'
                    to_lst = []
                    try:
                        for y in range(len(final_glh_lst)):
                            if final_glh_lst[y] == glh_nr:
                                to_lst.append(ibound_act_lay_idxs[y])
                            else:
                                lrs_lst.append([glh_tx, to_lst])
                                if glh_nr == 1:
                                    glh_nr = 2
                                    glh_tx = 'aqf'
                                else:
                                    glh_nr = 1
                                    glh_tx = 'aqt'                    
                                to_lst = []
                    except IndexError:
                        for y in range(len(ibound_act_lay_idxs)):
                            if final_glh_lst[y] == glh_nr:
                                to_lst.append(ibound_act_lay_idxs[y])
                            else:
                                lrs_lst.append([glh_tx, to_lst])
                                if glh_nr == 1:
                                    glh_nr = 2
                                    glh_tx = 'aqf'
                                else:
                                    glh_nr = 1
                                    glh_tx = 'aqt'                    
                                to_lst = []                        

                    #   get the upp layer limit of the top glhymps layer 
                    first_lay = 0
                    try:
                        for x in range(len(final_glh_lst)):
                            if final_glh_lst[x] == 1:
                                pass
                            else:
                                first_lay = ibound_act_lay_idxs[x - 1]
                                break    
                    except IndexError:
                        for x in range(len(ibound_act_lay_idxs)):
                            if final_glh_lst[x] == 1:
                                pass
                            else:
                                first_lay = ibound_act_lay_idxs[x - 1]
                                break                           
                        
                    first_col = a
                   
                #   also if the column is after the last end of a glhymps layer adjust the procedure
                elif a > end_col_idxs[-2]:                    
                    #   try except in case there are no active cells in the ibound column (happens in the end of the model IBOUND array)
                    try:
                        #   1) decide what layers can be active in the given model column
                        col_counter = 0
                        for b in range(1, len(end_col_idxs)):
                            if a > end_col_idxs[b]:
                                col_counter += 1                                        
                
                        #       get the glhymps type of the top model layers from the inland_aqf_lst
                        glh_typ_top = inland_aqf_lrs[col_counter][1]
                        if glh_typ_top == 'aqf':
                            glh_typ_top_type = 2
                        else:
                            glh_typ_top_type = 1     
                            
                        try:
                            for l in range(len(final_glh_lst)):
                                glh_typ = final_glh_lst[l]
                                if not const_geo_hk_vals: 
                                    if glh_typ == 1:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                                    elif glh_typ == 2:
                                        #hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)  
                                        hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                        if hk_cell_val > hk_aqf_mean / 10.:
                                            hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_cell_val
                                        else:
                                            hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_aqf_mean / 10.                                           
                                else:
                                    if glh_typ == 1:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_1_val
                                    elif glh_typ == 2:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_2_val  
                        except IndexError:
                            for l in range(len(ibound_act_lay_idxs)):
                                glh_typ = final_glh_lst[l]
                                if not const_geo_hk_vals: 
                                    if glh_typ == 1:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                                    elif glh_typ == 2:
                                        #hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)    
                                        hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                        if hk_cell_val > hk_aqf_mean / 10.:
                                            hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_cell_val
                                        else:
                                            hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_aqf_mean / 10.                                           
                                else:
                                    if glh_typ == 1:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_1_val
                                    elif glh_typ == 2:
                                        hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_2_val  

                        #   loop through the list and assign the values to the lrs_lst
                        for y in range(len(ibound_act_lay_idxs)):
                            if glh_typ_top_type == 1:
                                lrs_lst.append(['aqt', ibound_act_lay_idxs[:]])
                            else:
                                lrs_lst.append(['aqf', ibound_act_lay_idxs[:]])
                            
                    #   this happens at the end of contintental slope part of the domain
                    except (ZeroDivisionError, IndexError):
                        ibound_idx_grp = list(split(ibound_act_lay_idxs, 1))
                        for w in range(0, len(ibound_idx_grp)):
                            #   + 1 because the upper layer is already filled in by GLHYMPS 2.0
                            if not const_geo_hk_vals: 
                                if inland_aqf_lrs[w][1] == 'aqf':
                                    #   for every even layer fill in with GLHMYPS 2.0 (even is 0, 2, 4..)
                                    for t in range(len(ibound_idx_grp[w])):
                                        #hk_arr[ibound_idx_grp[w][t], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)
                                        hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                        if hk_cell_val > hk_aqf_mean / 10.:
                                            hk_arr[ibound_idx_grp[w][t], 0, a] = hk_cell_val
                                        else:
                                            hk_arr[ibound_idx_grp[w][t], 0, a] = hk_aqf_mean / 10.                                           
                                else:
                                    for t in range(len(ibound_idx_grp[w])):
                                        hk_arr[ibound_idx_grp[w][t], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8) 
                            else:
                                if inland_aqf_lrs[w][1] == 'aqf':
                                    #   for every even layer fill in with GLHMYPS 2.0 (even is 0, 2, 4..)
                                    for t in range(len(ibound_idx_grp[w])):
                                        hk_arr[ibound_idx_grp[w][t], 0, a] = glh_2_val
                                else:
                                    for t in range(len(ibound_idx_grp[w])):
                                        hk_arr[ibound_idx_grp[w][t], 0, a] = glh_1_val

                else:
                    #   1) decide what layers can be active in the given model column
                    col_counter = 0
                    for b in range(1, len(end_col_idxs)):
                        if a > end_col_idxs[b]:
                            col_counter += 1                                        
                    #       get the glhymps type of the top model layers from the inland_aqf_lst
                    glh_typ_top = inland_aqf_lrs[col_counter][1]
                    if glh_typ_top == 'aqf':
                        glh_typ_top_type = 2
                    else:
                        glh_typ_top_type = 1
                    #   at the beginning, check if we need to change the first_lay and first_col values

                    if 'final_glh_lst'  not in locals():
                        lay_idx_upp = ibound_act_lay_idxs.index(end_lay_idxs[0])
                        final_glh_lst = [max(set(sublist), key = sublist.count) for sublist in lay_glhymps_lst]
                        for x in range(lay_idx_upp):
                            final_glh_lst.insert(0, 2)

                    if a == end_col_idxs[col_counter] + 1:
                        first_lay = 0
                        #   make sure there are no remnants of the other glhymps type on top (usually one layer at the end of the stretch)
                        while final_glh_lst[0] != glh_typ_top_type:
                            final_glh_lst = final_glh_lst[1:]
                        for x in range(len(final_glh_lst)):
                            #print final_glh_lst[x]
                            if final_glh_lst[x] == glh_typ_top_type:
                                pass
                            else:
                                try:
                                    first_lay = ibound_act_lay_idxs[x + 1]
                                    break                    
                                except IndexError:
                                    pass
                        first_col = a

                    last_lay, last_col = end_lay_idxs[col_counter + 1], end_col_idxs[col_counter + 1]
                
                    #   calculate the cell step based on the distance between the end columns and end layers of the top layer
                    try:
                        cell_step = (last_col - first_col) / (last_lay - first_lay)          
                        #   based on the ascent decide how many cells will be added to the bottom layer of the top glhymps layer
                        add_lays = int(round((a - first_col) / float(cell_step), 0))
                    except (UnboundLocalError, ZeroDivisionError, NameError):
                        cell_step, add_lays, first_lay = 0, 0, 0

                    end_lay_new = first_lay + add_lays
                    
                    #   it can happen during the rounding up that a value is selected that doesnt exist in the actual list
                    try:
                        upp_lay_lim = ibound_act_lay_idxs.index(end_lay_new)
                    except ValueError:
                        upp_lay_lim = ibound_act_lay_idxs.index(min(ibound_act_lay_idxs, key = lambda x : abs(x - end_lay_new)))
                        
                    tot_cells = len(ibound_act_lay_idxs) - upp_lay_lim
                    
                    #   2) divide the column based on the % fractions from the inland_aqf_lst
                    #       get the pct of thickness per model layer
                    lay_thk_pct = round(1 / float(tot_cells) * 100., 0)
                    #       decide the layer types in the rest of the model layers based on the majority of glhymps type in that layer
                    lay_glh_lst = []                                

                    # in very deep systems the pct can be equal to 0, in that case round to one more decimal
                    if lay_thk_pct != 0.0:
                        lay_thk_pct = math.ceil(1 / float(tot_cells) * 100.)
                        #   create a list of 100 values (1 or 2 depending on the glhymps type)
                        for i in range(len(inland_aqf_lrs)):
                            glh_pct = inland_aqf_lrs[i][0]
                            glh_type = inland_aqf_lrs[i][1]   
                            #   adapt the number of cells to be assigned
                            new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))

                            for j in range(new_cell_n):
                                if glh_type == 'aqt':
                                    lay_glh_lst.append(1)
                                else:
                                    lay_glh_lst.append(2)                        
                        lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]                           
                    else:
                        #   create a list of 100 values (1 or 2 depending on the glhymps type)
                        for i in range(len(inland_aqf_lrs)):
                            glh_pct = inland_aqf_lrs[i][0]
                            glh_type = inland_aqf_lrs[i][1]                             
                            #   adapt the number of cells to be assigned
                            new_cell_n = int(round((tot_cells / 100) * glh_pct, 0))
                            for j in range(new_cell_n):
                                if glh_type == 'aqt':
                                    lay_glh_lst.append(1)
                                else:
                                    lay_glh_lst.append(2)                        
                        
                        lay_glhymps_lst = [lay_glh_lst[i:i + 1] for i in range(0, tot_cells, 1)]
                        
                    if lay_glhymps_lst != [[]] and [i for i in lay_glhymps_lst if i != []] != []:
                        last_non_empty = [z[0] for z in lay_glhymps_lst if z != []][-1]
                        
                    else:
                        glh_type_tofill = max([sublist for sublist in inland_aqf_lrs])[1]
                        if glh_type_tofill == 'aqf': 
                            last_non_empty = 2
                        elif glh_type_tofill == 'aqt': 
                            last_non_empty = 1            
                    
                    for x in range(len(lay_glhymps_lst)):
                        if lay_glhymps_lst[x] == []:
                            lay_glhymps_lst[x] = [last_non_empty]
                    
                    #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
                    if len(lay_glhymps_lst) > tot_cells:
                        lay_glhymps_lst = lay_glhymps_lst[:-1]
                                       
                    #   if some layers are missing (e.g. % is 6.0) then find out how many and at equidistant locations in the list repeat the glhymps sublist
                    if len(lay_glh_lst) < 100 and len(lay_glh_lst) > 0:
                        while len(lay_glh_lst) < 100:
                            lay_glh_lst = list(itertools.chain.from_iterable((itertools.repeat(i, 2) for i in lay_glh_lst)))  
                    
                    #   make sure the list above has the right amount of elements (in case the % is 33.0 for example it appends one extra value to reach length of 100)
                    if len(lay_glhymps_lst) > tot_cells:
                        while len(lay_glhymps_lst) > tot_cells:
                            diff_idx = len(lay_glhymps_lst) - tot_cells
                            rem_idxs = len(lay_glhymps_lst) / (diff_idx + 1)
                            idx_rem = int(rem_idxs)
                            for y in range(diff_idx):
                                try:
                                    del lay_glhymps_lst[idx_rem]
                                    idx_rem += int(rem_idxs)
                                except IndexError:
                                    pass
                            
                    #   if some layers are missing (e.g. % is 6.0) then find out how many and at equidistant locations in the list repeat the glhymps sublist
                    elif len(lay_glhymps_lst) < tot_cells:
                        while len(lay_glhymps_lst) < tot_cells: 
                            miss_lay_n = tot_cells - len(lay_glhymps_lst)
                            ins_idxs = len(lay_glhymps_lst) / (miss_lay_n + 1)
                            idx_ins = int(ins_idxs)
                            for x in range(miss_lay_n):
                                lay_glhymps_lst.insert(idx_ins, lay_glhymps_lst[idx_ins])
                                idx_ins += int(ins_idxs)

                    #   create a list of final glhymps values and change the first layers to glhymps 2 if necessary
                    final_glh_lst = [max(set(sublist), key = sublist.count) for sublist in lay_glhymps_lst]

                    for x in range(upp_lay_lim):
                        final_glh_lst.insert(0, glh_typ_top_type)

                    for l in range(len(final_glh_lst)):
                        glh_typ = final_glh_lst[l]
                        #print glh_typ
                        if not const_geo_hk_vals: 
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[l], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)
                            elif glh_typ == 2:
                                hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                if hk_cell_val > hk_aqf_mean / 10.:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_cell_val
                                else:
                                    hk_arr[ibound_act_lay_idxs[l], 0, a] = hk_aqf_mean / 10.          
                        else:
                            if glh_typ == 1:
                                hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_1_val
                            elif glh_typ == 2:
                                hk_arr[ibound_act_lay_idxs[l], 0, a] = glh_2_val  

                    #   loop through the list and assign the values to the lrs_lst
                    glh_nr = final_glh_lst[0]
                    if glh_nr == 1:
                        glh_tx = 'aqt'
                    else:
                        glh_tx = 'aqf'
                    to_lst = []
                    for y in range(len(final_glh_lst)):
                        if final_glh_lst[y] == glh_nr:
                            to_lst.append(ibound_act_lay_idxs[y])
                        else:
                            lrs_lst.append([glh_tx, to_lst])
                            if glh_nr == 1:
                                glh_nr = 2
                                glh_tx = 'aqf'
                            else:
                                glh_nr = 1
                                glh_tx = 'aqt'                    
                            to_lst = []  
                            
                offshore_lrs_lst.append([a, lrs_lst])


    ##  the implementation of medium and high sediment flux is the same, the only difference is the angle of past continental shelf edges
    elif sed_flux == 'medium' or sed_flux == 'high':    

        #   first calculate the average slopes of the cont. shelf and cont. slope, as the elevation difference between the shelf edge and the
        #   elevation at the coastline
        avg_sl_shelf = round(100 * (top_elev[cst_idx_geo] - top_elev[shelf_edge_idx]) / ((shelf_edge_idx - cst_idx_geo) * 100.), 2)
        avg_sl_slope = round(100 * (top_elev[shelf_edge_idx] - top_elev[-1]) / ((ibound_arr.shape[-1] - shelf_edge_idx) * 100.), 2)

        top = math.ceil(np.nanmax(top_elev) / dl) * dl  # top
        lay_elev = y_lst#.tolist()
        lay_elev = [i - dl / 2 for i in lay_elev]     
        lay_elev.insert(0, top)
        botm = lay_elev[1:]
        
        glhymps_2_top = False
        
        if sed_flux == 'medium':
            shelf_angle = 85.   #   this needs to be specified based on the sed_flux being either medium or high
        elif sed_flux == 'high':
            shelf_angle = 87.5
        
        """
        FIRST DEAL WITH THE AREA BETWEEN THE COASTLINE AND THE BEGINNING OF THE SHELF EDGES AREA
        """
        #   get the top points of each layer (not the top one..), different for each scenario
        cst_lay_pts = []
        
        #   calculate the thickness of the top glhymps_2 layer at the coastline, to be expanded into the offshore domain
        col_top_lay_n = int(round(soilgrids_thk_lst[-1] / dl))     
        #   get the active layer indexes at the coastline
        cst_ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, cst_offset_idx].tolist()) if x == 1]
        
        #   get the index of the shelf edge and also a list of active layers at that model column
        shelf_edge_idx = int(cont_shlf_edge[0])
        if shelf_edge_idx > ibound_arr.shape[-1]:
            shelf_edge_idx = ibound_arr.shape[-1] - 1
        shelf_edge_ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, shelf_edge_idx].tolist()) if x == 1]       
        
        #   divide the area between the shelf edge and model_obj bottom into the tot_layers identical amount of cells based on the ratio 
        tot_layers = len(inland_aqf_lrs)
        tot_cells = len(cst_ibound_act_lay_idxs[col_top_lay_n:])
        #   append the last layer of the top glh layer, if possible - can happen that the total thickness is lower than the glh top
        #   layer thickness, in that case append an empty list
        try:
            cst_lay_pts.append(cst_ibound_act_lay_idxs[col_top_lay_n])
        except IndexError:
            cst_lay_pts.append([cst_ibound_act_lay_idxs[-1]])
        cell_cnt_lst = []
        geo_lay_thk_start = col_top_lay_n
        
        #   if there is no glhyps_2_top layer defined then still continue the more permeable sediment into the offshore domain, but only
        #   till the last inland model layer where this sediment layer occurs 
        if glhymps_2_top is False:
            try:
                last_glh_2_top_lay = cst_ibound_act_lay_idxs[col_top_lay_n]
            except IndexError:
                last_glh_2_top_lay = cst_ibound_act_lay_idxs[-1]
        
        #   loop through the individual glhymps 1 and 2 layers and find the corresponding bottom model layers
        for q in range(tot_layers):
            geo_lay_thk = int(round((inland_aqf_lrs[q][0] * tot_cells) / 100, 0))   #   thickness of glhymps layer expressed in model layers
            #   check if the calculated thickness is 0 then just go to the next layer
            if geo_lay_thk == 0:
                pass
            else:
                if q == len(inland_aqf_lrs) - 1:#   len(inland_aqf_lrs) - 1 because of python counting..
                    pass
                else:
                    #   if the soilgrids layer is too thick and there are not enough cells left to fill with all the other glhymps layers then skip them
                    try:
                        cst_lay_pts.append(cst_ibound_act_lay_idxs[geo_lay_thk_start + geo_lay_thk])
                        geo_lay_thk_start += geo_lay_thk    
                    except IndexError:
                        pass
                        
        #   calculate the angle between the aquifer bottom and the perpendicular line at the shelf edge
        aqf_bot_y_diff = round(bot_elev[cst_idx_geo] - bot_elev[shelf_edge_idx], 2)
        aqf_bot_x_diff = round((shelf_edge_idx - cst_idx_geo) * 100., 2)
        #   calculate the necessary angles for estimating the position of the past shelf edges
        shelf_edge_bot_angle = 90. - round(math.degrees(math.atan(aqf_bot_y_diff / aqf_bot_x_diff)), 2)
        cst_angle = round(180. - shelf_angle - shelf_edge_bot_angle, 2)   # the missing 3rd angle
        #   also calculate the necessary distances and ratios of the model part where the past shelf edges are located
        len_shelf = len(shelf_edge_ibound_act_lay_idxs) * 10.
        sin_ratio = len_shelf / math.sin(math.radians(cst_angle))
        len_bot = round(sin_ratio * math.sin(math.radians(shelf_angle)), 2)
        len_top = round(sin_ratio * math.sin(math.radians(shelf_edge_bot_angle)), 2)
        
        #   split the len_top into tot_layers parts equally and get the [layer, column] tuple for each point
        #       first calculate the perpendicular distance of the edge from the shelf edge
        dist_from_edge = round(math.sin(math.radians(shelf_angle)) * len_top, 2)
        dist_in_cols = int(round(dist_from_edge / 100.))
        
        #   check that the dist in cols between the shelf edge and coast is lower, if not set it
        if dist_in_cols > shelf_edge_idx - cst_idx_geo:
            dist_in_cols = shelf_edge_idx - cst_idx_geo
        
        #   for each glhymps layer calculate the column position of its shelf edge 
        shelf_edges_col_idx = []
        pct_start = inland_aqf_lrs[0][0]
        for i in range(len(inland_aqf_lrs) - 1):
            dist_col = (dist_in_cols * pct_start) / 100.
            shelf_edges_col_idx.append(shelf_edge_idx - int(round(dist_col, 0)))
            pct_start += inland_aqf_lrs[i + 1][0]
        
        #   this changes if the glhymps 2 layer is extended over the whole shelf domain, lay top designates the upper model layer from which we start
        #   to calculate the thickness of each glhymps layer 
        if glhymps_2_top is True:
            lay_top = shelf_edge_ibound_act_lay_idxs[col_top_lay_n] 
        else:
            if last_glh_2_top_lay not in shelf_edge_ibound_act_lay_idxs:
                lay_top = shelf_edge_ibound_act_lay_idxs[0]
            else:
                lay_top = shelf_edge_ibound_act_lay_idxs[shelf_edge_ibound_act_lay_idxs.index(last_glh_2_top_lay)]
        #   same for the bottom extent in terms of model layers
        lay_bot = [i for i, x in enumerate(ibound_arr[:, shelf_edge_idx - dist_in_cols].tolist()) if x == 1][-1]   
        
        #   next, calculate the shelf edges layer bottoms in terms of model layers
        shelf_edges_lay_idx = []
        pct_start = inland_aqf_lrs[0][0]
        for i in range(len(inland_aqf_lrs) - 1):
            dist_lay = ((lay_bot - lay_top) * pct_start) / 100.
            shelf_edges_lay_idx.append(lay_top + int(round(dist_lay, 0)))
            pct_start += inland_aqf_lrs[i + 1][0]        
        
        #   find the end of the layer based on the slope of the continental slope
        line_A_lst, line_B_lst, cst_point_lst = [], [], []
        line_thk_end = ((0.0, bot_elev[cst_offset_idx]), (round(x_end - cst_offset_plot, 1), bot_elev[-1]))   
        
        for i in range(len(shelf_edges_col_idx)):
            x_coord = round(x_st - cst_offset_plot + shelf_edges_col_idx[i] / 10., 1)# + 0.05
            y_coord = top - shelf_edges_lay_idx[i] * 10.# + 5.
            #  try to calculate the y_coordinate at the coastline, in case the thickness is to shallow for all the layers to be stacked there
            #  assign the total bottom of the model as the top of these layers, they will then link with the shelf edges later on
            #y_coast = top - (cst_lay_pts[i + 1]) * 10 
            try:
                y_coast = top - (cst_lay_pts[i + 1]) * 10
            except IndexError:
                # TypeError: unsupported operand type(s) for -: 'float' and 'list' because clay_lay_pts = [[8]] (for example)
                try:
                    y_coast = top - (cst_lay_pts[-1]) * 10 
                except TypeError:
                    y_coast = top - (cst_lay_pts[-1][0]) * 10 
            #   calculate the y_coordinate of the line at the end of the model_obj domain
            y_coord_end = round(avg_sl_slope * (x_end - cst_offset_plot - x_coord) * 10., 1)
            line_A_lst.append((round(x_coord, 1), round(y_coord, 1)))
            line_B_lst.append((round(x_end - cst_offset_plot, 1), round(y_coord - y_coord_end, 1)))
            cst_point_lst.append((0.0, round(y_coast)))
        
        bot_point_lst = []
        for j in range(len(line_A_lst)):
            bot_pt = line_intersection((line_A_lst[j], line_B_lst[j]), line_thk_end)
            bot_point_lst.append([round(bot_pt[0], 1), round(bot_pt[1], 1)])
        
        
        #   if the shelf area is thinner than the end of the model domain it means that the bedrock line and the layer lines will never intersect offshore,
        #   instead they would intersect inland. Because of that, cross-section points will be calculated individually at the end of model domain
        shelf_active_layers = ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, shelf_edges_col_idx[-1]].tolist()) if x == 1]       
        end_active_layers = ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, ibound_arr.shape[-1] - 1].tolist()) if x == 1] 
        
        if len(end_active_layers) >= len(shelf_active_layers):
        
            #   check that the position of the intersection is in the offshore direction from the shelf break, if not then change the line_thk_end coordinates
            if bot_point_lst[-1][0] < line_A_lst[-1][0]:
                bot_point_lst = []
                
                x_end = round(x_st + round((ibound_arr.shape[-1] * (dc / 1000.)), 2), 1)
                y_top_end = top_elev[-1]
                y_bot_end = lay_elev[ibound_arr.shape[0]]
                
                line_B_lst = []
                tot_lay_thk_cnt = 0
                tot_thk = abs(y_bot_end) - abs(y_top_end)
                for h in range(len(inland_aqf_lrs) - 1):
                    #   calculate the depth of the layer end
                    pct_incr = round((tot_thk * inland_aqf_lrs[h][0]) / 100., 1)
                    tot_lay_thk_cnt += pct_incr
                    line_B_lst.append((x_end, round(y_top_end - tot_lay_thk_cnt, 1)))
                    
                line_thk_end = ((x_end, y_top_end), (x_end, y_bot_end))
                for j in range(len(line_A_lst)):
                    bot_pt = line_intersection((line_A_lst[j], line_B_lst[j]), line_thk_end)
                    bot_point_lst.append([round(bot_pt[0], 1), round(bot_pt[1], 1)])            

        shelf_edge_0_ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, shelf_edges_col_idx[-1]].tolist()) if x == 1]   
        
        #   find the limits for each of the geological layers 
        if glhymps_2_top is True:
            idx_1 = shelf_edge_0_ibound_act_lay_idxs[col_top_lay_n]  
        else:
            idx_1 = shelf_edge_0_ibound_act_lay_idxs[0]
        
        end_lay_idxs =  [] #end_lay_idxs =  [idx_1]
        act_cell_lst = [i for i, x in enumerate(ibound_arr[idx_1, :].tolist()) if x == 1]
        end_col_idxs = []#[act_cell_lst[-1]]                
        for b in range(tot_layers - 1):
            if glhymps_2_top is True:
                lay_idx = idx_1 + int(round((inland_aqf_lrs[b][0] * (ibound_arr.shape[0] - shelf_edge_0_ibound_act_lay_idxs[col_top_lay_n])) / 100 , 0))          
            else:
                lay_idx = idx_1 + int(round((inland_aqf_lrs[b][0] * (ibound_arr.shape[0] - shelf_edge_0_ibound_act_lay_idxs[0])) / 100 , 0))               
                            
            end_lay_idxs.append(lay_idx)
            act_cell_lst = [i for i, x in enumerate(ibound_arr[lay_idx - 1, :].tolist()) if x == 1]
            
            #   if the list is empty then just paste the same value as the last one
            if act_cell_lst != []:
                end_col_idxs.append(act_cell_lst[-1])
                idx_1 = lay_idx
            else:
                end_col_idxs.append(end_col_idxs[-1])
                idx_1 = lay_idx                
            
        #   append the last active layer and column as the end of the last layer
        end_lay_idxs.append(ibound_arr.shape[0])
        end_col_idxs.append(int(ibound_arr.shape[-1]) - 1)
        
        #   choose the shelf_edge index, has to be larger than the coastal index
        shelf_edges_col_idx = [i for i in shelf_edges_col_idx if i > cst_idx_geo]
        
        #   fill in the model domain between the coastline and the beginning of the shelf edges area
        for a in range(cst_idx_geo, shelf_edges_col_idx[-1]):    
            #   check if there are any ibound_act_lay_idxs indexes below the upper aquifer part
            #   fill in the part till the shelf edge with the constant thickness, after that point fill in straight line
            #   the upper part of the offshore aquifer domain
            ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, a].tolist()) if x == 1]
            lrs_lst = []  
            #   check if there are still cells from the upper glhymps 2 layer in the active cells offshore, if yes cut the list
            h = 0
            if last_glh_2_top_lay in ibound_act_lay_idxs:
                while last_glh_2_top_lay > ibound_act_lay_idxs[h]:
                    hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                    if hk_cell_val > hk_aqf_mean / 10.:
                        hk_arr[ibound_act_lay_idxs[h], 0, a] = hk_cell_val
                    else:
                        hk_arr[ibound_act_lay_idxs[h], 0, a] = hk_aqf_mean / 10.
                    h += 1
            ibound_act_lay_idxs = ibound_act_lay_idxs[h:]
            
            #   also check if the last glh 2 top layer index is actually higher than the last element of the active layer, then the full column is glh 2
            if last_glh_2_top_lay >= ibound_act_lay_idxs[-1]:
                for i in range(len(ibound_act_lay_idxs)):
                    hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                    if hk_cell_val > hk_aqf_mean / 10.:
                        hk_arr[ibound_act_lay_idxs[i], 0, a] = hk_cell_val
                    else:
                        hk_arr[ibound_act_lay_idxs[i], 0, a] = hk_aqf_mean / 10.
                    h += 1          
                ibound_act_lay_idxs = []
                continue
                    
            #   go shelf edge by edge and fill in the columns based on the line crossing
            pts_coord_start = cst_point_lst[:]
            ptc_coord_end = line_A_lst[:]
                    
            #   get the line of the column middle
            col_top_y = botm[ibound_act_lay_idxs[0] - 1] # round(top - (ibound_act_lay_idxs[0]) * 10 , 1)
            try:
                col_bot_y = botm[ibound_act_lay_idxs[-1]] # round(top - (ibound_act_lay_idxs[-1] + 1) * 10, 1) 
            except IndexError:
                col_bot_y = botm[-1] # round(top - (ibound_act_lay_idxs[-1] + 1) * 10, 1) 
            col_x =  round(x_st - cst_offset_plot + a / 10., 1)# + 0.05               
                    
            #   get the line crossing for each of the layer lines
            line_cross_lst, idx_cross_lst = [], []
            for x in range(len(pts_coord_start)):
                line_cross = line_intersection((pts_coord_start[x], ptc_coord_end[x]), ((col_x, col_top_y), (col_x, col_bot_y)))
                idx_cross_lst.append(int((top - round(line_cross[1], 0)) / 10.))    
            #   reverse the list to start from the top layer             
            add_lay_lst = [i for i in idx_cross_lst if i < ibound_act_lay_idxs[0]]   #   add layers (empty) on top if necessary  - sometimes there is no intersection
            #   remove values from the list that are not indexes of active cells
            idx_cross_lst = [i for i in idx_cross_lst if i in ibound_act_lay_idxs]
            
            if len(idx_cross_lst) == 0:
                idx_cross_lst = [ibound_act_lay_idxs[x:x+len(pts_coord_start)] for x in range(0, len(ibound_act_lay_idxs), len(pts_coord_start))]
                idx_cross_lst = idx_cross_lst[0]
            
            #   fill in the HK array based on the list of indexes from previous step!
            grp_lay_idx = [ibound_act_lay_idxs[:ibound_act_lay_idxs.index(idx_cross_lst[0])]]        
            
            for y in range(1, len(idx_cross_lst)):
                grp_lay_idx.append(ibound_act_lay_idxs[ibound_act_lay_idxs.index(idx_cross_lst[y - 1]) : ibound_act_lay_idxs.index(idx_cross_lst[y])])
            #   append the bottom layer till the end of the active model_obj domain
            grp_lay_idx.append(ibound_act_lay_idxs[ibound_act_lay_idxs.index(idx_cross_lst[-1]):])
            
            #   add empty lists on top of the system if necessary
            for z in range(len(add_lay_lst)):
                grp_lay_idx.insert(0, [])
            
            #   make sure there are no duplicates or empty lists in the grp_lay_idx
            grp_lay_idx_sorted = []
            for sublist in grp_lay_idx:
                #   remove any empty list from the sorted one
                #if sublist not in grp_lay_idx_sorted or sublist != []:
                if sublist not in grp_lay_idx_sorted and len(sublist) != 0:
                    grp_lay_idx_sorted.append(sublist)
            
            #   check if the upper glhymps 2 layer (from soilgrids) is still located within the offshore model domain, if so
            #   then assign the right property to those cells and cut the respective sublist to be filled in by the other layers
            for w in range(len(grp_lay_idx_sorted)):
                #   + 1 because the upper layer is already filled in by GLHYMPS 2.0
                if inland_aqf_lrs[w][1] == 'aqf':
                    lrs_lst.append(['aqf', grp_lay_idx_sorted[w]])
                    #   for every even layer fill in with GLHMYPS 2.0 (even is 0, 2, 4..)
                    for t in range(len(grp_lay_idx_sorted[w])):
                        hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                        if hk_cell_val > hk_aqf_mean / 10.:
                            hk_arr[grp_lay_idx_sorted[w][t], 0, a] = hk_cell_val
                        else:
                            hk_arr[grp_lay_idx_sorted[w][t], 0, a] = hk_aqf_mean / 10.                        
                        #hk_arr[grp_lay_idx_sorted[w][t], 0, a] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)
                else:
                    lrs_lst.append(['aqt', grp_lay_idx_sorted[w]])
                    for t in range(len(grp_lay_idx_sorted[w])):
                        hk_arr[grp_lay_idx_sorted[w][t], 0, a] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)           
            offshore_lrs_lst.append([a, lrs_lst])  
        
        """
        SECOND, DEAL WITH THE AREA WITH THE SHELF EDGES
        """
        
        edge_count = 1   
        shelf_edge_last_ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, shelf_edges_col_idx[0]].tolist()) if x == 1]   

        #   this changes if the glhymps 2 layer is extended over the whole shelf domain, lay top designates the upper model layer from which we start
        #   to calculate the thickness of each glhymps layer 
        if glhymps_2_top is True:
            lay_top = shelf_edge_last_ibound_act_lay_idxs[col_top_lay_n] 
        else:
            if last_glh_2_top_lay not in shelf_edge_last_ibound_act_lay_idxs:
                lay_top = shelf_edge_last_ibound_act_lay_idxs[0]
            else:
                lay_top = shelf_edge_last_ibound_act_lay_idxs[shelf_edge_last_ibound_act_lay_idxs.index(last_glh_2_top_lay)]

        #   find the limits for each of the geological layers 
        if lay_top <= last_glh_2_top_lay:
            idx_1 = last_glh_2_top_lay # shelf_edge_last_ibound_act_lay_idxs.index(last_glh_2_top_lay)
        else:
            idx_1 = shelf_edge_0_ibound_act_lay_idxs[0] 
            
        act_cell_lst = [i for i, x in enumerate(ibound_arr[idx_1, :].tolist()) if x == 1]
        end_col_idxs = []        
        for e in range(tot_layers - 1): 
            lay_idx = idx_1 + int(round((inland_aqf_lrs[e][0] * (ibound_arr.shape[0] - shelf_edge_last_ibound_act_lay_idxs[0])) / 100, 0))
            try:
                act_cell_lst = [i for i, x in enumerate(ibound_arr[lay_idx - 1, :].tolist()) if x == 1]
                end_col_idxs.append(act_cell_lst[-1])
            except IndexError:
                act_cell_lst = [i for i, x in enumerate(ibound_arr[-1, :].tolist()) if x == 1]
                #end_col_idxs.append(int(ncol) - 1)
                end_col_idxs.append(int(ibound_arr.shape[-1]))
            idx_1 = lay_idx
        #   append the last active layer and column as the end of the last layer
        end_col_idxs.append(int(ibound_arr.shape[-1]))
        
        for b in range(shelf_edges_col_idx[-1], shelf_edge_idx + 1):
            #   get the active layers in the column
            ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, b].tolist()) if x == 1]
            lrs_lst = []  
            #   check if there are still cells from the upper glhymps 2 layer in the active cells offshore, if yes cut the list
            h = 0
            if last_glh_2_top_lay in ibound_act_lay_idxs:
                while last_glh_2_top_lay > ibound_act_lay_idxs[h]:
                    hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                    if hk_cell_val > hk_aqf_mean / 10.:
                        hk_arr[ibound_act_lay_idxs[h], 0, b] = hk_cell_val
                    else:
                        hk_arr[ibound_act_lay_idxs[h], 0, b] = hk_aqf_mean / 10.                       
                    h += 1
            ibound_act_lay_idxs = ibound_act_lay_idxs[h:]            
            
            #   go shelf edge by edge and fill in the columns based on the line crossing
            pts_coord_start = cst_point_lst[:-edge_count] + line_A_lst[-edge_count:]
            ptc_coord_end = line_A_lst[:-edge_count] + line_B_lst[-edge_count:]
                    
            #   get the line of the column middle
            col_top_y = round(top - (ibound_act_lay_idxs[0] + 1) * 10 , 1)
            col_bot_y = round(top - (ibound_act_lay_idxs[-1] + 1) * 10, 1) 
            col_x = round(x_st - cst_offset_plot + b / 10., 1)# + 0.05               
                    
            #   get the line crossing for each of the layer lines
            line_cross_lst, idx_cross_lst = [], []
            for x in range(len(pts_coord_start)):
                line_cross = line_intersection((pts_coord_start[x], ptc_coord_end[x]), ((col_x, col_top_y), (col_x, col_bot_y)))
                #print(round(line_cross[0], 1), top - round(line_cross[1], 0))
                idx_cross_lst.append(int((top - round(line_cross[1], 0)) / 10.))    

            #   check that the indexes are not higher than the extend of the active cells
            for s in range(len(idx_cross_lst)):
                if idx_cross_lst[s] > ibound_act_lay_idxs[-1]:
                    idx_cross_lst[s] = ibound_act_lay_idxs[-1]
        
            #   fill in the HK array based on the list of indexes from previous step!
            try:
                grp_lay_idx = [ibound_act_lay_idxs[:ibound_act_lay_idxs.index(idx_cross_lst[0])]] 
            #   it can happen that the cross-section happens in the inactive zone above the ocean bottom, in that case fill in empty list
            except ValueError:
                grp_lay_idx = [[]] 
                
            for y in range(1, len(idx_cross_lst)):
                try:
                    grp_lay_idx.append(ibound_act_lay_idxs[ibound_act_lay_idxs.index(idx_cross_lst[y - 1]) : ibound_act_lay_idxs.index(idx_cross_lst[y])])
                #   same as above, the first index will become the first active cell in the ibound_act_lay_idx
                except ValueError:
                    try:
                        grp_lay_idx.append(ibound_act_lay_idxs[: ibound_act_lay_idxs.index(idx_cross_lst[y])])
                    except ValueError:
                        grp_lay_idx.append(ibound_act_lay_idxs[: ibound_act_lay_idxs[0]])
            #   append the bottom layer till the end of the active model_obj domain
            grp_lay_idx.append(ibound_act_lay_idxs[ibound_act_lay_idxs.index(idx_cross_lst[-1]):])
            
            #   make sure there are no duplicates or empty lists in the grp_lay_idx
            grp_lay_idx_sorted = []
            for sublist in grp_lay_idx:
                if sublist not in grp_lay_idx_sorted or sublist == []:
                    grp_lay_idx_sorted.append(sublist)
            
            #   check if the upper glhymps 2 layer (from soilgrids) is still located within the offshore model domain, if so
            #   then assign the right property to those cells and cut the respective sublist to be filled in by the other layers
            for w in range(len(grp_lay_idx_sorted)):
                #   + 1 because the upper layer is already filled in by GLHYMPS 2.0
                if inland_aqf_lrs[w][1] == 'aqf':
                    lrs_lst.append(['aqf', grp_lay_idx_sorted[w]])
                    #   for every even layer fill in with GLHMYPS 2.0 (even is 0, 2, 4..)
                    for t in range(len(grp_lay_idx_sorted[w])):
                        hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                        if hk_cell_val > hk_aqf_mean / 10.:
                            hk_arr[grp_lay_idx_sorted[w][t], 0, b] = hk_cell_val
                        else:
                            hk_arr[grp_lay_idx_sorted[w][t], 0, b] = hk_aqf_mean / 10.                                 
                else:
                    lrs_lst.append(['aqt', grp_lay_idx_sorted[w]])
                    for t in range(len(grp_lay_idx_sorted[w])):
                        hk_arr[grp_lay_idx_sorted[w][t], 0, b] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)           
            upp_lim = 0
            
            #   check if the first sublist of the lrs_lst is glhymps_1, if yes check if there are indexes lower or equal to the upp_lim
            if lrs_lst[0][0] == 'aqt':
                #   check if the upp_lim index is in the list of indexes
                try:
                    split_idx = lrs_lst[0][1].index(ibound_act_lay_idxs[upp_lim])                        
                    lrs_lst.insert(0, ['aqf', lrs_lst[0][1][:split_idx]])
                    lrs_lst[1][1] = lrs_lst[1][1][split_idx:]
                except ValueError:
                    pass
                
            edge_count = 1
            #   check the counter
            for c in range(len(shelf_edges_col_idx) - 1):
                if b > shelf_edges_col_idx[c]:
                    edge_count += 1      
        
            offshore_lrs_lst.append([b, lrs_lst])     
        
        """
        IN THE LAST STEP, FILL IN THE REST OF THE OFFSHORE MODEL DOMAIN
        """
        
        #   Instead of just filling in between the lines and thus creating a disproporiantely large upper sediment zone we choose a different approach
        #   that takes into account the % specified for each of the glhymps layers. 
        #   First we need to find the ending points of each glhymps bottom layer (excpet the current last layer) based on the specified % thickness.
        #   To do that, we need to define the starting and ending point for the loop. First we start with the last column from the shelf edges zone and
        #   choose the first column from the offshore end of these glhymps layers as calculated earlier based on the average slope of the continental slope.
        
        #   define the starting and ending indexes
        st_idx = shelf_edges_col_idx[0]
        end_idx = end_col_idxs[0]
        
        #   create a counter for how many layers should be left out from the proportion calculation
        lrs_cnt = 1                     #   the counter
        lrs = tot_layers - lrs_cnt      #   total  number of layers to take into account
        
        #   create lists of new starting and ending points
        end_col_layer_pts = []
        offshore_aqf_thk_lst = []
        
        #   find the limits for each of the geological layers 
        #   create a list of columns designating the end of each glhymps layer in the offshore domain based on the average slope of the continental slope
        end_col_idx_lst = [int((abs(x_st - cst_offset_plot) + i[0]) * 10.) for i in bot_point_lst[::-1]]
        
        for t in range(len(end_col_idx_lst) - 1):
            lrs = tot_layers - lrs_cnt 
            aqf_ratio_lst = []
            #   get the list of all active cells in the given column
            try:
                ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, end_col_idx_lst[t]].tolist()) if x == 1]    
                col_x =  round(x_st - cst_offset_plot + end_col_idx_lst[t] / 10., 1)# + 0.05   
            except IndexError:
                ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, -1].tolist()) if x == 1]    
                col_x =  round(x_st - cst_offset_plot + end_col_idx_lst[t] / 10., 1)# + 0.05                   
            
            #   next, change the % of the glhymps layer by distributing the % of the layer(s) not included over the rest of them
            active_lrs = inland_aqf_lrs[:lrs]
            pct_left = sum(i[0] for i in inland_aqf_lrs[lrs:])    
            tot_assigned, pct_ratio = 0, int(round(pct_left / len(active_lrs)))
            for j in range(len(active_lrs)):
                if j != len(active_lrs) - 1:
                    aqf_ratio_lst.append([active_lrs[j][0] + pct_ratio, active_lrs[j][1]])
                    tot_assigned += pct_ratio
                else:
                    aqf_ratio_lst.append([active_lrs[j][0] + (pct_left - tot_assigned), active_lrs[j][1]])
        
            #   find the bottom of each glhymps layer in model layers
            end_lay_idxs =  []     
            idx_1 = ibound_act_lay_idxs[0]    
            for e in range(lrs - 1): 
                lay_idx = idx_1 + int(round(aqf_ratio_lst[e][0] * len(ibound_act_lay_idxs) / 100, 0))
                end_lay_idxs.append(lay_idx)
                act_cell_lst = [i for i, x in enumerate(ibound_arr[lay_idx - 1, :].tolist()) if x == 1]
                idx_1 = lay_idx
            #end_lay_idxs.append(ibound_act_lay_idxs[-1])
                
            #   get the corresponding depth of these points to calculate intersections later on
            end_lay_idxs_m = []
            for g in range(len(end_lay_idxs)):
                end_lay_idxs_m.append([col_x, top - (end_lay_idxs[g] * 10.)])
            end_lay_idxs_m.append(bot_point_lst[len(end_lay_idxs)])
            
            end_col_layer_pts.append(end_lay_idxs_m)
            lrs_cnt += 1
            offshore_aqf_thk_lst.append(aqf_ratio_lst)
        
        #   insert the starting points from the past shelf edges part into the point list that we will work with further
        end_col_layer_pts.insert(0, line_A_lst)
        end_col_layer_pts.append([line_B_lst[0]])
        
        pts_coord_start = end_col_layer_pts[0][:]
        ptc_coord_end = end_col_layer_pts[1][:]    
            
        for z in range(shelf_edge_idx + 1, ibound_arr.shape[-1]):
            ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, z].tolist()) if x == 1]       
            lrs_lst = []  
            if ibound_act_lay_idxs != []:
                edge_count = 1
                #   check the counter
                for c in range(len(end_col_idx_lst) - 1):
                    if z > end_col_idx_lst[c]:
                        edge_count += 1
                #   check if there are still cells from the upper glhymps 2 layer in the active cells offshore, if yes cut the list
                h = 0
                if last_glh_2_top_lay in ibound_act_lay_idxs:
                    while last_glh_2_top_lay > ibound_act_lay_idxs[h]:
                        hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                        if hk_cell_val > hk_aqf_mean / 10.:
                            hk_arr[ibound_act_lay_idxs[h], 0, z] = hk_cell_val
                        else:
                            hk_arr[ibound_act_lay_idxs[h], 0, z] = hk_aqf_mean / 10.            
                        h += 1
                ibound_act_lay_idxs = ibound_act_lay_idxs[h:]                  
                
                #   go shelf edge by edge and fill in the columns based on the line crossing
                if edge_count <= 1:
                    pts_coord_start = end_col_layer_pts[0][:]
                    ptc_coord_end = end_col_layer_pts[1][:]    
                else:
                    pts_coord_start = end_col_layer_pts[edge_count - 1][:-1]
                    ptc_coord_end = end_col_layer_pts[edge_count][:]            
                
                #   get the line of the column middle
                col_top_y = round(top - (ibound_act_lay_idxs[0] + 1) * 10 , 1)
                col_bot_y = round(top - (ibound_act_lay_idxs[-1] + 1) * 10, 1) 
                col_x =  round(x_st - cst_offset_plot + z / 10., 1)# + 0.05               
                       
                try:
                    #   get the line crossing for each of the layer lines
                    line_cross_lst, idx_cross_lst = [], []
                    for x in range(len(pts_coord_start)):
                        line_cross = line_intersection((pts_coord_start[x], ptc_coord_end[x]), ((col_x, col_top_y), (col_x, col_bot_y)))
                        idx_cross_lst.append(int((top - round(line_cross[1], 0)) / 10.))                        
                    if idx_cross_lst[-1] > ibound_act_lay_idxs[-1] and len(idx_cross_lst) > 1:
                        idx_cross_lst[-1] = ibound_act_lay_idxs[-1]
                    
                    #   check that the indexes are actually active cells, if not then replace with the first active cell 
                    for g in range(len(idx_cross_lst)):
                        if idx_cross_lst[g] not in ibound_act_lay_idxs and idx_cross_lst[g] < ibound_act_lay_idxs[0]:
                            idx_cross_lst[g] = ibound_act_lay_idxs[0]

                    #   check that the indexes are not higher than the extend of the active cells
                    for s in range(len(idx_cross_lst)):
                        if idx_cross_lst[s] > ibound_act_lay_idxs[-1]:
                            idx_cross_lst[s] = ibound_act_lay_idxs[-1]

                    #   fill in the HK array based on the list of indexes from previous step!
                    try:
                        grp_lay_idx = [ibound_act_lay_idxs[:ibound_act_lay_idxs.index(idx_cross_lst[0])]] 
                    #   it can happen that the cross-section happens in the inactive zone above the ocean bottom, in that case fill in empty list
                    except ValueError:
                        grp_lay_idx = [[]] 
                        
                    for y in range(1, len(idx_cross_lst)):
                        try:
                            grp_lay_idx.append(ibound_act_lay_idxs[ibound_act_lay_idxs.index(idx_cross_lst[y - 1]) : ibound_act_lay_idxs.index(idx_cross_lst[y])])
                        #   same as above, the first index will become the first active cell in the ibound_act_lay_idx
                        except ValueError:
                            try:
                                grp_lay_idx.append(ibound_act_lay_idxs[: ibound_act_lay_idxs.index(idx_cross_lst[y])])
                            except ValueError:
                                grp_lay_idx.append(ibound_act_lay_idxs[: ibound_act_lay_idxs[0]])
                    #   append the bottom layer till the end of the active model_obj domain
                    grp_lay_idx.append(ibound_act_lay_idxs[ibound_act_lay_idxs.index(idx_cross_lst[-1]):])
                    
                    #   make sure there are no duplicates or empty lists in the grp_lay_idx
                    #grp_lay_idx = [x for x in grp_lay_idx if x != []]
                    grp_lay_idx_sorted = []
                    for sublist in grp_lay_idx:
                        if sublist not in grp_lay_idx_sorted or sublist == []:
                            grp_lay_idx_sorted.append(sublist)
                    
                    #   check if the upper glhymps 2 layer (from soilgrids) is still located within the offshore model domain, if so
                    #   then assign the right property to those cells and cut the respective sublist to be filled in by the other layers
                    
                    for w in range(len(grp_lay_idx_sorted)):
                        #   + 1 because the upper layer is already filled in by GLHYMPS 2.0
                        if inland_aqf_lrs[w][1] == 'aqf':
                            lrs_lst.append(['aqf', grp_lay_idx_sorted[w]])
                            #   for every even layer fill in with GLHMYPS 2.0 (even is 0, 2, 4..)
                            for t in range(len(grp_lay_idx_sorted[w])):
                                #hk_arr[grp_lay_idx_sorted[w][t], 0, z] = round(abs(np.random.normal(hk_top_mean, hk_top_std)), 4)
                                
                                hk_cell_val = round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8)
                                if hk_cell_val > hk_aqf_mean / 10.:
                                    hk_arr[grp_lay_idx_sorted[w][t], 0, z] = hk_cell_val
                                else:
                                    hk_arr[grp_lay_idx_sorted[w][t], 0, z] = hk_aqf_mean / 10.                                            
                        else:
                            lrs_lst.append(['aqt', grp_lay_idx_sorted[w]])
                            for t in range(len(grp_lay_idx_sorted[w])):
                                hk_arr[grp_lay_idx_sorted[w][t], 0, z] = round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8)     

                    #   check if the first sublist of the lrs_lst is glhymps_1, if yes check if there are indexes lower or equal to the upp_lim
                    if lrs_lst[0][0] == 'aqt':
                        #   check if the upp_lim index is in the list of indexes
                        try:
                            split_idx = lrs_lst[0][1].index(ibound_act_lay_idxs[upp_lim])                        
                        except ValueError:
                            pass
                    try:
                        if split_idx:
                            lrs_lst.insert(0, ['aqf', lrs_lst[0][1][:split_idx]])
                            lrs_lst[1][1] = lrs_lst[1][1][split_idx:]
                    #   error - NameError: name 'split_idx' is not defined
                    except NameError:
                        pass
                 
                except Exception:
                    grp_lay_idx = grp_lay_idx
                
                upp_lim = 0
                edge_count = 1
                #   check the counter
                for c in range(len(end_col_idx_lst) - 1):
                    if z > end_col_idx_lst[c]:
                        edge_count += 1
                
            offshore_lrs_lst.append([z, lrs_lst])     

    ##  Place the clay cells in the model domain
    
    #   define the layers that will get the clay property based on the p_fact value
    def choose_rand_clay_lrs(p_val, n_lrs, n_clay_lrs):
        #   first create a cummulative sum of weights 
        intervals = [int(round(i, 2) * 100) for i in np.linspace(p_fact, 1 - p_fact, n_lrs).tolist()]
        intervals_cumsum = []
        for i in range(1, len(intervals) + 1):
            intervals_cumsum.append(sum(intervals[: i]))
        #   from the create list of sum of weights select the desired number of clay layers
        def get_rand_layer(int_lst):
            rand_number = random.randint(0, intervals_cumsum[-1])
            rand_lay = 0
            #   check which layer that number corresponds to 
            for g in range(len(intervals_cumsum)):
                if rand_number >= intervals_cumsum[g]:
                    rand_lay = rand_lay + 1
                else:
                    break
            return rand_lay
                
        clay_lrs_idx = []
        j = 0
        while j < n_clay_lrs:
            r_lay = get_rand_layer(intervals_cumsum)
            if j == 0:
                clay_lrs_idx.append(r_lay)
                j += 1
            else:
                if r_lay in clay_lrs_idx:
                    pass
                else:
                    clay_lrs_idx.append(r_lay)         
                    j += 1
        return clay_lrs_idx
    
    #   create an array with the dimensions of the ibound_arr, set all values to 0, the clay cells will be marked as 1 in this 
    #   array and then it will be used to assign all the clay values at once
    sed_arr = ibound_arr * 0    
    #   get the column index where the offshore sed_type adjustment starts
    off_sed_start_idx = int((abs(x_st - cst_offset_plot) + off_lay_start) * (1000. / dc))
    #   combine the two lists
    tot_clay_lst = inland_lrs_lst + offshore_lrs_lst
    #   find the starting index in the offshore_lrs_lst based on the distance from coast calculated above
    off_lays_idxs = [item[0] for item in tot_clay_lst]
    #print off_lays_idxs, off_sed_start_idx
    try:
        off_lay_idx_start = off_lays_idxs.index(off_sed_start_idx)
    #   sometimes it throws an error that the off_sed_start_idx is not in the off_lays_idxs list, probably because of islands.
    except ValueError:
        off_lay_idx_start = off_sed_start_idx
    end_col_glhymps_1_start = end_col_idxs[1:][::2]
    end_col_glhymps_1_end = end_col_idxs[2:][::2][:-1]
    end_col_glhymps = []
    for d in range(len(end_col_glhymps_1_end)):
        end_col_glhymps.append([end_col_glhymps_1_start[d], end_col_glhymps_1_end[d]])
    end_col_glhymps.append([ibound_arr.shape[-1], ibound_arr.shape[-1]])
    #   counter for the different end layers 
    end_lay_cnt = 0
    
    lst_glh_1_vals, lst_glh_2_vals = [], []
    for w in range(10):
        lst_glh_1_vals.append(round(abs(np.random.normal(hk_aqt_mean, hk_aqt_std)), 8))
        lst_glh_2_vals.append(round(abs(np.random.normal(hk_aqf_mean, hk_aqf_std)), 8))

    if np.mean(lst_glh_2_vals) > np.mean(lst_glh_1_vals):
        clay_lay = 'aqt'
    else:
        clay_lay = 'aqf'

    if sed_flux == 'low':
        
        #   loop through the columns where the sed_type adjustment will be implemented        
        for i in range(off_lay_idx_start, len(tot_clay_lst)):
            #   check for the 'glhymps_1 columns', select those from the offshore_lrs_lst
            col_idx = tot_clay_lst[i][0]
            try:
                #   select only the parts where there is 'glhymps_1' sediment type
                lay_idxs = [item for item in tot_clay_lst[i][1] if item[0] == clay_lay]
                #   get the total amount of cells that should have the clay/silt properties, based on the % filled in the off_lay_thk_ratio                    
                lay_lst_1 = [item[1] for item in tot_clay_lst[i][1] if item[0] == clay_lay]
                #Maybe in future substract the clay capping cells from the ones that are then changed in the column..
                for y in range(len(lay_lst_1)):
                    if lay_lst_1[y] != []:
                        sed_cells = int(round(len(lay_lst_1[y]) * (off_lay_thk_ratio / 100.), 0))
                        #   define the layers that will get the clay property based on the p_fact value
                        clay_idxs = choose_rand_clay_lrs(p_fact, len(lay_lst_1[y]), sed_cells)     
                        for x in range(len(clay_idxs)):
                            sed_arr[lay_lst_1[y][clay_idxs[x]], col_idx] = 1  
            except IndexError:
                pass   
            
    if sed_flux == 'medium' or sed_flux == 'high':                     
        #   loop through the columns where the sed_type adjustment will be implemented        
        for i in range(off_lay_idx_start, len(tot_clay_lst)):
            #   check for the 'glhymps_1 columns', select those from the offshore_lrs_lst
            col_idx = tot_clay_lst[i][0]            
            try:
                #   select only the parts where there is 'glhymps_1' sediment type
                lay_idxs = [item for item in tot_clay_lst[i][1] if item[0] == clay_lay]
                #   get the total amount of cells that should have the clay/silt properties, based on the % filled in the off_lay_thk_ratio                    
                lay_lst_1 = [item[1] for item in tot_clay_lst[i][1] if item[0] == clay_lay]
                #Maybe in future substract the clay capping cells from the ones that are then changed in the column..
                for y in range(len(lay_lst_1)):
                    if lay_lst_1[y] != []:
                        sed_cells = int(round(len(lay_lst_1[y]) * (off_lay_thk_ratio / 100.), 0))
                        #   define the layers that will get the clay property based on the p_fact value
                        clay_idxs = choose_rand_clay_lrs(p_fact, len(lay_lst_1[y]), sed_cells)     
                        for x in range(len(clay_idxs)):
                            sed_arr[lay_lst_1[y][clay_idxs[x]], col_idx] = 1   
            except IndexError:
                pass     
    
    #   if the shelf clay cap is turned on then add the clay cells in the upper part of the model domain based on the thickness indicated
    #   first, calculate the amount of layers that will get the clay HK value based on the thickness specified in clay_cap_thk
    cap_cells_shelf = int(round(clay_cap_shelf_thk / 10., 0))
    cap_cells_slope = int(round(clay_cap_slope_thk / 10., 0))
    
    if clay_cap_shelf:
        #   do this for cells between the coastline and the shelf edge
        for x in range(cst_idx_geo, shelf_edge_idx):
            #   select the upper part of the model domain, number of cells determined above
            ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, x].tolist()) if x == 1][:cap_cells_shelf]
            for layer in ibound_act_lay_idxs:
                if top_elev[x] < 0.0:
                    sed_arr[layer, x] = 1  
                
    #   same for the clay cap in the continental slope area
    if clay_cap_slope:
        #   do this for cells between the shelf edge and the end of the model domain
        for y in range(shelf_edge_idx, ibound_arr.shape[-1]):
            #   select the upper part of the model domain, number of cells determined above
            ibound_act_lay_idxs = [i for i, x in enumerate(ibound_arr[:, y].tolist()) if x == 1][:cap_cells_slope]
            for layer in ibound_act_lay_idxs:
                sed_arr[layer, y] = 1      
    
    #   select randomly an amount of cells based on the lay_pres_y1 preservation index, and assign them back to 0. 
    #   this simulates the reworking of the internal layers with time and creates openings in these layers..
    sed_cells_idxs = np.where(sed_arr == 1)
    #   create an empty list and loop through the array created above to assign the location of each clay cell to the list
    sed_cells_idxs_lst = []
    for f in range(sed_cells_idxs[0].shape[0]):
        sed_cells_idxs_lst.append([sed_cells_idxs[0][f], sed_cells_idxs[1][f]])                       
    #   calculate the number of cells to be selected based on the lay_pres_y1 preservation index
    rework_cells_cnt = int(round(len(sed_cells_idxs_lst) * erosion_fact / 100., 0))
    rework_lst = random.sample(sed_cells_idxs_lst, rework_cells_cnt)            
    #   now go through the list of the reworked cells and assign those back to 0 in the clay_arr
    for g in range(len(rework_lst)):
        sed_arr[rework_lst[g][0], rework_lst[g][1]] = 0            

    #   now loop through the clay_arr, and for each occurrence of 1 change the value to a random value in the clay range
    for i in range(sed_arr.shape[0]):
        for j in range(sed_arr.shape[-1]):
            if sed_arr[i, j] == 1:
                if not const_geo_hk_vals: 
                    hk_arr[i, 0, j] = round(abs(np.random.normal(hk_clay_mean, hk_clay_std)), 10)
                else:
                    hk_arr[i, 0, j] = clay_val

    #   check of NaN values - if there are any, replace by average value of the hk array    
    for i in range(ibound_arr.shape[0]): 
        for j in range(hk_arr.shape[-1]):
            try:
                if ibound_arr[i, j] == 1:
                    if np.isnan(hk_arr[i, 0, j]): 
                        hk_arr[i, 0, j] = round(np.nanmean(hk_arr), 10)
            #   IndexError: index 144 is out of bounds for axis 0 with size 144
            except IndexError:
                pass
    
    return hk_arr[:, 0, :]
    




#   function that stores the geological heterogeneity values into numpy dictionary - to be loaded again into the GUI fields once we want to load a model in
def save_geo_dict(out_dir, rand_seed_in, n_aqf_aqt_lrs, p_fact, sand_pct, mud_pct, erosion_fact, clay_cap_shelf_thk, clay_cap_slope_thk, off_lay_start,\
                  sed_flux, hk_aqf_mean, hk_aqf_std, hk_aqt_mean, hk_aqt_std, hk_clay_mean, hk_clay_std):
    dict_out = {}
    dict_out['rand_seed_in'] = rand_seed_in
    dict_out['n_aqf_aqt_lrs'] = n_aqf_aqt_lrs
    dict_out['p_fact'] = p_fact
    dict_out['sand_pct'] = sand_pct
    dict_out['mud_pct'] = mud_pct
    dict_out['erosion_fact'] = erosion_fact
    dict_out['clay_cap_shelf_thk'] = clay_cap_shelf_thk
    dict_out['clay_cap_slope_thk'] = clay_cap_slope_thk
    dict_out['off_lay_start'] = off_lay_start
    dict_out['sed_flux'] = sed_flux
    dict_out['hk_aqf_mean'] = hk_aqf_mean
    dict_out['hk_aqf_std'] = hk_aqf_std
    dict_out['hk_aqt_mean'] = hk_aqt_mean
    dict_out['hk_aqt_std'] = hk_aqt_std
    dict_out['hk_clay_mean'] = hk_clay_mean
    dict_out['hk_clay_std'] = hk_clay_std
    np.save(out_dir, dict_out)





















