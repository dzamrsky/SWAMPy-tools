# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:30:53 2020

@author: daniel
"""

#import numpy as np
import os
#import sys
#import shutil
import pandas as pd
from .. import xarray as xr
#import tarfile
import numpy as np
#import math
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

#source_dir = r'/projects/0/qt16165/_dzamrsky/_A4_SRM_models/_SLR_models'
#master_csv_dir = r'/projects/0/qt16165/_dzamrsky/_A4_SRM_models/_SLR_models/_coscat_reg_main_tb_MODELRUNS.csv'
source_dir = r'g:\Water_Nexus\_A4_models\_SLR_models_OUT_files\_avg_nc_files'
out_dir = r'g:\Water_Nexus\_A4_models\_SLR_models_OUT_files\_avg_nc_files_MERGING'
master_csv_dir = r'g:\Water_Nexus\_A4_models\_SLR_models_OUT_files\RCP_85_results.csv'

#   iterate over the csv file row by row
csv_in = pd.read_csv(master_csv_dir)

for index, row in csv_in.iterrows():
    cs_id = row['coscat']
    srm_id = row['srm']
    cs_srm_id = row['cs_srm_id']
    inl_ext = row['inl_ext_km']
    #print(cs_srm_id)
    
    #   adjust the coscat_id if necessary by adding 0s in front of the number
    if cs_id < 10:
        cs_id_str = '000' + str(int(cs_id))
    elif cs_id >= 10 and cs_id < 100:
        cs_id_str = '00' + str(int(cs_id))
    elif cs_id >= 100 and cs_id < 1000:
        cs_id_str = '0' + str(int(cs_id))
    else:
        cs_id_str = str(int(cs_id))
    
    #   add 0s in front of the name to have an ordered list of COSCAT ids in the folder 
    if srm_id < 10:
        srm_id_str = '00' + str(int(srm_id))
    elif srm_id >= 10 and srm_id < 100:
        srm_id_str = '0' + str(int(srm_id))
    else:
        srm_id_str = str(int(srm_id))  
    
    #   select all files from the folder with the right SRM 
    cs_srm_str_code = cs_id_str + '_SRM_' + srm_id_str
    cs_srm_nc_lst = []
    for fname in os.listdir(source_dir):
        if cs_srm_str_code in fname:
            print(cs_srm_str_code, fname)
            cs_srm_nc_lst.append(os.path.join(source_dir, fname))

    #   now go through the list of netcdf files and merge that into a single netcdf file for each region 
    new_name = os.path.join(out_dir, cs_srm_str_code + '.nc')
    to_merge_lst = []
    #   first loop through all the files and add the time coordinate
    for nc_file in cs_srm_nc_lst:
        with xr.open_mfdataset(nc_file, autoclose=True) as da:
            
            rcp_val = nc_file.split('_RCP_')[-1].split('_')[0]
            topo_val = nc_file.split('_MERGED')[0].split('_')[-1]
                  
            #   delete unnecessary variables add the new dimensions to the netcdf
            data = da.load()            

            sel_times_lst = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,\
                             11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000,\
                             20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000,\
                             28500, 29000, 29500, 29600, 29800, 30000, 30050, 30100, 30150,\
                             30200, 30250, 30300, 30350, 30400, 30450, 30500]
            sel_times_lst_new = [x - 30000 for x in sel_times_lst]
            
            #   transform the y coordinates so they give the mid-cell y coordinate 
            y_coords = data['y'].values.tolist()
            new_y_coords = np.arange(y_coords[0], y_coords[0] - len(y_coords) * 10, -10.)
            
            new_conc = []
            #for i in range(data['solute concentration'].values.shape[0]):
            for ts in sel_times_lst:
                
                #   if the ts = -30000 it doesnt put the negative values there so do that manually
                if ts == 0:
                    conc = data.sel(time = ts)['solute concentration'].values * 100
                    conc_0 = data.sel(time = 30000)['solute concentration'].values * 100
                    for i in range(conc.shape[0]):
                        for j in range(conc.shape[1]):
                            if np.isnan(conc_0[i, j]):
                                conc[i, j] = np.nan
                else:
                    conc = data.sel(time = ts)['solute concentration'].values * 100
                conc_no_nan = np.nan_to_num(conc, nan = -1000)
                new_conc.append(conc_no_nan.astype('int16'))
            new_conc_arr = np.asarray(new_conc)
            
            new_head = []
            for ts in sel_times_lst:
                head = data.sel(time = ts)['heads'].values 
                new_head.append(head.astype('int16'))
            new_head_arr = np.asarray(new_head)

            #   create a new netcdf file
            xa_sum = xr.Dataset(data_vars = {'salinity' : (('time', 'y', 'x'), new_conc_arr),
                                             'heads' : (('time', 'y', 'x'), new_head_arr)},
                                coords = {'x' : data['x'].values.tolist(),
                                          'y' : new_y_coords,#data['y'].values.tolist(),
                                          'time' : sel_times_lst_new})
            xa_sum = xa_sum.assign_coords(rcp = rcp_val)
            xa_sum = xa_sum.expand_dims('rcp')
            xa_sum = xa_sum.assign_coords(dem = topo_val)
            xa_sum = xa_sum.expand_dims('dem')
            xa_name = cs_srm_str_code + '_rcp_' + rcp_val + '_dem_' + topo_val + '.nc'
            xa_sum.to_netcdf(os.path.join(out_dir, xa_name))       
            to_merge_lst.append(os.path.join(out_dir, xa_name))
            del xa_sum, data

    if len(to_merge_lst) > 1:
        #   now merge the newly created netcdf files along the rcp and dem coordinates = create a single netcdf 
        ds_rcp = xr.open_mfdataset(to_merge_lst, concat_dim = 'rcp', autoclose = True)
        ds_rcp = ds_rcp.assign_coords(cs_srm = cs_id_str + '_SRM_' + str(srm_id))
        ds_rcp = ds_rcp.expand_dims('cs_srm')      
        merge_name = os.path.join(out_dir, cs_id_str + '_SRM_' + str(srm_id) + '.nc')
        ds_rcp.to_netcdf(merge_name, engine = 'netcdf4')
        del ds_rcp
        
        #   remove the previously created netcdf files
        for nc in to_merge_lst:
            if os.path.isfile(nc):
                os.remove(nc)
    
    
    
    
