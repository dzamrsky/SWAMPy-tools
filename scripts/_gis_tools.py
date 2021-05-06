""" ************************************************************************ """
"""                     IMPORT ALL NECESSARY LIBRARIES                       """
""" ************************************************************************ """

import psycopg2
import math
from ._utm_conversion import to_latlon, from_latlon
from collections import Counter

""" ************************************************************************ """
"""                     DEFINE ALL FUNCTIONS NECESSARY                       """
""" ************************************************************************ """

#   function to connect to the database
def connect_to_dtbase(db_name, db_user, db_host, db_pass):
    #   create connection string
    conn_string = str("dbname=%s user=%s host=%s password=%s") % (db_name, db_user, db_host, db_pass)
    try:
        conn = psycopg2.connect(conn_string)
        print("Successfully connected to database : " + str(db_name))
        #   set the cursor
        cur = conn.cursor()
        return conn, cur
    except:
        print("I am unable to connect to database : " + str(db_name))


#  Function to create a database table with specified columns
#  Intended for creation of values extracted from input data at cross-section
#  points location..
def sql_create_table(db_conn, db_cursor, tb_name, id_name, tb_cols):
    ##  first check that the cursor exists
    if db_cursor:
        ##  if yes then create the SQL and run it
        sql_command = "CREATE TABLE IF NOT EXISTS %s (\
                            %s serial PRIMARY KEY, %s);" % (tb_name, id_name, tb_cols)
        #   print sql_command
        db_cursor.execute(sql_command)
        db_conn.commit()
    else:
        print('Database cursor doesnt exist!')


#  Function to create a list of columns depending on distance between coastal
#  point and the points on the cross-section
#  col_count (equal to n_points), dist - distance between points on cross-section
def create_column_string(col_count, dist, col_type):
    col_string = ""
    for i in range(col_count + 1):
        dist_to_cs = i - (col_count/2)
        ##  for negative distance values
        if dist_to_cs < 0:
            col_string += 'dist_minus_' + str(int((abs(dist_to_cs) * dist))) + 'm ' + col_type + ','
        ##  for coastal point - distance is 0..
        elif dist_to_cs == 0:
            col_string += 'dist_' + str(int(dist_to_cs * dist)) + 'm ' + col_type + ','
        ##  for positive distance values
        else:
            col_string += 'dist_plus_' + str(int(dist_to_cs * dist)) + 'm ' + col_type + ','
    ##  remove the last comma in the string to get correct SQL command
    col_string = col_string[:-1]
    return col_string


#  Function for insert SQL into specific columns of the table
def sql_insert(db_conn, db_cursor, db_table, db_columns, ins_vals):
    ##  first check that the cursor exists
    if db_cursor:
        ##  if yes then perform the insert SQL
        sql_command = "INSERT INTO %s (%s) VALUES (%s)" % (db_table, db_columns, ins_vals)
        db_cursor.execute(sql_command)
        db_conn.commit()
    else:
        print('Database cursor doesnt exist!')


#  Function for inserting values in existing rows - updating the row
def sql_update(db_conn, db_cursor, db_table, db_columns, ins_vals, where_condition):
    ##  first check that the cursor exists
    #print db_cursor, db_table, db_columns, ins_vals, where_condition
    if db_cursor:
        ##  if yes then perform the insert SQL
        sql_command = "UPDATE %s SET %s = %s WHERE %s" % (db_table, db_columns, ins_vals, where_condition)
        #print sql_command
        db_cursor.execute(sql_command)
        db_conn.commit()
    else:
        print('Database cursor doesnt exist!')


#   function to calculate the coordinates from triangle information
def coord_from_triangle(x0, y0, x1, y1, b_len):
    c_len = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    ##  abs value in case (x0-x1) < 0 - fixes the wrong angle the line has in case d_len < 0
    d_len = abs(x0 - x1)
    # print x0, y0, x1, y1, b_len, c_len, d_len
    delta = math.degrees(math.acos(d_len / c_len))
    omega = 90 - delta
    ##  calculate the coordinates for both directions from the point_5km coastal point
    ##  conditions below make sure that the cross-section is perpendicular
    if x0 > x1 and y0 > y1:
        C_coord_x_sea = y0 + math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x0 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y0 - math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x0 + math.cos(math.radians(omega)) * b_len
    elif x0 < x1 and y0 < y1:
        C_coord_x_sea = y0 + math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x0 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y0 - math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x0 + math.cos(math.radians(omega)) * b_len
    else:
        """
        C_coord_x_sea = y0 + math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x0 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y0 - math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x0 + math.cos(math.radians(omega)) * b_len
        """
        C_coord_x_sea = y0 - math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x0 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y0 + math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x0 + math.cos(math.radians(omega)) * b_len
        
    ##  return all the calculated coordinates
    return C_coord_x_land, C_coord_y_land, C_coord_x_sea, C_coord_y_sea


#   Function to get the closest raster pixel value to point coordinates
#   this fc is used in case the (coastal) point lies on a novalue pixel..
#   avg_type -> either average or maximal frequency value (for land_cover etc)
#   in case the raster provides data in classed values
def find_closest_neighbour(point_x_col, point_y_row, r_band, r_noval ,raster_x_size, raster_y_size, avg_type):
    ##  check if the col or row are not out of bounds of the raster extent
    if point_x_col >= raster_x_size:
        point_x_col_min_1 = point_x_col - 1
        point_x_col = point_x_col - raster_x_size
        point_x_col_plus_1 = point_x_col + 1
    elif point_x_col <= 0:
        point_x_col_plus_1 = point_x_col + 1
        point_x_col = point_x_col + raster_x_size - 1
        point_x_col_min_1 = point_x_col - 1
    elif point_y_row >= raster_y_size:
        point_y_row = point_y_row - raster_y_size
    elif point_y_row < 0:
        point_y_row = point_y_row + raster_y_size
    else:
        point_x_col_plus_1 = point_x_col + 1
        if point_x_col_plus_1 >= raster_x_size:
            point_x_col_plus_1 = point_x_col_plus_1 - raster_x_size
        point_x_col_min_1 = point_x_col - 1
    ##  get an average value from the surrounding pixels, omit the noval pixels
    non_noval_list = []
    non_noval_list.append(r_band.ReadAsArray(point_x_col_min_1, point_y_row - 1, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col_min_1, point_y_row, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col_min_1, point_y_row + 1, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col, point_y_row - 1, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col, point_y_row + 1, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col_plus_1, point_y_row - 1, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col_plus_1, point_y_row, 1, 1)[0][0])
    non_noval_list.append(r_band.ReadAsArray(point_x_col_plus_1, point_y_row + 1, 1, 1)[0][0])
    ##  remove the noval values from the list
    non_noval_list = [x for x in non_noval_list if x != r_noval]
    #print non_noval_list
    ##  calculate the average value or max_freq values
    if avg_type == 'avg':
        try:
            pixel_value = sum(non_noval_list) / float(len(non_noval_list))
        ##  in case the non_noval_list is empty
        except ZeroDivisionError:
            pixel_value = -9999
    elif avg_type == 'max_freq':
        try:
            pixel_value = Counter(non_noval_list).most_common(1)[0][0]
        ##  in case the non_noval_list is empty
        except IndexError:
            pixel_value = -9999
    ##  return the calculated pixel_value
    return pixel_value


#  Function to check the position to the equator (south or north)
#  if north ( > 10 000 000m) then substract it and move the zone_let from M to N
def equator_position(coords_1, coords_2, coords_3, coords_4, zone_num, zone_let):
    #print coords_1, coords_2, coords_3, coords_4, zone_num, zone_let
    if coords_1 > 10000000 and coords_3 > 10000000:
        zone_let_c_land, zone_let_c_sea = 'N', 'N'
        c_coords_wgs_land = to_latlon(coords_1, coords_2 - 10000000, zone_num, zone_let_c_land)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4 - 10000000, zone_num, zone_let_c_sea)

    elif coords_1 > 10000000 and coords_3 < 10000000:
        zone_let_c_land, zone_let_c_sea = 'N', zone_let
        c_coords_wgs_land = to_latlon(coords_1, coords_2 - 10000000, zone_num, zone_let_c_land)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4, zone_num, zone_let_c_sea)

    elif coords_1 < 10000000 and coords_3 > 10000000:
        zone_let_c_land, zone_let_c_sea = zone_let , 'N'
        c_coords_wgs_land = to_latlon(coords_1, coords_2, zone_num, zone_let_c_land)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4 - 10000000, zone_num, zone_let_c_sea)

    else:
        c_coords_wgs_land = to_latlon(coords_1, coords_2, zone_num, zone_let)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4, zone_num, zone_let)
    return c_coords_wgs_land, c_coords_wgs_sea


#  Function to check the position to the equator (south or north)
#  if north ( > 10 000 000m) then substract it and move the zone_let from M to N
"""    
coords_1 = off_utm[1]
coords_2 = off_utm[0]
coords_3 = off_utm[3]
coords_4 = off_utm[2]
zone_num
zone_let
"""
def equator_position_angles(coords_1, coords_2, coords_3, coords_4, zone_num, zone_let):

    #print coords_1, coords_2, coords_3, coords_4

    if coords_2 < 0 and coords_4 < 0:
        #print ('-2')
        zone_let_c_land, zone_let_c_sea = 'M', 'M'
        c_coords_wgs_land = to_latlon(coords_1, coords_2 + 10000000, zone_num, zone_let_c_land, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4 + 10000000, zone_num, zone_let_c_sea, strict=False)

    elif coords_2 < 0 and coords_4 > 0:
        #print ('-1')
        zone_let_c_land, zone_let_c_sea = 'M', zone_let
        c_coords_wgs_land = to_latlon(coords_1, coords_2 + 10000000, zone_num, zone_let_c_land, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4, zone_num, zone_let_c_sea, strict=False)

    elif coords_2 > 0 and coords_4 < 0:
        #print ('0')
        zone_let_c_land, zone_let_c_sea = zone_let, 'M'
        c_coords_wgs_land = to_latlon(coords_1, coords_2, zone_num, zone_let, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4 + 10000000, zone_num, zone_let_c_sea, strict=False)

    elif coords_2 > 10000000 and coords_4 > 10000000:
        #print ('1')
        zone_let_c_land, zone_let_c_sea = 'N', 'N'
        c_coords_wgs_land = to_latlon(coords_1, coords_2 - 10000000, zone_num, zone_let_c_land, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4 - 10000000, zone_num, zone_let_c_sea, strict=False)

    elif coords_2 > 10000000 and coords_4 < 10000000:
        #print ('2')
        zone_let_c_land, zone_let_c_sea = 'N', zone_let
        c_coords_wgs_land = to_latlon(coords_1, coords_2 - 10000000, zone_num, zone_let_c_land, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4, zone_num, zone_let_c_sea, strict=False)

    elif coords_2 < 10000000 and coords_4 > 10000000:
        #print ('3')
        zone_let_c_land, zone_let_c_sea = zone_let , 'N'
        c_coords_wgs_land = to_latlon(coords_1, coords_2, zone_num, zone_let_c_land, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4 - 10000000, zone_num, zone_let_c_sea, strict=False)

    else:
        #print ('4')
        #print coords_1, coords_2, coords_3, coords_4
        c_coords_wgs_land = to_latlon(coords_1, coords_2, zone_num, zone_let, strict=False)
        c_coords_wgs_sea = to_latlon(coords_3, coords_4, zone_num, zone_let, strict=False)

    return c_coords_wgs_land, c_coords_wgs_sea



#   function to calculate the coordinates from triangle information
#   works for the non-perpendicular cross-sections, need to specify alfa
#   X0 is the coastal point and X1 is a cross-section point
def coord_from_triangle_alfa(x0, y0, x1, y1, b_len, alfa):
    ##  calculate the distance between the coastal point and the cross-section point
    c_len = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    #print c_len
    ##  calculate the difference between y coordinates of two points
    a_1 = abs(y0 - y1)
    #print a_1
    ##  calculate angle alfa_1
    alfa_1 = math.degrees(math.asin(a_1 / c_len))
    #print alfa_1
    ##  calculate alfa_2 in radians
    alfa_2 = math.radians(90 - alfa_1 - alfa)
    #print alfa_2, 90 - alfa_1 - alfa
    ##  with alfa_2 calculate the distances b1 and b2
    b_1 = b_len * math.cos(alfa_2)
    b_2 = b_len * math.sin(alfa_2)
    #print b_1, b_2
    ##  with these two distances it is now possible to calculate the coordinates
    if x0 > x1 and y0 > y1:
        coord_x_sea = x0 + b_1
        coord_y_sea = y0 - b_2
        coord_x_land = x0 - b_1
        coord_y_land = y0 + b_2
        if alfa == 45:
            return coord_x_sea, coord_y_sea, coord_x_land, coord_y_land
        else:
            return coord_x_land, coord_y_land, coord_x_sea, coord_y_sea
    elif x0 < x1 and y0 < y1:
        coord_x_sea = x0 - b_1
        coord_y_sea = y0 + b_2
        coord_x_land = x0 + b_1
        coord_y_land = y0 - b_2
        if alfa == 135:
            return coord_x_sea, coord_y_sea, coord_x_land, coord_y_land
        else:
            return coord_x_land, coord_y_land, coord_x_sea, coord_y_sea
    #elif x0 < x1 and y0 > y1:
    #    coord_x_sea = x0 - b_1
    #    coord_y_sea = y0 + b_2
    #    coord_x_land = x0 + b_1
    #    coord_y_land = y0 - b_2
    else:
        coord_x_sea = x0 - b_1
        coord_y_sea = y0 - b_2
        coord_x_land = x0 + b_1
        coord_y_land = y0 + b_2
        if alfa == 135:
            return coord_x_sea, coord_y_sea, coord_x_land, coord_y_land
        else:
            return coord_x_land, coord_y_land, coord_x_sea, coord_y_sea

#   function to calculate the average value of a profile point
#   creates a perpendicular cross-section of the cross-section at given
#   point and gives a list of coordinates that can then be used for
#   extracting values from a raster
def avg_cs_point_coord(x0, y0, x1, y1, b_len):
    c_len = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    ##  in case c_len = 0 change the value to prevent division by zero (this is for the
    ##  coastal point averages!)
    if c_len == 0:
        c_len = 0.0000001
    ##  abs value in case (x0-x1) < 0 - fixes the wrong angle the line has in case d_len < 0
    d_len = abs(x0 - x1)
    # print x0, y0, x1, y1, b_len, c_len, d_len
    delta = math.degrees(math.acos(d_len / c_len))
    omega = 90 - delta
    ##  calculate the coordinates for both directions from the point_5km coastal point
    ##  conditions below make sure that the cross-section is perpendicular
    if x0 > x1 and y0 > y1:
        C_coord_x_sea = y1 + math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x1 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y1 - math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x1 + math.cos(math.radians(omega)) * b_len
    elif x0 < x1 and y0 < y1:
        C_coord_x_sea = y1 + math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x1 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y1 - math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x1 + math.cos(math.radians(omega)) * b_len
    else:
        C_coord_x_sea = y1 - math.sin(math.radians(omega)) * b_len
        C_coord_y_sea = x1 - math.cos(math.radians(omega)) * b_len
        C_coord_x_land = y1 + math.sin(math.radians(omega)) * b_len
        C_coord_y_land = x1 + math.cos(math.radians(omega)) * b_len
    ##  return all the calculated coordinates
    return C_coord_x_land, C_coord_y_land, C_coord_x_sea, C_coord_y_sea


#   function to extract values from raster. In case the point coordinates are in
#   the edge of the raster (x > 180 or x < 180) read raster values from the other
#   'side' of the raster. E.g. if the raster column is 2 pixels higher than the
#   total of raster columns than read the column 2 of the raster = earth is round!
"""
in_rb = rb_coastal
gt_coastal = in_gt
rx_coastal = raster_x_size
ry_coastal = raster_y_size
point_lat = point_coord_x
point_lon = point_coord_y
"""
def read_raster_val(in_rb, in_gt, raster_x_size, raster_y_size, point_coord_x, point_coord_y):
    ##  calculate the col and row for the point coordinates
    px = int((point_coord_x - in_gt[0]) / in_gt[1]) #x pixel
    py = int((point_coord_y - in_gt[3]) / in_gt[5]) #y pixel
    #print point_coord_x, point_coord_y
    #print px, py
    #print('-----------------------------------------------------')
    ##  check if the col and row are within the raster extent range, if not
    ##  change them so they are - earth is round!
    ##  this is now adapted for the MERIT DEM files - added the line with == raster_x_size, and raster_y_size.
    ##  if it is the case then just take the -1 cell which is the last column/row in the raster file.
    if px > raster_x_size:
        px = px - raster_x_size
    elif px == raster_x_size:
        px = raster_x_size - 1
    elif px < 0:
        px = px + raster_x_size
    if py > raster_y_size:
        py = py - raster_y_size
    elif py == raster_y_size:
        py = raster_y_size - 1
    elif py < 0:
        py = py + raster_y_size
    ##  get the raster value in the px and py
    #print px, py
    pixel_val = in_rb.ReadAsArray(px, py, 1, 1)[0][0]
    ##  return the pixel value
    return pixel_val








