# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:33:18 2020

@author: daniel

from PyQt5.QtCore import Qt
from qgis.gui import QgsMapTool
from qgis.utils import iface
from PyQt5.QtWidgets import  QHBoxLayout, QLabel, QDialog

from qgis.gui import QgsMapToolEmitPoint
class PrintClickedPoint(QgsMapToolEmitPoint):
    def __init__(self, canvas):
        self.canvas = canvas
        QgsMapToolEmitPoint.__init__(self, self.canvas)

    def canvasPressEvent( self, e ):
        point = self.toMapCoordinates(self.canvas.mouseLastXY())
        return point[0], point[1]

class CoordTool(QgsMapTool):
    def __init__(self, canvas):
        QgsMapTool.__init__(self, canvas)
        self.canvas = canvas
        self.setCursor(Qt.CrossCursor)
        self.point = None

    def canvasClicked(self, event):
        self.point = self.toMapCoordinates(event.pos())
        return str(self.point.x()), str(self.point.y())

    def canvasReleaseEvent(self, event):
        self.point = self.toMapCoordinates(event.pos())
        coords = "X: "+ str(self.point.x()) +"/ Y: "+str( self.point.y())
        #print(coords)
        #return str(point.x()), str(point.y())
        
        dlg =QDialog()
        label =QLabel(coords)
        layout=QHBoxLayout()
        layout.addWidget(label)
        dlg.setLayout(layout)
        dlg.exec_()
"""

from qgis.PyQt.QtCore import pyqtSignal, Qt, QVariant
from PyQt5.QtGui import QColor 
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsVertexMarker
from qgis.core import QgsWkbTypes, QgsPointXY, QgsApplication, QgsPoint, QgsGeometry, QgsVectorLayer, QgsFeature, QgsFields, QgsProject,\
QgsExpression, QgsFeatureRequest, QgsVectorFileWriter, QgsCoordinateReferenceSystem, QgsField, QgsRasterLayer
import processing, os
from PyQt5.QtWidgets import QMessageBox
from ._utm_conversion import to_latlon, from_latlon
from ._gis_tools import coord_from_triangle, equator_position_angles
import pandas as pd
import math

#   define functions for calculating angles - from https://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get angle in radians and then convert to degrees
    angle = math.acos(round(dot_prod / magB / magA, 4))
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360
    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else: 
        return ang_deg


class CoordTool(QgsMapToolEmitPoint):
    mouseMoved = pyqtSignal(QgsPointXY)
    mouseClicked = pyqtSignal(QgsPointXY)
    doubleClicked = pyqtSignal()
    
    def __init__(self, canvas):
        super(CoordTool, self).__init__(canvas)

        self.mapCanvas = canvas
        self.m = QgsVertexMarker(canvas)
        self.m.setColor(QColor(0,191,255))
        self.m.setIconSize(5)
        self.m.setIconType(QgsVertexMarker.ICON_CIRCLE) # or ICON_CROSS, ICON_X
        self.m.setPenWidth(3)

        #self.rubberBand = QgsRubberBand(self.mapCanvas, QgsWkbTypes.PointGeometry)
        #self.rubberBand.setColor(Qt.blue)
        #self.rubberBand.setWidth(2)
        self.setCursor(QgsApplication.getThemeCursor(QgsApplication.Cursor.CrossHair))

    def canvasMoveEvent(self, e):
        originalPoint = QgsPointXY(self.mapCanvas.getCoordinateTransform().toMapCoordinates(e.x(), e.y()))
        self.mouseMoved.emit(originalPoint)

    def canvasPressEvent(self, e):
        if e.button() == Qt.LeftButton:
            originalPoint = QgsPointXY(self.mapCanvas.getCoordinateTransform().toMapCoordinates(e.x(), e.y()))
            self.mouseClicked.emit(originalPoint)

            point1 = QgsPointXY(self.mapCanvas.getCoordinateTransform().toMapCoordinates(e.x() - 1, e.y() - 1))
            point2 = QgsPointXY(self.mapCanvas.getCoordinateTransform().toMapCoordinates(e.x() + 1, e.y() - 1))
            point3 = QgsPointXY(self.mapCanvas.getCoordinateTransform().toMapCoordinates(e.x() + 1, e.y() + 1))
            point4 = QgsPointXY(self.mapCanvas.getCoordinateTransform().toMapCoordinates(e.x() - 1, e.y() + 1))

            self.m.setCenter(point1)
            self.m.setCenter(point2)
            self.m.setCenter(point3)
            self.m.setCenter(point4)
            
            #self.rubberBand.reset(QgsWkbTypes.PolygonGeometry )
            #self.rubberBand.addPoint(point1, False)
            #self.rubberBand.addPoint(point2, False)
            #self.rubberBand.addPoint(point3, False)
            #self.rubberBand.addPoint(point4, True)
            self.m.show()

        elif e.button() == Qt.RightButton:
            #self.mapCanvas.unsetMapTool(self)
            self.deactivate()
            
    def drawCrossSection(self, x_inl, y_inl, x_off, y_off, cst_layer, cst_lines_layer, model_name, out_dir, dc, dl, perpendicular = True):
        r = QgsRubberBand(self.mapCanvas, QgsWkbTypes.LineGeometry)  # False = not a polygon
        r.setColor(QColor(0,191,255))
        r.setWidth(2)
        points = [QgsPoint(float(x_inl), float(y_inl)), QgsPoint(float(x_off), float(y_off))]
        self.cross_section = r.setToGeometry(QgsGeometry.fromPolyline(points), None)
        #   create output directory
        self.tmp_dir = os.path.join(out_dir, model_name, 'temp_files')
        os.makedirs(self.tmp_dir, exist_ok = True)

        # create a new memory layer
        v_layer = QgsVectorLayer("LineString", "line", "memory")
        pr = v_layer.dataProvider()
        # create a new feature
        seg = QgsFeature()
        # add the geometry to the feature, 
        seg.setGeometry(QgsGeometry.fromPolyline(points))
        # add the geometry to the layer
        pr.addFeatures([seg])
        # show the line  
        QgsProject.instance().addMapLayers([v_layer], False)

        #   if perpendicular is True then find the intersection with coastline
        if perpendicular:
            #   use the qgis function to find the intersection points between coastline and cross-section    
            if not os.path.exists(os.path.join(self.tmp_dir, 'cst_intersect_points.shp')):
                parameters = {'INPUT' : v_layer, 'INTERSECT' : cst_layer, 'INPUT_FIELDS' : None, 'INTERSECT_FIELDS' : None,
                              'OVERLAY_FIELDS_PREFIX' : None, 'OUTPUT' : os.path.join(self.tmp_dir, 'cst_intersect_points.shp')}
                processing.run("qgis:lineintersections", parameters)

            if not os.path.exists(os.path.join(self.tmp_dir, 'cs_point.shp')):
                cst_points = QgsVectorLayer(os.path.join(self.tmp_dir, 'cst_intersect_points.shp'), "cst_points", "ogr")
                pr = cst_points.dataProvider()
                pr.addAttributes([QgsField("dist",  QVariant.Double)])
                cst_points.startEditing()
                off_point = QgsPointXY(float(x_off), float(y_off)) #Point(-5.182, 5.088)
                for feature in cst_points.getFeatures():
                    geom = feature.geometry()
                    distance = off_point.distance(geom.asPoint())
                    feature['dist'] = distance
                    cst_points.updateFeature(feature)
                cst_points.commitChanges()
                #   select the minumum value point and export
                idx = cst_points.fields().indexFromName('dist')
                cst_points.selectByExpression('dist=%s' % cst_points.minimumValue(idx))
                processing.run('qgis:saveselectedfeatures', {'INPUT' : cst_points, 'OUTPUT' : os.path.join(self.tmp_dir, 'cs_point.shp')})
            
            for feature in QgsVectorLayer(os.path.join(self.tmp_dir, 'cs_point.shp'), "cs_point", "ogr").getFeatures():
                self.cs_point = feature.geometry()                
                
            #   select the closest 100 vertices to the coastal point - just to be sure we cover enough
            #       first create a buffer around the coastal point
            if not os.path.exists(os.path.join(self.tmp_dir, 'cst_buff.shp')):
                processing.run("native:buffer", {'INPUT': os.path.join(self.tmp_dir, 'cs_point.shp'), 'DISTANCE' : 0.25, 'SEGMENTS' : 5, 'END_CAP_STYLE': 0,
                               'JOIN_STYLE' : 0, 'MITER_LIMIT' : 2,'DISSOLVE' : False, 'OUTPUT': os.path.join(self.tmp_dir, 'cst_buff.shp')})                
            #       clip the vertices layer by the created buffer
            #if not os.path.exists(os.path.join(tmp_dir, 'cst_buff_vertices.shp')):
            #    processing.run("native:clip", {'INPUT' : cst_nodes_layer, 'OVERLAY' : os.path.join(tmp_dir, 'cst_buff.shp'),
            #                    'OUTPUT' : os.path.join(tmp_dir, 'cst_buff_vertices.shp')})       

            #       clip the vertices layer by the created buffer
            if not os.path.exists(os.path.join(self.tmp_dir, 'cst_clipped.shp')):
                processing.run("native:clip", {'INPUT' : cst_lines_layer, 'OVERLAY' : os.path.join(self.tmp_dir, 'cst_buff.shp'),
                                'OUTPUT' : os.path.join(self.tmp_dir, 'cst_clipped.shp')})  
            #       loop through the single lines and extract the vertice id - sort the list next
            for feature in QgsVectorLayer(os.path.join(self.tmp_dir, 'cst_clipped.shp'), "cs_clipped", "ogr").getFeatures(): 
                if self.cs_point.distance(feature.geometry()) < 10e-10:
                    node_point = feature.geometry().constGet()[0][0]
                    #print(feature.geometry(), feature.geometry().constGet()[0][0]) 

            """
            x_inl, y_inl = -16.54603599, 15.32007746
            x_off, y_off = -17.53061299, 15.67004559
            cs_pt = -16.77957,15.40268
            node_pt = -16.78985,15.38475
            
            cs_point_utm = from_latlon(cs_pt[1], cs_pt[0])
            inl_point_utm = from_latlon(y_inl, x_inl)
            off_point_utm = from_latlon(y_off, x_off)
            inl_dist_utm = round(((cs_point_utm[0] - inl_point_utm[0])**2 + (cs_point_utm[1] - inl_point_utm[1])**2)**(.5) / dc, 0) * dc
            off_dist_utm = round(((cs_point_utm[0] - off_point_utm[0])**2 + (cs_point_utm[1] - off_point_utm[1])**2)**(.5) / dc, 0) * dc  

            #       get the utm coordinates of the new inland and offshore point
            node_point_utm = from_latlon(node_pt[1], node_pt[0])
            zone_num, zone_let = node_point_utm[2], node_point_utm[3]
            #   check that zone_num is between 1 and 60, case for areas that are split between the zones 1 and 60 (e.g. certain islands in Pacific..)
            if zone_num > 60:
                zone_num = zone_num - 60
            
            inl_utm = coord_from_triangle(cs_point_utm[0], cs_point_utm[1], node_point_utm[0], node_point_utm[1], inl_dist_utm)            
            inl_wgs = equator_position_angles(inl_utm[1], inl_utm[0], inl_utm[3], inl_utm[2], zone_num, zone_let)
            #self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[0][1], inl_wgs[0][0]
            #       same for the offshore point
            off_utm = coord_from_triangle(cs_point_utm[0], cs_point_utm[1], node_point_utm[0], node_point_utm[1], off_dist_utm)            
            off_wgs = equator_position_angles(off_utm[1], off_utm[0], off_utm[3], off_utm[2], zone_num, zone_let)



            #   depending on the initial X coordinates assign the right inland and offshore extent
            #   define the lines inland
            #print(cs_point.asPoint().x())
            line_inl_orig = ((float(x_inl), float(y_inl)), (float(cs_pt[0]), float(cs_pt[1])))
            line_inl_0 = ((inl_wgs[0][1], inl_wgs[0][0]), (float(cs_pt[0]), float(cs_pt[1])))
            line_inl_1 = ((inl_wgs[1][1], inl_wgs[1][0]), (float(cs_pt[0]), float(cs_pt[1])))

            #   calculate the angle  between the line and the perpendicular lines
            ang_0 = ang(line_inl_0, line_inl_orig)
            ang_1 = ang(line_inl_1, line_inl_orig)
            #print('Angle 0 in degrees = ', ang_0)
            #print('Angle 1 in degrees = ', ang_1)
            #   the angle between the original line and the perpendicular one always has to be lower than 90 degrees
            #   and the sum of both angles is going to be 180 degrees always = choose the smaller angle in all cases
            if ang_0 < ang_1:
                idx_inl = 0
            else:
                idx_inl = 1

            #   define the lines offshore
            line_off_orig = ((float(x_off), float(y_off)), (float(cs_pt[0]), float(cs_pt[1])))
            line_off_0 = ((off_wgs[0][1], off_wgs[0][0]), (float(cs_pt[0]), float(cs_pt[1])))
            line_off_1 = ((off_wgs[1][1], off_wgs[1][0]), (float(cs_pt[0]), float(cs_pt[1])))
            #   calculate the angle  between the line and the perpendicular lines
            ang_0 = ang(line_off_0, line_off_orig)
            ang_1 = ang(line_off_1, line_off_orig)
            #print('Angle 0 in degrees = ', ang_0)
            #print('Angle 1 in degrees = ', ang_1)
            #   the angle between the original line and the perpendicular one always has to be lower than 90 degrees
            #   and the sum of both angles is going to be 180 degrees always = choose the smaller angle in all cases
            if ang_0 < ang_1:
                idx_off = 0
            else:
                idx_off = 1

            #   select the right coordinates
            inl_wgs_x, inl_wgs_y =  inl_wgs[idx_inl][1], inl_wgs[idx_inl][0]
            off_wgs_x, off_wgs_y =  off_wgs[idx_off][1], off_wgs[idx_off][0]   


            #   depending on the initial X coordinates assign the right inland and offshore extent
            x_inl_lst = [i[1] for i in inl_wgs]
            y_inl_lst = [i[0] for i in inl_wgs]
            x_off_lst = [i[1] for i in off_wgs]
            y_off_lst = [i[0] for i in off_wgs]

            x_inl_max = max(x_inl_lst)                    
            idx_inl_max = x_inl_lst.index(x_inl_max)
            x_inl_min = min(x_inl_lst)   
            idx_inl_min = x_inl_lst.index(x_inl_min)

            if float(x_inl) > float(x_off):
                if float(y_inl) > float(y_off):
                    inl_wgs_x, inl_wgs_y =  inl_wgs[idx_inl_max][1], inl_wgs[idx_inl_max][0]
                    off_wgs_x, off_wgs_y =  off_wgs[idx_inl_max][1], off_wgs[idx_inl_max][0]                              
                else:
                    inl_wgs_x, inl_wgs_y =  inl_wgs[idx_inl_max][1], inl_wgs[idx_inl_max][0]
                    off_wgs_x, off_wgs_y =  off_wgs[idx_inl_max + 1][1], off_wgs[idx_inl_max + 1][0]                        
            else:
                if float(y_inl) > float(y_off):
                    inl_wgs_x, inl_wgs_y =  inl_wgs[idx_inl_min][1], inl_wgs[idx_inl_min][0]
                    off_wgs_x, off_wgs_y =  off_wgs[idx_inl_min][1], off_wgs[idx_inl_min][0]                              
                else:
                    inl_wgs_x, inl_wgs_y =  inl_wgs[idx_inl_min][1], inl_wgs[idx_inl_min][0]
                    off_wgs_x, off_wgs_y =  off_wgs[idx_inl_min + 1][1], off_wgs[idx_inl_min + 1][0]       
            """   

            #line_inl_orig = ((float(x_inl), float(y_inl)), (cs_pt[0], cs_pt[1]))
            #line_inl_0 = ((inl_wgs[0][1], inl_wgs[0][0]), (cs_pt[0], cs_pt[1]))
            #line_inl_1 = ((inl_wgs[1][1], inl_wgs[1][0]), (cs_pt[0], cs_pt[1]))
            #line_off_orig = ((float(x_inl), float(y_inl)), (cs_pt[0], cs_pt[1]))
            #line_off_0 = ((off_wgs[0][1], off_wgs[0][0]), (cs_pt[0], cs_pt[1]))
            #line_off_1 = ((off_wgs[1][1], off_wgs[1][0]), (cs_pt[0], cs_pt[1]))

            #   calculate the distance inland and offshore in the original cross-section, and then recalculate the right position in the perpendicular cross-section
            self.cs_point_utm = from_latlon(self.cs_point.asPoint().y(), self.cs_point.asPoint().x())
            inl_point_utm = from_latlon(float(y_inl), float(x_inl))
            off_point_utm = from_latlon(float(y_off), float(x_off))
            #       the original distances in meters (UTM coordinate system), round the point distances to match the DC values
            inl_dist_utm = round(((self.cs_point_utm[0] - inl_point_utm[0])**2 + (self.cs_point_utm[1] - inl_point_utm[1])**2)**(.5) / dc, 0) * dc
            off_dist_utm = round(((self.cs_point_utm[0] - off_point_utm[0])**2 + (self.cs_point_utm[1] - off_point_utm[1])**2)**(.5) / dc, 0) * dc    
            
            #       get the utm coordinates of the new inland and offshore point
            self.node_point_utm = from_latlon(node_point.y(), node_point.x())
            self.zone_num, self.zone_let = self.node_point_utm[2], self.node_point_utm[3]
            #   check that zone_num is between 1 and 60, case for areas that are split between the zones 1 and 60 (e.g. certain islands in Pacific..)
            if self.zone_num > 60:
                self.zone_num = self.zone_num - 60
            #       calculate the b_length and determine the location of the new inland point
            inl_utm = coord_from_triangle(self.cs_point_utm[0], self.cs_point_utm[1], self.node_point_utm[0], self.node_point_utm[1], inl_dist_utm)            
            inl_wgs = equator_position_angles(inl_utm[1], inl_utm[0], inl_utm[3], inl_utm[2], self.zone_num, self.zone_let)
            #self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[0][1], inl_wgs[0][0]
            #       same for the offshore point
            off_utm = coord_from_triangle(self.cs_point_utm[0], self.cs_point_utm[1], self.node_point_utm[0], self.node_point_utm[1], off_dist_utm)            
            off_wgs = equator_position_angles(off_utm[1], off_utm[0], off_utm[3], off_utm[2], self.zone_num, self.zone_let)
            #self.off_wgs_x, self.off_wgs_y =  off_wgs[1][1], off_wgs[1][0]

            #   depending on the initial X coordinates assign the right inland and offshore extent
            #   define the lines inland
            #print(self.cs_point.asPoint().x())
            line_inl_orig = ((float(x_inl), float(y_inl)), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_inl_0 = ((inl_wgs[0][1], inl_wgs[0][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_inl_1 = ((inl_wgs[1][1], inl_wgs[1][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))

            #   calculate the angle  between the line and the perpendicular lines
            ang_0 = ang(line_inl_0, line_inl_orig)
            ang_1 = ang(line_inl_1, line_inl_orig)
            #print('Angle 0 in degrees = ', ang_0)
            #print('Angle 1 in degrees = ', ang_1)
            #   the angle between the original line and the perpendicular one always has to be lower than 90 degrees
            #   and the sum of both angles is going to be 180 degrees always = choose the smaller angle in all cases
            if ang_0 < ang_1:
                idx_inl = 0
            else:
                idx_inl = 1

            #   define the lines offshore
            line_off_orig = ((float(x_off), float(y_off)), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_off_0 = ((off_wgs[0][1], off_wgs[0][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_off_1 = ((off_wgs[1][1], off_wgs[1][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            #   calculate the angle  between the line and the perpendicular lines
            ang_0 = ang(line_off_0, line_off_orig)
            ang_1 = ang(line_off_1, line_off_orig)
            #print('Angle 0 in degrees = ', ang_0)
            #print('Angle 1 in degrees = ', ang_1)
            #   the angle between the original line and the perpendicular one always has to be lower than 90 degrees
            #   and the sum of both angles is going to be 180 degrees always = choose the smaller angle in all cases
            if ang_0 < ang_1:
                idx_off = 0
            else:
                idx_off = 1

            #   select the right coordinates
            self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[idx_inl][1], inl_wgs[idx_inl][0]
            self.off_wgs_x, self.off_wgs_y =  off_wgs[idx_off][1], off_wgs[idx_off][0]    
            
            """
            if float(x_inl) > float(x_off):
                if float(y_inl) > float(y_off):
                    self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[idx_inl_max][1], inl_wgs[idx_inl_max][0]
                    self.off_wgs_x, self.off_wgs_y =  off_wgs[idx_inl_max][1], off_wgs[idx_inl_max][0]                              
                else:
                    self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[idx_inl_max][1], inl_wgs[idx_inl_max][0]
                    self.off_wgs_x, self.off_wgs_y =  off_wgs[idx_off_min][1], off_wgs[idx_off_min][0]                        
            else:
                if float(y_inl) > float(y_off):
                    self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[idx_inl_min][1], inl_wgs[idx_inl_min][0]
                    self.off_wgs_x, self.off_wgs_y =  off_wgs[idx_off_max][1], off_wgs[idx_off_max][0]                              
                else:
                    self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[idx_inl_min][1], inl_wgs[idx_inl_min][0]
                    self.off_wgs_x, self.off_wgs_y =  off_wgs[idx_off_max][1], off_wgs[idx_off_max][0]       

            
            #   depending on the initial X coordinates assign the right inland and offshore extent
            if float(x_inl) > float(x_off) and float(y_inl) > float(y_off):
                self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[0][1], inl_wgs[0][0]
                self.off_wgs_x, self.off_wgs_y =  off_wgs[1][1], off_wgs[1][0]
            else:
                self.inl_wgs_x, self.inl_wgs_y =  inl_wgs[1][1], inl_wgs[1][0]
                self.off_wgs_x, self.off_wgs_y =  off_wgs[0][1], off_wgs[0][0]
            """
            
            self.inl_ext, self.off_ext = inl_dist_utm / 1000., off_dist_utm / 1000.
            r.hide()
            #self.eleteLines()
            
            v_layer = QgsVectorLayer("LineString", "line_2", "memory")
            pr = v_layer.dataProvider()
            seg = QgsFeature()
            points = [QgsPoint(self.inl_wgs_x, self.inl_wgs_y), QgsPoint(self.off_wgs_x, self.off_wgs_y)]
            #points = [QgsPoint(float(inl_wgs[1][1]), float(inl_wgs[1][0])), QgsPoint(float(off_wgs[0][1]), float(off_wgs[0][0]))]
            seg.setGeometry(QgsGeometry.fromPolyline(points))
            pr.addFeatures([seg])
            QgsProject.instance().addMapLayers([v_layer], False)
            r = QgsRubberBand(self.mapCanvas, QgsWkbTypes.LineGeometry)  # False = not a polygon
            r.setColor(QColor(0,0,255))
            r.setWidth(3)
            self.cross_section_perpendicular = r.setToGeometry(QgsGeometry.fromPolyline(points), None)
            
            #   create the cross-section points and export them as csv
            self.createCrossSectionPoints(x_inl, y_inl, x_off, y_off, dc, self.tmp_dir)
        
        #   clean the memory
        del v_layer, r, points, node_point
        #   remove the points from canvas
        self.deletePoints()

    #   creating cross-section point dataset - to be also saved as a csv file - in this way we can also let the user change values manually in e.g. excel
    def createCrossSectionPoints(self, x_inl, y_inl, x_off, y_off, dc, out_dir):
        #   the cross-section points will be placed between grid points - e.g. halfway between inland boundary and the next cell boundary 
        #   start from the inland boundary and move towards the offshore boundary, calculate the amount of points to be created inland
        inl_pts_num = int(self.inl_ext * 1000. / dc)
        #       now loop through the points and find coordinates for each point - save to list
        inl_coords_lst = []
        cs_pt_id = 1
        for a in range(inl_pts_num):
            dist_from_coast = self.inl_ext * 1000. - dc / 2. - a * dc    # the dc / 2. is there to make sure we get coordinates in mid-cell 
            #       calculate the b_length and determine the location of the new inland point
            inl_utm = coord_from_triangle(self.cs_point_utm[0], self.cs_point_utm[1], self.node_point_utm[0], self.node_point_utm[1], dist_from_coast)            
            inl_wgs = equator_position_angles(inl_utm[1], inl_utm[0], inl_utm[3], inl_utm[2], self.zone_num, self.zone_let)    
            
            line_inl_orig = ((float(x_inl), float(y_inl)), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_inl_0 = ((inl_wgs[0][1], inl_wgs[0][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_inl_1 = ((inl_wgs[1][1], inl_wgs[1][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            #   calculate the angle  between the line and the perpendicular lines
            ang_0 = ang(line_inl_0, line_inl_orig)
            ang_1 = ang(line_inl_1, line_inl_orig)
            #print('Angle 0 in degrees = ', ang_0)
            #print('Angle 1 in degrees = ', ang_1)
            #   the angle between the original line and the perpendicular one always has to be lower than 90 degrees
            #   and the sum of both angles is going to be 180 degrees always = choose the smaller angle in all cases
            if ang_0 < ang_1:
                idx_inl = 0
            else:
                idx_inl = 1
            inl_wgs_x, inl_wgs_y =  inl_wgs[idx_inl][1], inl_wgs[idx_inl][0]
            inl_coords_lst.append([cs_pt_id, inl_wgs_x, inl_wgs_y, -1 * dist_from_coast])
            
            """
            #   depending on the initial X coordinates assign the right inland and offshore extent
            if float(x_inl) > float(x_off) and float(y_inl) > float(y_off):
                inl_coords_lst.append([cs_pt_id, inl_wgs[0][1], inl_wgs[0][0], -1 * dist_from_coast])
            else:
                inl_coords_lst.append([cs_pt_id, inl_wgs[1][1], inl_wgs[1][0], -1 * dist_from_coast])
            """
            cs_pt_id += 1
        #   now do the same for the offshore points
        off_pts_num = int(self.off_ext * 1000. / dc)
        #       now loop through the points and find coordinates for each point - save to list, this time we loop decreasingly to keep the id points correct
        off_coords_lst = []
        for b in range(off_pts_num, -1, -1):
            dist_from_coast = self.off_ext * 1000. - dc / 2. - b * dc    # the dc / 2. is there to make sure we get coordinates in mid-cell 
            #       calculate the b_length and determine the location of the new inland point
            off_utm = coord_from_triangle(self.cs_point_utm[0], self.cs_point_utm[1], self.node_point_utm[0], self.node_point_utm[1], dist_from_coast)            
            off_wgs = equator_position_angles(off_utm[1], off_utm[0], off_utm[3], off_utm[2], self.zone_num, self.zone_let)            

            #   define the lines offshore
            line_off_orig = ((float(x_off), float(y_off)), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_off_0 = ((off_wgs[0][1], off_wgs[0][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            line_off_1 = ((off_wgs[1][1], off_wgs[1][0]), (float(self.cs_point.asPoint().x()), float(self.cs_point.asPoint().y())))
            #   calculate the angle  between the line and the perpendicular lines
            ang_0 = ang(line_off_0, line_off_orig)
            ang_1 = ang(line_off_1, line_off_orig)
            #   the angle between the original line and the perpendicular one always has to be lower than 90 degrees
            #   and the sum of both angles is going to be 180 degrees always = choose the smaller angle in all cases
            if ang_0 < ang_1:
                idx_off = 0
            else:
                idx_off = 1
            #   select the right coordinates
            off_wgs_x, off_wgs_y =  off_wgs[idx_off][1], off_wgs[idx_off][0]   
            off_coords_lst.append([cs_pt_id, off_wgs_x, off_wgs_y, dist_from_coast])

            """
            #   depending on the initial X coordinates assign the right inland and offshore extent
            if float(x_inl) > float(x_off) and float(y_inl) > float(y_off):
                off_coords_lst.append([cs_pt_id, off_wgs[1][1], off_wgs[1][0], dist_from_coast])
            else:
                off_coords_lst.append([cs_pt_id, off_wgs[0][1], off_wgs[0][0], dist_from_coast])
            """
            cs_pt_id += 1            
        #   write the output csv file
        csv_headers = ['id_cs_pt', 'x_wgs84', 'y_wgs84', 'dist_coast_m']
        df_out = pd.DataFrame(inl_coords_lst + off_coords_lst)
        self.csv_dir = os.path.join(out_dir, 'cs_points.csv')
        df_out.to_csv(self.csv_dir, index = False, header = csv_headers)        
        
    #   loading data from raster input datasets into the csv file
    def loadRasterInfoToCrossSection(self, multipl_val, col_header, raster_dir):
        #   load in the csv file and the raster input file, then loop through the coastal points and extract value from the raster
        df_in = pd.read_csv(self.csv_dir)
        rlayer = QgsRasterLayer(raster_dir, col_header)        
        val_lst = []
        for index, row in df_in.iterrows():
            x, y = row['x_wgs84'], row['y_wgs84']
            val, res = rlayer.dataProvider().sample(QgsPointXY(x, y), 1)      
            val_lst.append(round(val * multipl_val, 8))
        #   append the column to the csv file and save the updated version
        df_in[col_header] = val_lst 
        #csv_headers = list(df_in.columns)
        df_in.to_csv(self.csv_dir, index = False, header = True)           

    #   get the coastal thickness from the closest ATE point, if there is no point closer than 25km give the default value
    def getClosestCstThk(self, cst_thk_dir):
        #   check if the input dir is not empty
        if os.path.exists(cst_thk_dir):
            #   clip the ate points in a buffer around the coastal point
            if not os.path.exists(os.path.join(self.tmp_dir, 'cst_thickness_clipped.shp')):
                processing.run("native:clip", {'INPUT' : cst_thk_dir, 'OVERLAY' : os.path.join(self.tmp_dir, 'cst_buff.shp'),
                                'OUTPUT' : os.path.join(self.tmp_dir, 'cst_thickness_clipped.shp')})  
            #       loop through the single lines and extract the vertice id - sort the list next
            for feature in QgsVectorLayer(os.path.join(self.tmp_dir, 'cs_point.shp'), "cs_point", "ogr").getFeatures():
                cs_point = feature.geometry()
            min_dist = 1.
            #   check if the layer is not empty, if it is just assign a default value of 100m
            if QgsVectorLayer(os.path.join(self.tmp_dir, 'cst_thickness_clipped.shp'), "cst_thickness_clipped", "ogr").featureCount() > 0:
                for feature in QgsVectorLayer(os.path.join(self.tmp_dir, 'cst_thickness_clipped.shp'), "cst_thickness_clipped", "ogr").getFeatures(): 
                    if cs_point.distance(feature.geometry()) < min_dist:
                        min_dist = cs_point.distance(feature.geometry())
                        self.ate_val = round(feature["overall_av"], 0)
            else:
                self.ate_val = 100.
        else:
            self.ate_val = 100.    


                    
    """
    def changeExtentCrossSection(self, inl_len, off_len, perpendicular = True):
        #   recalculate the inland and offshore extent of the cross-section
        #   if perpendicular is True then find the intersection with coastline
        if perpendicular:    
            #   first clear the canvas from the previous lines
            #self.deleteLines()
            #   calculate new coordinates
            inl_utm = coord_from_triangle(self.cs_point_utm[0], self.cs_point_utm[1], self.node_point_utm[0], self.node_point_utm[1], inl_len)            
            inl_wgs = equator_position_angles(inl_utm[1], inl_utm[0], inl_utm[3], inl_utm[2], self.zone_num, self.zone_let)
            self.inl_wgs_x, self.inl_wgs_y = inl_wgs[0][0], inl_wgs[0][1]      
            off_utm = coord_from_triangle(self.cs_point_utm[0], self.cs_point_utm[1], self.node_point_utm[0], self.node_point_utm[1], off_len)            
            off_wgs = equator_position_angles(off_utm[1], off_utm[0], off_utm[3], off_utm[2], self.zone_num, self.zone_let)
            self.off_wgs_x, self.off_wgs_y =  off_wgs[1][0], off_wgs[1][1]
            self.inl_ext, self.off_ext = inl_len / 1000., off_len / 1000.
            #   redraw the cross-section
            v_layer = QgsVectorLayer("LineString", "line_2", "memory")
            pr = v_layer.dataProvider()
            seg = QgsFeature()
            points = [QgsPoint(float(self.inl_wgs_x), float(self.inl_wgs_y)), QgsPoint(float(self.off_wgs_x), float(self.off_wgs_y))]
            seg.setGeometry(QgsGeometry.fromPolyline(points))
            pr.addFeatures([seg])
            QgsProject.instance().addMapLayers([v_layer], False)
            r = QgsRubberBand(self.mapCanvas, QgsWkbTypes.LineGeometry)  # False = not a polygon
            r.setColor(QColor(0,0,255))
            r.setWidth(3)
            self.cross_section_perpendicular = r.setToGeometry(QgsGeometry.fromPolyline(points), None)
    """

    def deleteLines(self):
        line_items = [ i for i in self.mapCanvas.scene().items() if issubclass(type(i), QgsRubberBand)]
        for line in line_items:
            if line in self.mapCanvas.scene().items():
                self.mapCanvas.scene().removeItem(line)                

    def deletePoints(self):
        vertex_items = [ i for i in self.mapCanvas.scene().items() if issubclass(type(i), QgsVertexMarker)]
        for ver in vertex_items:
            if ver in self.mapCanvas.scene().items():
                self.mapCanvas.scene().removeItem(ver)        

    def deactivate(self):
        #self.m.reset(QgsWkbTypes.PointGeometry)
        super(CoordTool, self).deactivate()










"""


cst_points = QgsVectorLayer(r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_intersect_points.shp' , "cst_points", "ogr")
dist = 100.
off_point = QgsPointXY(-5.182, 5.088) #Point(-5.182, 5.088)
for feature in cst_points.getFeatures():
    geom = feature.geometry()
    distance = off_point.distance(geom.asPoint())
    if distance < dist:
        cs_point = geom.asPoint()

#   create a vertices layer from the coastline shapefile
param_ext_vert = {'INPUT' : r'G:\_modelbuilder\swampy\data\_coastline.shp',
                  'OUTPUT' : r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_vertices.shp'}
processing.run("qgis:extractvertices", param_ext_vert)

#   select the closest 100 vertices to the coastal point - just to be sure we cover enough
#       first create a buffer around the coastal point
cst_buffer = QgsGeometry.fromPointXY(cs_point).buffer(1, -1)
#       get the vertices that are within that buffer
param_buff_vert = {'INPUT':r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_vertices.shp',
                   'PREDICATE':[6],
                   'INTERSECT':cst_buffer,
                   'METHOD':0}
processing.run("native:selectbylocation", param_buff_vert)
        
#   select the closest 100 vertices to the coastal point - just to be sure we cover enough
#       first create a buffer around the coastal point
processing.run("native:buffer", {'INPUT':'cs_pt.shp|layername=cs_pt','DISTANCE':1,'SEGMENTS':5,'END_CAP_STYLE':0,
               'JOIN_STYLE':0,'MITER_LIMIT':2,'DISSOLVE':False,'OUTPUT':r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_buff.shp'})
#       clip the vertices layer by the created buffer
processing.run("native:clip", {'INPUT':r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_vertices.shp',
                'OVERLAY':r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_buff.shp',
                'OUTPUT':r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_buff_vertices.shp'})
#       loop through the vertices and extract the vertice id - sort the list next
vert_lay = QgsVectorLayer(r'g:\_modelbuilder\swampy\data\test_model\temp_files\cst_buff_vertices.shp' , "cst_buff_vertices", "ogr")
vert_id_lst = []
for id_vert in vert_lay.getFeatures():
    vert_id_lst.append(id_vert['vertex_ind'])
vert_id_lst.sort()
#       now loop through the list and create a line between pairs of points - check if the cs_pt lies on that line and if yes then stop
for g in range(vert_id_lst[0], vert_id_lst[-1] - 1, 1):
    st_vert_sel = vert_lay.getFeatures(QgsFeatureRequest(QgsExpression('"vertex_ind" = %i' % g)))
    end_vert_sel = vert_lay.getFeatures(QgsFeatureRequest(QgsExpression('"vertex_ind" = %i' % (g + 1))))
    for feature in st_vert_sel:
        st_vert = feature.geometry().asMultiPoint()[0]
    for feature in end_vert_sel:
        end_vert = feature.geometry().asMultiPoint()[0]
    #   check if the cs_point lies on the line between these two points
    line_string = QgsGeometry.fromPolylineXY([st_vert, end_vert])
    #   if the distance from the line is lower than e-10 then assign the st_point to be the triangle point to calculate the cross-section points
    if QgsGeometry.fromPointXY(cs_point).distance(line_string) < 10e-10:
        pt_perpendicular = st_vert
        
#   calculate the distance inland and offshore in the original cross-section, and then recalculate the right position in the perpendicular cross-section
cs_point_utm = utm.from_latlon(cs_point.x(), cs_point.y())
inl_point_utm = utm.from_latlon(-5.196061706, 5.249923448)
off_point_utm = utm.from_latlon(-5.1825631, 5.0879444)
#       the original distances in meters (UTM coordinate system), round the point distances to match the DC values
dc = 100
dl = 10
inl_dist_utm = round(((cs_point_utm[0] - inl_point_utm[0])**2 + (cs_point_utm[1] - inl_point_utm[1])**2)**(.5) / dc, 0) * dc
off_dist_utm = round(((cs_point_utm[0] - off_point_utm[0])**2 + (cs_point_utm[1] - off_point_utm[1])**2)**(.5) / dc, 0) * dc
        

        
"""
