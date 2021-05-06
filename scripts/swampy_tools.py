# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 15:07:22 2020

@author: daniel
"""

# -*- coding: utf-8 -*-
#-----------------------------------------------------------
#
# Profile
# Copyright (C) 2008  Borys Jurgiel
# Copyright (C) 2012  Patrice Verchere
#-----------------------------------------------------------
#
# licensed under the terms of GNU GPL 2
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, print to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
#---------------------------------------------------------------------

#Qt import
from qgis.PyQt import uic, QtCore, QtGui
try:
    from qgis.PyQt.QtGui import QWidget
except:
    from qgis.PyQt.QtWidgets import QWidget
from qgis.PyQt.QtSvg import * # required in some distros
#qgis import
import qgis
from qgis.core import *
from qgis.gui import *
#other
from math import sqrt
import numpy as np
#plugin import
import os 

"""
import platform
import sys
from .dataReaderTool import DataReaderTool
from .plottingtool import PlottingTool
from .ptmaptool import ProfiletoolMapTool, ProfiletoolMapToolRenderer
from ..ui.ptdockwidget import PTDockWidget
from . import profilers
from .selectlinetool import SelectLineTool
"""

#uiFilePath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'swampy_dockwidget_base.ui'))
uiFilePath = os.path.abspath(os.path.join(os.path.join(__file__, "../.."), 'SWAMPy_tools_dockwidget_base.ui'))
FormClass = uic.loadUiType(uiFilePath)[0]

class SwampyTools(QWidget, FormClass):

    TITLE = "Swampy"
    TYPE = None

    closed = QtCore.pyqtSignal()
    
    def __init__(self, iface,plugincore, parent = None):
        QWidget.__init__(self, parent)
        self.iface = iface
        self.plugincore = plugincore

        #remimber repository for saving
        if QtCore.QSettings().value("swampy/lastdirectory") != '':
            self.loaddirectory = QtCore.QSettings().value("swampy/lastdirectory")
        else:
            self.loaddirectory = ''

        """
        #mouse tracking
        self.doTracking = False
        #the datas / results
        self.profiles = None        #dictionary where is saved the plotting data {"l":[l],"z":[z], "layer":layer1, "curve":curve1}
        #The line information
        self.pointstoDraw = []
        #he renderer for temporary polyline
        #self.toolrenderer = ProfiletoolMapToolRenderer(self)
        self.toolrenderer = None
        #the maptool previously loaded
        self.saveTool = None                #Save the standard mapttool for restoring it at the end
        # Used to remove highlighting from previously active layer.
        self.previousLayerId = None
        self.x_cursor = None    # Keep track of last x position of cursor
        #the dockwidget
        self.dockwidget = PTDockWidget(self.iface,self)
        # Initialize the dockwidget combo box with the list of available profiles.
        # (Use sorted list to be sure that Height is always on top and
        # the combobox order is consistent)
        for profile in sorted(profilers.PLOT_PROFILERS):
            self.dockwidget.plotComboBox.addItem(profile)
        self.dockwidget.plotComboBox.setCurrentIndex(0)
        self.dockwidget.plotComboBox.currentIndexChanged.connect(
            lambda index: self.plotProfil())
        #dockwidget graph zone
        self.dockwidget.changePlotLibrary( self.dockwidget.cboLibrary.currentIndex() )
        """

    """
    def activateProfileMapTool(self):
        self.saveTool = self.iface.mapCanvas().mapTool()            #Save the standard mapttool for restoring it at the end
        #Listeners of mouse
        self.toolrenderer = ProfiletoolMapToolRenderer(self)
        self.toolrenderer.connectTool()
        self.toolrenderer.setSelectionMethod(
                self.dockwidget.comboBox.currentIndex())
        #init the mouse listener comportement and save the classic to restore it on quit
        self.iface.mapCanvas().setMapTool(self.toolrenderer.tool)
    """

