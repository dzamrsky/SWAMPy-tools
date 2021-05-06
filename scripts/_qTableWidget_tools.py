# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:23:23 2021

@author: daniel
"""

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import pandas as pd
import math
import csv

#   create a class for the table widget
class BC_inputTable(QtWidgets.QTableWidget):
    def __init__(self, rch_files_lst, parent=None):
        super(BC_inputTable, self).__init__(0, 21, parent)
        #self.setFont(QtGui.QFont("Helvetica", 10, QtGui.QFont.Normal, italic = False))   
        #   define header titles
        self.headertitle = ('ID', 'Name', 'Duration (yrs)', 'TS total', 'TS duration (yrs)', 'Sea-level (m)', 'RCH type', 'RCH mean (m/d)', 'RCH stdev (m/d)', 'RCH datasource',\
                       'DRN type', 'DRN elev.', 'DRN conductance', 'BC inland type', 'BC inland head elev.', 'BC inland conductance', 'BC inland conc.', 'BC offshore type',\
                       'BC offshore head elev.', 'BC offshore conductance', 'BC offshore conc.')
        self.setHorizontalHeaderLabels(self.headertitle)
        self.verticalHeader().hide()
        stylesheet = "::section{Background-color:rgb(202,224,252); border: 1px solid black; border-radius: 1px;}"
        self.horizontalHeader().setStyleSheet(stylesheet)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        
        #   set the initial column width
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.setColumnWidth(0, 30)
        self.setColumnWidth(1, 60)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 60)        
        self.setColumnWidth(4, 100)          
        self.setColumnWidth(5, 100)
        self.setColumnWidth(6, 100)        
        self.setColumnWidth(7, 120)    
        self.setColumnWidth(8, 120)
        self.setColumnWidth(9, 120)        
        self.setColumnWidth(10, 100)  
        self.setColumnWidth(11, 70)
        self.setColumnWidth(12, 120)        
        self.setColumnWidth(13, 120)  
        self.setColumnWidth(14, 120)
        self.setColumnWidth(15, 140)        
        self.setColumnWidth(16, 140)  
        self.setColumnWidth(17, 140)        
        self.setColumnWidth(18, 140)         
        self.setColumnWidth(19, 140)        
        self.setColumnWidth(20, 120)   

        self.rch_files_lst = rch_files_lst
    
        """
        #   set the first index to be 0
        item = QtWidgets.QTableWidgetItem(str(0)) # create the item
        item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
        self.setItem(0, 0, item)

        #   set the time step duration to be inactive
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        item.setBackground(QtGui.QColor(220, 220, 220))
        self.setItem(0, 4, item)

        #   add combobox for RCH type
        combox_rch = QtWidgets.QComboBox(self)
        combox_rch.addItems(['None', 'Randomized', 'Datasource'])
        self.setCellWidget(0, 6, combox_rch)
        self.cellChanged.connect(self._cellclicked_rch)

        #   add combobox for DRN type
        combox_drn = QtWidgets.QComboBox(self)
        combox_drn.addItems(['None', 'Constant'])
        self.setCellWidget(0, 10, combox_drn)
        self.cellChanged.connect(self._cellclicked_drn)

        #   add combobox for inland BC
        combox_inl_bc = QtWidgets.QComboBox(self)
        combox_inl_bc.addItems(['None', 'GHB', 'CHD'])
        self.setCellWidget(0, 13, combox_inl_bc)
        self.cellChanged.connect(self._cellclicked_inl_bc)

        #   add combobox for offshore BC
        combox_off_bc = QtWidgets.QComboBox(self)
        combox_off_bc.addItems(['None', 'GHB', 'CHD'])
        self.setCellWidget(0, 17, combox_off_bc)
        self.cellChanged.connect(self._cellclicked_off_bc)   
        """
        
        #item = QtWidgets.QTableWidgetItem(self.cellWidget(0, 6).currentText())
        #item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
        #self.setItem(0, 7, QtWidgets.QTableWidgetItem(self.cellWidget(0, 6).currentText()))
        
    @QtCore.pyqtSlot(int, int)
    def _cellclicked_rch(self, r, c):
        it = self.item(r, c)
        it.setTextAlignment(QtCore.Qt.AlignCenter)        

    def _cellclicked_rch_files(self, r, c):
        it = self.item(r, c)
        it.setTextAlignment(QtCore.Qt.AlignCenter)   
        
    def _cellclicked_drn(self, r, c):
        it = self.item(r, c)
        it.setTextAlignment(QtCore.Qt.AlignCenter)   

    def _cellclicked_inl_bc(self, r, c):
        it = self.item(r, c)
        it.setTextAlignment(QtCore.Qt.AlignCenter)   

    def _cellclicked_off_bc(self, r, c):
        it = self.item(r, c)
        it.setTextAlignment(QtCore.Qt.AlignCenter)   

    #   adding rows from csv file
    def _addrow_from_csv(self, csv_dir, rch_lst):
        #   first clean the tablewidget 
        self.setRowCount(0)
        self.rch_files_lst = rch_lst
        
        df_in = pd.read_csv(csv_dir)#r'g:\_modelbuilder\swampy\data\test_model6\temp_files\test_model6_SP_input.csv')
        for index, row in df_in.iterrows():
            #   add new row 
            self._addrow()
            
            #   set the first index to be 0
            item = QtWidgets.QTableWidgetItem(str(index)) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 0, item)
            
            #   the name of the stress period - if its nan then change it to ''
            if math.isnan(row[1]):
                name_item = ''
            else:
                name_item = str(row[1])
            item = QtWidgets.QTableWidgetItem(name_item) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 1, item)            

            #   the duration of the stress period
            item = QtWidgets.QTableWidgetItem(str(row[2])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 2, item)

            #   total amount of time steps of the stress period
            item = QtWidgets.QTableWidgetItem(str(row[3])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 3, item)
            
            #   calculate the time step duration
            item = QtWidgets.QTableWidgetItem(str(row[2] / row[3])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.setItem(index, 4, item)

            #   sea-level relative to current state (+ means sea-level rise, - means lower sea-level)
            item = QtWidgets.QTableWidgetItem(str(row[5])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 5, item)
            
            #   RCH type
            combox_rch = QtWidgets.QComboBox(self)
            combox_rch.addItems(['None', 'Randomized', 'Datasource'])
            combox_rch.setCurrentIndex(['None', 'Randomized', 'Datasource'].index(row[6]))
            self.setCellWidget(index, 6, combox_rch)
            self.cellChanged.connect(self._cellclicked_rch)            
            
            #   mean RCH value
            item = QtWidgets.QTableWidgetItem(str(row[7])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 7, item)            
            
            #   stdev RCH value
            item = QtWidgets.QTableWidgetItem(str(row[8])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 8, item)                    

            #   RCH file combobox
            combox_rch_files = QtWidgets.QComboBox(self)
            combox_rch_files.addItems(rch_lst)
            combox_rch_files.setCurrentIndex(0)
            self.setCellWidget(index, 9, combox_rch_files)
            self.cellChanged.connect(self._cellclicked_rch_files) 

            """
            if math.isnan(row[9]):
                rchsource_item = ''
            else:
                rchsource_item = str(row[9])
            item = QtWidgets.QTableWidgetItem(rchsource_item) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 9, item)     
            """
            
            #   DRN type
            combox_drn = QtWidgets.QComboBox(self)
            combox_drn.addItems(['None', 'Constant'])
            combox_drn.setCurrentIndex(['None', 'Constant'].index(row[10]))
            self.setCellWidget(index, 10, combox_drn)
            self.cellChanged.connect(self._cellclicked_drn)

            #   DRN elevation
            item = QtWidgets.QTableWidgetItem(str(row[11])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 11, item)            
            
            #   DRN conductance
            item = QtWidgets.QTableWidgetItem(str(row[12])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 12, item)   

            #   BC inland type
            combox_inl_bc = QtWidgets.QComboBox(self)
            combox_inl_bc.addItems(['None', 'GHB', 'CHD'])
            combox_inl_bc.setCurrentIndex(['None', 'GHB', 'CHD'].index(row[13]))
            self.setCellWidget(index, 13, combox_inl_bc)
            self.cellChanged.connect(self._cellclicked_inl_bc)

            #   BC inland head elev.
            item = QtWidgets.QTableWidgetItem(str(row[14])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 14, item)            
            
            #   BC inland conductance
            item = QtWidgets.QTableWidgetItem(str(row[15])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 15, item)   

            #   BC inland conc.
            item = QtWidgets.QTableWidgetItem(str(row[16])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 16, item)   

            #   BC offshore type
            combox_off_bc = QtWidgets.QComboBox(self)
            combox_off_bc.addItems(['None', 'GHB', 'CHD'])
            combox_off_bc.setCurrentIndex(['None', 'GHB', 'CHD'].index(row[17]))
            self.setCellWidget(index, 17, combox_off_bc)
            self.cellChanged.connect(self._cellclicked_off_bc)  

            #   BC offshore head elev.
            item = QtWidgets.QTableWidgetItem(str(row[18])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 18, item)            
            
            #   BC offshore conductance
            item = QtWidgets.QTableWidgetItem(str(row[19])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 19, item)   

            #   BC offshore conc.
            item = QtWidgets.QTableWidgetItem(str(row[20])) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(index, 20, item)               
            
    @QtCore.pyqtSlot()
    def _addrow(self):
        rowcount = self.rowCount()
        
        #   set the ID number
        self.insertRow(rowcount)        
        self.setItem(rowcount, 0, QtWidgets.QTableWidgetItem(str(rowcount)))

        #   set the time step duration to be inactive
        item = QtWidgets.QTableWidgetItem()
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        self.setItem(rowcount, 4, item)

        #   add combobox for RCH type
        combox_rch = QtWidgets.QComboBox(self)
        combox_rch.addItems(['None', 'Randomized', 'Datasource'])
        self.setCellWidget(rowcount, 6, combox_rch)
        self.cellChanged.connect(self._cellclicked_rch)

        #   add combobox for RCH files
        combox_rch_files = QtWidgets.QComboBox(self)
        combox_rch_files.addItems(self.rch_files_lst)
        self.setCellWidget(rowcount, 9, combox_rch_files)
        self.cellChanged.connect(self._cellclicked_rch_files)

        #   add combobox for DRN type
        combox_drn = QtWidgets.QComboBox(self)
        combox_drn.addItems(['None', 'Constant'])
        self.setCellWidget(rowcount, 10, combox_drn)
        self.cellChanged.connect(self._cellclicked_drn)

        #   add combobox for inland BC
        combox_inl_bc = QtWidgets.QComboBox(self)
        combox_inl_bc.addItems(['None', 'GHB', 'CHD'])
        self.setCellWidget(rowcount, 13, combox_inl_bc)
        self.cellChanged.connect(self._cellclicked_inl_bc)

        #   add combobox for offshore BC
        combox_off_bc = QtWidgets.QComboBox(self)
        combox_off_bc.addItems(['None', 'GHB', 'CHD'])
        self.setCellWidget(rowcount, 17, combox_off_bc)
        self.cellChanged.connect(self._cellclicked_off_bc)

    @QtCore.pyqtSlot()
    def _removerow(self):
        if self.rowCount() > 0:
            self.removeRow(self.rowCount() - 1)


    @QtCore.pyqtSlot()
    def _copyrow(self, ):
        rowcount = self.rowCount()
        
        if rowcount > 0:

            #   get values from the last row
            fields = []
            for columnNumber in range(self.columnCount()):
                widget = self.cellWidget(rowcount - 1, columnNumber)
                if isinstance(widget, QtWidgets.QComboBox):
                    current_value = widget.currentText()
                    fields.append(current_value)
                else:
                    if self.item(rowcount - 1, columnNumber) is not None:
                        fields.append(self.item(rowcount - 1, columnNumber).text())
                    else:
                        fields.append("")

            #   set the ID number
            self.insertRow(rowcount)        
            self.setItem(rowcount, 0, QtWidgets.QTableWidgetItem(str(rowcount)))

            #   the name of the stress period - if its nan then change it to ''
            name_item = fields[1]
            item = QtWidgets.QTableWidgetItem(name_item) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 1, item)    

            #   the duration of the stress period
            item = QtWidgets.QTableWidgetItem(fields[2]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 2, item)

            #   total amount of time steps of the stress period
            item = QtWidgets.QTableWidgetItem(fields[3]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 3, item)
            
            #   calculate the time step duration
            item = QtWidgets.QTableWidgetItem(fields[4]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.setItem(rowcount, 4, item)

            #   sea-level relative to current state (+ means sea-level rise, - means lower sea-level)
            item = QtWidgets.QTableWidgetItem(fields[5]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 5, item)
 
            #   RCH type
            combox_rch = QtWidgets.QComboBox(self)
            combox_rch.addItems(['None', 'Randomized', 'Datasource'])
            combox_rch.setCurrentIndex(['None', 'Randomized', 'Datasource'].index(fields[6]))
            self.setCellWidget(rowcount, 6, combox_rch)
            self.cellChanged.connect(self._cellclicked_rch)            
            
            #   mean RCH value
            item = QtWidgets.QTableWidgetItem(fields[7]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 7, item)            
            
            #   stdev RCH value
            item = QtWidgets.QTableWidgetItem(fields[8]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 8, item)                    

            #   RCH file combobox
            combox_rch_files = QtWidgets.QComboBox(self)
            combox_rch_files.addItems(self.rch_files_lst)
            combox_rch_files.setCurrentIndex(0)
            self.setCellWidget(rowcount, 9, combox_rch_files)
            self.cellChanged.connect(self._cellclicked_rch_files) 
           
            #   DRN type
            combox_drn = QtWidgets.QComboBox(self)
            combox_drn.addItems(['None', 'Constant'])
            combox_drn.setCurrentIndex(['None', 'Constant'].index(fields[10]))
            self.setCellWidget(rowcount, 10, combox_drn)
            self.cellChanged.connect(self._cellclicked_drn)

            #   DRN elevation
            item = QtWidgets.QTableWidgetItem(fields[11]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 11, item)            
            
            #   DRN conductance
            item = QtWidgets.QTableWidgetItem(fields[12]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 12, item)   

            #   BC inland type
            combox_inl_bc = QtWidgets.QComboBox(self)
            combox_inl_bc.addItems(['None', 'GHB', 'CHD'])
            combox_inl_bc.setCurrentIndex(['None', 'GHB', 'CHD'].index(fields[13]))
            self.setCellWidget(rowcount, 13, combox_inl_bc)
            self.cellChanged.connect(self._cellclicked_inl_bc)

            #   BC inland head elev.
            item = QtWidgets.QTableWidgetItem(fields[14]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 14, item)            
            
            #   BC inland conductance
            item = QtWidgets.QTableWidgetItem(fields[15]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 15, item)   

            #   BC inland conc.
            item = QtWidgets.QTableWidgetItem(fields[16]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 16, item)   

            #   BC offshore type
            combox_off_bc = QtWidgets.QComboBox(self)
            combox_off_bc.addItems(['None', 'GHB', 'CHD'])
            combox_off_bc.setCurrentIndex(['None', 'GHB', 'CHD'].index(fields[17]))
            self.setCellWidget(rowcount, 17, combox_off_bc)
            self.cellChanged.connect(self._cellclicked_off_bc)  

            #   BC offshore head elev.
            item = QtWidgets.QTableWidgetItem(fields[18]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 18, item)            
            
            #   BC offshore conductance
            item = QtWidgets.QTableWidgetItem(fields[19]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 19, item)   

            #   BC offshore conc.
            item = QtWidgets.QTableWidgetItem(fields[20]) # create the item
            item.setTextAlignment(QtCore.Qt.AlignCenter) # change the alignment
            self.setItem(rowcount, 20, item)               

    #   saving the table into csv
    def _save_to_csv(self, csv_dir):
        #   open the csv and create a writer object        
        with open(csv_dir, "w", newline='') as fileOutput:
            writer = csv.writer(fileOutput)
            #   first write the header
            writer.writerow(self.headertitle) 
            #   loop through the rows one by one
            for rowNumber in range(self.rowCount()):
                fields = []
                for columnNumber in range(self.columnCount()):
                    widget = self.cellWidget(rowNumber, columnNumber)
                    if isinstance(widget, QtWidgets.QComboBox):
                        current_value = widget.currentText()
                        fields.append(current_value)
                    else:
                        if self.item(rowNumber, columnNumber) is not None:
                            fields.append(self.item(rowNumber, columnNumber).text())
                        else:
                            fields.append("")
                writer.writerow(fields)    


#   create a class for the table widget
class SEAWAT_parameters_testTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super(SEAWAT_parameters_testTable, self).__init__(0, 17, parent)
        #self.setFont(QtGui.QFont("Helvetica", 10, QtGui.QFont.Normal, italic = False))   
        #   define header titles
        self.headertitle = ('SP', 'DIS', 'BAS', 'LPF', 'GHB/CHD', 'RCH', 'DRN', 'PCG', 'OC', 'BTN', 'ADV', 'DSP', 'GCG', 'VDF', 'SSM', 'Converged', 'Runtime (min)')
        self.setHorizontalHeaderLabels(self.headertitle)
        self.verticalHeader().hide()
        stylesheet_header = "::section{Background-color:rgb(202,224,252); border: 1px solid black; border-radius: 1px;}"
        #stylesheet_ok = "::section{Background-color:rgb(202,224,252); border: 1px solid black; border-radius: 1px;}"
        #stylesheet_fail = "::section{Background-color:rgb(202,224,252); border: 1px solid black; border-radius: 1px;}"
        self.horizontalHeader().setStyleSheet(stylesheet_header)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Fixed)
        #   set the initial column width
        self.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        width_pckg = 60
        width = 70
        self.setColumnWidth(0, width_pckg)
        self.setColumnWidth(1, width_pckg)
        self.setColumnWidth(2, width_pckg)
        self.setColumnWidth(3, width_pckg)        
        self.setColumnWidth(4, width_pckg)          
        self.setColumnWidth(5, width_pckg)
        self.setColumnWidth(6, width_pckg)        
        self.setColumnWidth(7, width_pckg)    
        self.setColumnWidth(8, width_pckg)
        self.setColumnWidth(9, width_pckg)        
        self.setColumnWidth(10, width_pckg)  
        self.setColumnWidth(11, width_pckg)
        self.setColumnWidth(12, width_pckg)        
        self.setColumnWidth(13, width_pckg)  
        self.setColumnWidth(14, width_pckg)
        self.setColumnWidth(15, width)  
        self.setColumnWidth(16, width + 20)

    def add_row_initial(self):
        #   first clean the tablewidget 
        self.setRowCount(0)
        #   set the ID number
        self.insertRow(self.rowCount())  
        for i in range(self.rowCount()):
            it = QtWidgets.QTableWidgetItem("")
            it.setData(QtCore.Qt.UserRole, 'red')
            self.setItem(self.rowCount(), i, it) 
            
    #   add the testing row - just one
    def add_row_testing(self, lst_in):
        #   first clean the tablewidget 
        self.setRowCount(0)
        rowcount = self.rowCount()
        #   set the ID number
        self.insertRow(rowcount)  
        #   add the values from list one by one
        it = QtWidgets.QTableWidgetItem(str(lst_in[0]))
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        self.setItem(rowcount, 0, it)             

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[1] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 1, it)
        
        it = QtWidgets.QTableWidgetItem('')
        if lst_in[2] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 2, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[3] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 3, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[4] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 4, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[5] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 5, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[6] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 6, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[7] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 7, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[8] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 8, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[9] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 9, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[10] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 10, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[11] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 11, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[12] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 12, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[13] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 13, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[14] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 14, it)

        it = QtWidgets.QTableWidgetItem('')
        if lst_in[15] is True:
            it.setBackground(QtGui.QColor(0, 255, 0))
        else:
            it.setBackground(QtGui.QColor(255, 0, 0))
        self.setItem(rowcount, 15, it)
        
        it = QtWidgets.QTableWidgetItem(str(lst_in[16]))
        it.setTextAlignment(QtCore.Qt.AlignCenter)
        self.setItem(rowcount, 16, it)      

"""
class ThirdTabLoads(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ThirdTabLoads, self).__init__(parent)    

        table = BC_inputTable()

        add_button = QtWidgets.QPushButton("Add")
        add_button.clicked.connect(table._addrow)

        add_csv_button = QtWidgets.QPushButton("Add_CSV")
        add_csv_button.clicked.connect(table._addrow_from_csv)

        # (r'g:\_modelbuilder\swampy\data\test_model6\temp_files\test_model6_SP_input.csv')

        delete_button = QtWidgets.QPushButton("Delete")
        delete_button.clicked.connect(table._removerow)

        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addWidget(add_button, alignment=QtCore.Qt.AlignBottom)
        button_layout.addWidget(delete_button, alignment=QtCore.Qt.AlignTop)
        button_layout.addWidget(add_csv_button, alignment=QtCore.Qt.AlignTop)

        tablehbox = QtWidgets.QHBoxLayout()
        tablehbox.setContentsMargins(10, 10, 10, 10)
        tablehbox.addWidget(table)

        grid = QtWidgets.QGridLayout(self)
        grid.addLayout(button_layout, 0, 1)
        grid.addLayout(tablehbox, 0, 0)        

if __name__ == '__main__':
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainWin = ThirdTabLoads()
        mainWin.show()
        app.exec_()
    run_app()   
"""


    
    