# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LLAMA_UI.ui'
#
# Created by: PyQt4 UI code generator 4.11.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_Lake_Level_UI(object):
    def setupUi(self, Lake_Level_UI):

        #Overall UI form attributes
        Lake_Level_UI.setObjectName(_fromUtf8("Lake_Level_UI"))
        Lake_Level_UI.setEnabled(True)
        Lake_Level_UI.resize(360, 360)
        Lake_Level_UI.setMinimumSize(QtCore.QSize(360, 360))
        Lake_Level_UI.setMaximumSize(QtCore.QSize(360, 360))

        #Algae Checkbox (raster option)
        self.algaeCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.algaeCheckbox.setGeometry(QtCore.QRect(60, 170, 151, 17))
        self.algaeCheckbox.setObjectName(_fromUtf8("algaeCheckbox"))

        #Turbidity Checkbox (raster option)
        self.turbidityCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.turbidityCheckbox.setGeometry(QtCore.QRect(60, 150, 131, 17))
        self.turbidityCheckbox.setObjectName(_fromUtf8("turbidityCheckbox"))

        #Graph Checkbox (chart option)
        self.graphCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.graphCheckbox.setGeometry(QtCore.QRect(60, 240, 131, 17))
        self.graphCheckbox.setObjectName(_fromUtf8("graphCheckbox"))

        #Table Checkbox (chart option)
        self.tableCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.tableCheckbox.setGeometry(QtCore.QRect(60, 260, 161, 17))
        self.tableCheckbox.setObjectName(_fromUtf8("tableCheckbox"))

        #Heatmap Checkbox (raster option)
        self.heatmapCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.heatmapCheckbox.setGeometry(QtCore.QRect(60, 130, 151, 17))
        self.heatmapCheckbox.setObjectName(_fromUtf8("heatmapCheckbox"))

        #Label/box for the raster options
        self.groupBox = QtGui.QGroupBox(Lake_Level_UI)
        self.groupBox.setGeometry(QtCore.QRect(20, 110, 321, 91))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))

        #Label/box for the chart options
        self.groupBox_2 = QtGui.QGroupBox(Lake_Level_UI)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 220, 321, 71))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))

        #OK button for overall UI
        self.okBtn = QtGui.QPushButton(Lake_Level_UI)
        self.okBtn.setGeometry(QtCore.QRect(100, 310, 75, 23))
        self.okBtn.setObjectName(_fromUtf8("okBtn"))

        #Cancel button
        self.cancelBtn = QtGui.QPushButton(Lake_Level_UI)
        self.cancelBtn.setGeometry(QtCore.QRect(180, 310, 75, 23))
        self.cancelBtn.setObjectName(_fromUtf8("cancelBtn"))

        #Calendar for end date
        self.endDate = QtGui.QDateEdit(Lake_Level_UI)
        self.endDate.setGeometry(QtCore.QRect(250, 71, 83, 20))
        self.endDate.setDate(QtCore.QDate(2015, 1, 1))
        self.endDate.setCalendarPopup(True)
        self.endDate.setObjectName(_fromUtf8("endDate"))

        #Label for start date
        self.lblStartDate = QtGui.QLabel(Lake_Level_UI)
        self.lblStartDate.setGeometry(QtCore.QRect(31, 73, 54, 16))
        self.lblStartDate.setObjectName(_fromUtf8("lblStartDate"))

        #Calendar for start date
        self.startDate = QtGui.QDateEdit(Lake_Level_UI)
        self.startDate.setGeometry(QtCore.QRect(91, 71, 83, 20))
        self.startDate.setDate(QtCore.QDate(1984, 4, 25))
        self.startDate.setMinimumDate(QtCore.QDate(1984, 4, 1))
        self.startDate.setCalendarPopup(True)
        self.startDate.setObjectName(_fromUtf8("startDate"))

        #Drop-down menu to select lake
        self.selectlakeDropMenu = QtGui.QComboBox(Lake_Level_UI)
        self.selectlakeDropMenu.setGeometry(QtCore.QRect(30, 40, 301, 20))
        self.selectlakeDropMenu.setObjectName(_fromUtf8("selectlakeDropMenu"))
        self.selectlakeDropMenu.addItem(_fromUtf8(""))
        self.selectlakeDropMenu.addItem(_fromUtf8(""))

        #Label/box for dropdown menu and calendar dates
        self.groupBox_3 = QtGui.QGroupBox(Lake_Level_UI)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 20, 321, 81))
        self.groupBox_3.setObjectName(_fromUtf8("groupBox_3"))

        #Label for end date
        self.lblEndDate = QtGui.QLabel(self.groupBox_3)
        self.lblEndDate.setGeometry(QtCore.QRect(177, 54, 48, 16))
        self.lblEndDate.setObjectName(_fromUtf8("lblEndDate"))


        self.groupBox_3.raise_()
        self.endDate.raise_()
        self.lblStartDate.raise_()
        self.startDate.raise_()
        self.selectlakeDropMenu.raise_()
        self.groupBox_2.raise_()
        self.groupBox.raise_()
        self.algaeCheckbox.raise_()
        self.turbidityCheckbox.raise_()
        self.graphCheckbox.raise_()
        self.tableCheckbox.raise_()
        self.heatmapCheckbox.raise_()
        self.okBtn.raise_()
        self.cancelBtn.raise_()

        self.retranslateUi(Lake_Level_UI)
        QtCore.QObject.connect(self.cancelBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), Lake_Level_UI.close)
        QtCore.QMetaObject.connectSlotsByName(Lake_Level_UI)

    def retranslateUi(self, Lake_Level_UI):
        Lake_Level_UI.setWindowTitle(_translate("Lake_Level_UI", "LLAMA", None))
        self.algaeCheckbox.setText(_translate("Lake_Level_UI", "Algae", None))
        self.turbidityCheckbox.setText(_translate("Lake_Level_UI", "Turbidity", None))
        self.graphCheckbox.setText(_translate("Lake_Level_UI", "Graph", None))
        self.tableCheckbox.setText(_translate("Lake_Level_UI", "Table", None))
        self.heatmapCheckbox.setText(_translate("Lake_Level_UI", "Heat Map", None))
        self.groupBox.setTitle(_translate("Lake_Level_UI", "Select Rasters", None))
        self.groupBox_2.setTitle(_translate("Lake_Level_UI", "Select Lake Area Charts", None))
        self.okBtn.setText(_translate("Lake_Level_UI", "OK", None))
        self.cancelBtn.setText(_translate("Lake_Level_UI", "Cancel", None))
        self.endDate.setDisplayFormat(_translate("Lake_Level_UI", "M/d/yyyy", None))
        self.lblStartDate.setText(_translate("Lake_Level_UI", "Start Date:", None))
        self.selectlakeDropMenu.setItemText(0, _translate("Lake_Level_UI", "Lake Tahoe", None))
        self.selectlakeDropMenu.setItemText(1, _translate("Lake_Level_UI", "Fallen Leaf", None))
        self.groupBox_3.setTitle(_translate("Lake_Level_UI", "Measure Area for...", None))
        self.lblEndDate.setText(_translate("Lake_Level_UI", "End Date:", None))
