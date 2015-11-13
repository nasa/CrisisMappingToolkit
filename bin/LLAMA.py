# -----------------------------------------------------------------------------
# Copyright * 2014, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration. All
# rights reserved.
#
# The Crisis Mapping Toolkit (CMT) v1 platform is licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.
# -----------------------------------------------------------------------------

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

    def __init__(self):
        super(Ui_Lake_Level_UI, self).__init__()

        self.setupUi()

    def setupUi(self, Lake_Level_UI):

       #Overall UI form attributes
        Lake_Level_UI.setObjectName(_fromUtf8("Lake_Level_UI"))
        Lake_Level_UI.setEnabled(True)
        Lake_Level_UI.resize(360, 360)
        Lake_Level_UI.setMinimumSize(QtCore.QSize(360, 360))
        Lake_Level_UI.setMaximumSize(QtCore.QSize(360, 360))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("../../Downloads/Complete.PNG")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Lake_Level_UI.setWindowIcon(icon)

       #Algae checkbox (index option)
        self.algaeCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.algaeCheckbox.setGeometry(QtCore.QRect(60, 170, 50, 17))
        self.algaeCheckbox.setObjectName(_fromUtf8("algaeCheckbox"))

       #Turbidity checkbox (index option)
        self.turbidityCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.turbidityCheckbox.setGeometry(QtCore.QRect(60, 150, 64, 17))
        self.turbidityCheckbox.setObjectName(_fromUtf8("turbidityCheckbox"))

       #Graph checkbox (one of the options for the output)
        self.graphCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.graphCheckbox.setGeometry(QtCore.QRect(60, 240, 52, 17))
        self.graphCheckbox.setObjectName(_fromUtf8("graphCheckbox"))

       #Table checkbox (one of the options for the output)
        self.tableCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.tableCheckbox.setGeometry(QtCore.QRect(60, 260, 48, 17))
        self.tableCheckbox.setObjectName(_fromUtf8("tableCheckbox"))

       #Lake Area checkbox (index option)
        self.lake_areaCheckbox = QtGui.QCheckBox(Lake_Level_UI)
        self.lake_areaCheckbox.setGeometry(QtCore.QRect(60, 130, 71, 17))
        self.lake_areaCheckbox.setObjectName(_fromUtf8("lake_areaCheckbox"))

       #Label/box for the indices
        self.groupBox = QtGui.QGroupBox(Lake_Level_UI)
        self.groupBox.setGeometry(QtCore.QRect(20, 110, 321, 91))
        self.groupBox.setObjectName(_fromUtf8("groupBox"))

       #Label/box for the output options
        self.groupBox_2 = QtGui.QGroupBox(Lake_Level_UI)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 220, 321, 71))
        self.groupBox_2.setObjectName(_fromUtf8("groupBox_2"))

       #Button for overall UI (processes data)
        self.okBtn = QtGui.QPushButton(Lake_Level_UI)
        self.okBtn.setGeometry(QtCore.QRect(100, 310, 75, 23))
        self.okBtn.setObjectName(_fromUtf8("okBtn"))

       #Button to close the UI
        self.cancelBtn = QtGui.QPushButton(Lake_Level_UI)
        self.cancelBtn.setGeometry(QtCore.QRect(180, 310, 75, 23))
        self.cancelBtn.setObjectName(_fromUtf8("cancelBtn"))

       #Calendar for end date
        self.endDate = QtGui.QDateEdit(Lake_Level_UI)
        self.endDate.setGeometry(QtCore.QRect(255, 71, 83, 20))
        self.endDate.setObjectName(_fromUtf8("endDate"))
       #Label (tells user where to enter end date)
        self.lblEndDate = QtGui.QLabel(Lake_Level_UI)
        #self.endDate.setMinimumDate('1/1/1984')
        #self.endDate.setMaximumDate(QDate.currentDate())
        self.lblEndDate.setGeometry(QtCore.QRect(201, 71, 48, 16))
        self.lblEndDate.setObjectName(_fromUtf8("lblEndDate"))

        #Label (tells user where to enter start date)
        self.lblStartDate = QtGui.QLabel(Lake_Level_UI)
        self.lblStartDate.setGeometry(QtCore.QRect(31, 71, 54, 16))
        self.lblStartDate.setObjectName(_fromUtf8("lblStartDate"))
        #Calendar for start date
        self.startDate = QtGui.QDateEdit(Lake_Level_UI)
        #self.startDate.setMinimumDate('1/1/1984')
        #self.startDate.setMaximumDate(QDate.currentDate())
        self.startDate.setGeometry(QtCore.QRect(91, 71, 83, 20))
        self.startDate.setObjectName(_fromUtf8("startDate"))

       #Label (tells user where to select lake)
        self.lblSelectLake = QtGui.QLabel(Lake_Level_UI)
        self.lblSelectLake.setGeometry(QtCore.QRect(31, 34, 58, 16))
        self.lblSelectLake.setObjectName(_fromUtf8("lblSelectLake"))

       #Drop-down menu to select which lake the user wants
        self.selectlakeDropMenu = QtGui.QComboBox(Lake_Level_UI)
        self.selectlakeDropMenu.setGeometry(QtCore.QRect(95, 31, 80, 20))
        self.selectlakeDropMenu.setObjectName(_fromUtf8("selectlakeDropMenu"))
        self.selectlakeDropMenu.addItem(_fromUtf8(""))
        self.selectlakeDropMenu.addItem(_fromUtf8(""))
        self.selectlakeDropMenu.addItem(_fromUtf8(""))


        self.groupBox_2.raise_()
        self.groupBox.raise_()
        self.algaeCheckbox.raise_()
        self.turbidityCheckbox.raise_()
        self.graphCheckbox.raise_()
        self.tableCheckbox.raise_()
        self.lake_areaCheckbox.raise_()
        self.okBtn.raise_()
        self.cancelBtn.raise_()

        #Closes the UI when the cancel button is clicked
        self.retranslateUi(Lake_Level_UI)
        QtCore.QObject.connect(self.cancelBtn, QtCore.SIGNAL(_fromUtf8("clicked()")), Lake_Level_UI.close)
        QtCore.QMetaObject.connectSlotsByName(Lake_Level_UI)

    def retranslateUi(self, Lake_Level_UI):
        Lake_Level_UI.setWindowTitle(_translate("Lake_Level_UI", "Lake Level Automated Monitoring Algorithm", None))
        self.algaeCheckbox.setText(_translate("Lake_Level_UI", "Algae", None))
        self.turbidityCheckbox.setText(_translate("Lake_Level_UI", "Turbidity", None))
        self.graphCheckbox.setText(_translate("Lake_Level_UI", "Graph", None))
        self.tableCheckbox.setText(_translate("Lake_Level_UI", "Table", None))
        self.lake_areaCheckbox.setText(_translate("Lake_Level_UI", "Lake Area", None))
        self.groupBox.setTitle(_translate("Lake_Level_UI", "Select Measurement:", None))
        self.groupBox_2.setTitle(_translate("Lake_Level_UI", "Select Output", None))
        self.okBtn.setText(_translate("Lake_Level_UI", "OK", None))
        self.cancelBtn.setText(_translate("Lake_Level_UI", "Cancel", None))
        self.lblEndDate.setText(_translate("Lake_Level_UI", "End Date:", None))
        self.lblStartDate.setText(_translate("Lake_Level_UI", "Start Date:", None))
        self.lblSelectLake.setText(_translate("Lake_Level_UI", "Select Lake:", None))
        self.selectlakeDropMenu.setItemText(0, _translate("Lake_Level_UI", "Lake Tahoe", None))
        self.selectlakeDropMenu.setItemText(1, _translate("Lake_Level_UI", "Fallen Leaf", None))
        self.selectlakeDropMenu.setItemText(2, _translate("Lake_Level_UI", "Salt Lake", None))
