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

from PyQt4 import QtGui, QtCore
import sys
from threading import Thread
from LLAMA import Ui_Lake_Level_UI
from plot_water_levelui import *
from lake_measure import *

class ProgressPopup(QtGui.QWidget):

    update_signal = QtCore.pyqtSignal(int, int, str, str, int, int)
    def __init__(self, cancel_function):
        QtGui.QWidget.__init__(self)

        self.update_signal.connect(self.apply_update, QtCore.Qt.QueuedConnection)
        self.cancel_function = cancel_function

        self.lake_totals = None
        self.lake_counts = None

        self.progressBar = QtGui.QProgressBar(self)
        self.progressBar.setMinimumSize(500, 50)
        self.progressBar.setMaximumSize(500, 50)
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)

        self.status = QtGui.QLabel(self)
        self.status.setText("")

        self.cancelButton = QtGui.QPushButton('Cancel', self)
        self.cancelButton.setMinimumSize(50, 30)
        self.cancelButton.setMaximumSize(100, 50)
        self.cancelButton.clicked[bool].connect(self._cancel)

        vbox = QtGui.QVBoxLayout(self)
        vbox.addWidget(self.progressBar)
        vbox.addWidget(self.status)
        vbox.addWidget(self.cancelButton)
        vbox.addStretch(1)

        self.setLayout(vbox)

    def update_function(self, lakes_number, lakes_total, lake_name, lake_date, lake_image, lake_image_total):
        self.update_signal.emit(lakes_number, lakes_total, lake_name, lake_date, lake_image, lake_image_total)

    def apply_update(self, lakes_number, lakes_total, lake_name, lake_date, lake_image, lake_image_total):
        if self.lake_totals == None:
            self.lake_totals = [10] * lakes_total
            self.lake_counts = [0] * lakes_total
        self.lake_totals[lakes_number] = lake_image_total
        self.lake_counts[lakes_number] = lake_image
        total = sum(self.lake_totals)
        progress = sum(self.lake_counts)
        self.status.setText('Completed processing %s on %s.' % (lake_name, lake_date))
        self.progressBar.setValue(float(progress) / total * 100)

    def closeEvent(self, event):
        if self.cancel_function != None:
            self.cancel_function()
        event.accept()

    def _cancel(self):
        self.close()

class Lake_Level_App(QtGui.QMainWindow, Ui_Lake_Level_UI):
    def __init__(self):

        super(self.__class__, self).__init__()
        self.setupUi(self)
        self.start_date = '1984-04-25'
        # Sets end date to current date.
        self.end_date = str((QtCore.QDate.currentDate()).toString('yyyy-MM-dd'))
        self.selected_lake = 'Lake Tahoe'
        self.selectlakeDropMenu.activated[str].connect(self.selectLakeHandle)
        self.okBtn.clicked.connect(self.okHandle)
        # Sets end date as current date. Couldn't set this option in QT Designer
        self.endDate.setDate(QtCore.QDate.currentDate())
        self.endDate.dateChanged[QtCore.QDate].connect(self.endHandle)
        self.startDate.dateChanged[QtCore.QDate].connect(self.startHandle)
        self.faiState = False
        self.ndtiState = False
        self.completedSignal.connect(self.completeLakeThread, QtCore.Qt.QueuedConnection)

    def selectLakeHandle(self, text):
        self.selected_lake = str(text)

    def startHandle(self, date):
        self.start_date = str(date.toString('yyyy-MM-dd'))

    def endHandle(self, date):
        self.end_date = str(date.toString('yyyy-MM-dd'))

    completedSignal = QtCore.pyqtSignal()
    @QtCore.pyqtSlot()
    def completeLakeThread(self):
        if self.tableCheckbox.isChecked():
            table_water_level(self.selected_lake, self.start_date, self.end_date, result_dir='results', output_file=self.table_output_file)
        if self.graphCheckbox.isChecked():
            plot_water_level(self.selected_lake, self.start_date, self.end_date, result_dir='results')
        self.popup.close()

    def okHandle(self):

        if self.algaeCheckbox.isChecked():
            self.faiState = True
        else:
            self.faiState = False

        if self.turbidityCheckbox.isChecked():
            self.ndtiState = True
        else:
            self.ndtiState = False

        # Heat map checkbox is not functioning. Add under here:
        # if self.lake_areaCheckbox.isChecked():

        if self.tableCheckbox.isChecked():
            self.table_output_file = QtGui.QFileDialog.getSaveFileName(self, 'Choose Output File', 'results/' + self.selected_lake + '.csv', 'CSV File (*.csv *.txt)')
        self.popup = ProgressPopup(Lake_Level_Cancel)
        self.lake_thread = Thread(target=Lake_Level_Run, args=(self.selected_lake, self.start_date, self.end_date, \
                       'results', self.faiState, self.ndtiState, self.popup.update_function, self.completedSignal.emit))
        self.popup.show()
        self.lake_thread.start()

        # CHANGE THIS. NEED TO MAKE THESE PARTS WAIT UNTIL LAKE_THREAD IS FINISHED.

def main():
    app = QtGui.QApplication(sys.argv)  # A new instance of QApplication
    form = Lake_Level_App()                 # We set the form to be our ExampleApp (design)
    form.show()                         # Show the form
    app.exec_()                         # and execute the app


if __name__ == '__main__':              # if we're running file directly and not importing it
    main()
