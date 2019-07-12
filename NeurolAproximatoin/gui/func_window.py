from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

from PyQt5 import uic
import numpy as np
import numexpr


class WindowFunction(QWidget):
    data_pipe = QtCore.pyqtSignal(np.ndarray, np.ndarray)

    def __init__(self):
        super(WindowFunction, self).__init__()
        uic.loadUi('gui/draw_function.ui', self)
        self.setFixedSize(self.size())
        self.x = np.ndarray
        self.y = np.ndarray
        self.pushOK.clicked.connect(self.ok)
        self.pushCancel.clicked.connect(self.cancel)

    def ok(self):
        min_x = self.doubleSpinMin.text()
        min_x = float(min_x.replace(',', '.'))
        max_x = self.doubleSpinMax.text()
        max_x = float(max_x.replace(',', '.'))
        number_points = self.spinPoints.text()
        number_points = int(number_points)
        func = self.line_funct.text()
        try:
            x = np.linspace(min_x, max_x, number_points)
            y = numexpr.evaluate(func)
        except:
            return
        self.data_pipe.emit(x, y)
        self.close()

    def cancel(self):
        self.close()
