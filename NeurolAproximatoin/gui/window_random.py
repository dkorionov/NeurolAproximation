from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QFileDialog
import numpy as np
from PyQt5 import uic
import time


class WindowRandom(QWidget):
    data_pipe = QtCore.pyqtSignal(np.ndarray)

    def __init__(self):
        super(WindowRandom, self).__init__()
        self.setFixedSize(400, 400)
        uic.loadUi('gui/window_random.ui', self)
        self.pushRandom.clicked.connect(self.random_data)
        self.pushClear.clicked.connect(self.clear)
        self.pushReadFile.clicked.connect(self.read_file)
        self.pushOk.clicked.connect(self.push_ok)
        self.x = np.ndarray

    def random_data(self):
        self.plainTextRandom.clear()
        np.random.seed(int(time.time()))
        self.x = np.ndarray
        min_x = self.doubleSpinFrom.text().replace(',', '.')
        min_x = float(min_x)
        max_x = self.doubleSpinTo.text().replace(',', '.')
        max_x = float(max_x)
        points = int(self.spinPoints.text())
        random_x = np.random.rand(points) * max_x + min_x
        self.x = np.sort(random_x)
        self.plainTextRandom.appendPlainText(str(self.x))
        # self.data_pipe.emit(random_x)

    def read_file(self):
        try:
            self.plainTextRandom.clear()
            f = QFileDialog.getOpenFileName(self, 'Open file', '.')
            file = open(f[0], 'r')
            self.x = None
            str_x = file.readline().replace(',', '.')
            random_x = (str_x.split(' '))
            random_x.sort()
            random_x = list(map(float, random_x))
            random_x = np.array(random_x)
            self.plainTextRandom.appendPlainText(str(random_x))
            self.x = random_x
        except:
            return

    def clear(self):
        self.plainTextRandom.clear()

    def push_ok(self):
        self.data_pipe.emit(self.x)
        self.close()

