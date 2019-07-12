from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np


class NN_Thread(QThread):
    def __init__(self):
        super().__init__()
        self.is_running = False

    pipe_data = pyqtSignal(np.ndarray, np.ndarray, np.float64)

    def run(self, nn, x, y, lr, epochs):
            self.is_running = True
            nn.train(x, y, lr, epochs, self.pipe_data)
            self.exit(1)
            self.is_running = False


