from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
from PyQt5 import uic
from nn.neurol_network import NeuralNetwork


class WindowSettingsNN(QWidget):
    data_pipe = QtCore.pyqtSignal(NeuralNetwork)

    def __init__(self):
        super(WindowSettingsNN, self).__init__()
        uic.loadUi('gui/window_settings_nn.ui', self)
        self.data_pipe.connect(self.draw_info)

    def draw_info(self, nn):

        i = 1
        for l in nn._layers:
            self.plainWeights.appendPlainText("LAYER - %d\nweights -  %s\nbies - %s " %
                                              (i, str(l.weights), str(l.bias)))
            self.plainSettings.appendPlainText( "LAYER - %d, inputs - %d neurons - %d activate - %s" % (
                i, l.inputs, l.neurons, l.activation)
            )
            i+=1
