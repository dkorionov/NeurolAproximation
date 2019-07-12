from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget
from nn.layer import Layer

from PyQt5 import uic


class WindowNN(QWidget):
    data_pipe = QtCore.pyqtSignal(str, list)

    def __init__(self):
        super(WindowNN, self).__init__()
        self.layersList = []
        self.inputs = 1
        self.setFixedSize(400, 400)
        uic.loadUi('gui/create_nn.ui', self)
        self.comboListFuncts.addItem(None)
        self.comboListFuncts.addItem('sigmoid')
        self.comboListFuncts.addItem('tanh')
        self.pushAddLayer.clicked.connect(self.add_layer)
        self.pushClose.clicked.connect(self.close)
        self.pushDeleteLayer.clicked.connect(self.delete_layer)
        self.pushAddNN.clicked.connect(self.add_nn)
        self.setWindowTitle("Create Neural Network")


    def delete_layer(self):
        try:
            if self.listWidgetLayers.currentItem() is None:
                self.listWidgetLayers.takeItem(
                    self.listWidgetLayers.row(self.listWidgetLayers.currentItem())
                )
                index = self.listWidgetLayers.row(self.listWidgetLayers.currentItem())
                self.layersList.pop(index)
        except:
            return

    def add_layer(self):
        neurons = int(self.spinCountNeurons.text())
        item = 'inputs - {}, neurons - {}, {} '.format(self.inputs, neurons, self.comboListFuncts.currentText())
        self.listWidgetLayers.addItem(item)
        self.layersList.append(Layer(self.inputs, neurons, self.comboListFuncts.currentText()))
        self.inputs = neurons

    def add_nn(self):
        nn_name = self.lineNameNN.text()
        if nn_name != "" and len(self.layersList) != 0:
            self.data_pipe.emit(nn_name, self.layersList)
        self.close()
