import PyQt5.QtWidgets
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QFileDialog
from .nn_thread import NN_Thread
from .nn_window import WindowNN
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from .gui_mainwindow import Ui_MainWindow
from .func_window import WindowFunction
from .window_random import WindowRandom
from .window_setting import WindowSettingsNN
import numpy as np
from copy import copy
from nn.neurol_network import NeuralNetwork
from PyQt5 import uic


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        uic.loadUi('gui/mainwindow.ui', self)
        self.fig = Figure()
        # self.setupUi(self)
        self.axs = self.fig.subplots(2, 1)
        self.ln1, = self.axs[0].plot([], [], '-b')
        self.ln2, = self.axs[0].plot([], [], '-r')
        self.ln3, = self.axs[1].plot([], [], '-g')
        self.errors = []
        self.iters = []
        self.iter = 0
        self.axs[0].grid()
        self.axs[1].grid()
        self.layout_plot = PyQt5.QtWidgets.QVBoxLayout(self.plotWidget)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout_plot.addWidget(self.canvas)
        self.layout_plot.addWidget(self.toolbar)
        self.nn_thread = NN_Thread()
        self.nn_thread.pipe_data.connect(self.plot_animation)
        self.x = None
        self.y = None
        self.nns_list = []
        self.pushPlotGraph.clicked.connect(self.show_window_func)
        self.pushTrain.clicked.connect(self.start_train)
        self.pushClear.clicked.connect(self.update_data)
        self.pushCreateNN.clicked.connect(self.show_window_nn)
        self.pushDeleteNN.clicked.connect(self.delete_nn)
        self.pushStop.clicked.connect(self.stop_train)
        self.func_window = WindowFunction()
        self.func_window.data_pipe.connect(self.plot_function)
        self.pushReadFromFile.clicked.connect(self.read_from_file)
        self.pushPredict.clicked.connect(self.show_window_random)
        self.window_random = WindowRandom()
        self.window_random.data_pipe.connect(self.get_predict)
        self.pushNNInfo.clicked.connect(self.show_window_setting_nn)

    def show_window_setting_nn(self):
        self.window_settings = WindowSettingsNN()
        self.window_settings.show()
        nn_index = self.listWidgetNNs.row(self.listWidgetNNs.currentItem())
        if not self.nn_thread.is_running and nn_index is not -1:
            self.window_settings.data_pipe.emit(copy(self.nns_list[nn_index]))

    def show_window_random(self):
        self.window_random.show()

    def get_predict(self, x):
        nn_index = self.listWidgetNNs.row(self.listWidgetNNs.currentItem())
        if not self.nn_thread.is_running and nn_index is not -1:
            if x is None or self.y is None:
                return
            try:
                cur_nn = copy(self.nns_list[nn_index])
                x_norm = self.normalize_data(x)
                pred_y = cur_nn.predict(x_norm.reshape(len(x), 1))
                norm_y = self.normalize_data(self.y, pred_y)

                self.update_data()
                self.draw(self.ln1, self.x, self.y, self.axs[0])
                self.draw(self.ln2, x.reshape(len(x), 1), norm_y, self.axs[0])
                self.plainTextErrors.appendPlainText(
                "AMSE = %f" % (cur_nn.accuracy(norm_y, self.y))
                )
            except: return

    def add_nn(self, name, list_nn):
        nn = NeuralNetwork()
        for layer in list_nn:
            nn.add_layer(layer)
        self.nns_list.append(nn)
        self.listWidgetNNs.addItem(name)

    def read_from_file(self):
        f = QFileDialog.getOpenFileName(self, 'Open file', '.')
        try:
            file = open(f[0], 'r')
            str_x = file.readline().replace(',', '.')
            str_y = file.readline().replace(',', '.')
            x = (str_x.split(' '))
            x.sort()
            y = (str_y.split(' '))
            x = list(map(float, x))
            y = list(map(float, y))
        except:
            return

        self.x = np.array(x)
        self.y = np.array(y)
        self.plot_function(self.x, self.y)

    def delete_nn(self):
        if self.listWidgetNNs.currentItem() is not None:
            self.listWidgetNNs.takeItem(
                self.listWidgetNNs.row(self.listWidgetNNs.currentItem())
            )
            index = self.listWidgetNNs.row(self.listWidgetNNs.currentItem())
            self.nns_list.pop(index)

    def stop_train(self):
        if self.nn_thread.is_running:
            self.nn_thread.terminate()

    def show_window_nn(self):
        self.nn_window = WindowNN()
        self.nn_window.data_pipe.connect(self.add_nn)
        self.nn_window.show()

    def show_window_func(self):
        self.func_window.show()

    def update_data(self):
        if self.nn_thread.is_running:
            return
        self.axs[0].legend(labels='')
        self.axs[1].legend(labels='')
        self.axs[0].plot([], [])
        self.axs[1].plot([], [])
        self.ln1.set_data([], [])
        self.ln2.set_data([], [])
        self.ln3.set_data([], [])
        self.canvas.draw()
        self.canvas.flush_events()
        self.errors.clear()
        self.iters.clear()
        self.iter = 0
        self.plainTextErrors.clear()

    def normalize_data(self, x, norm_x=None):
        if norm_x is None:
            rez = (x - min(x)) / (max(x) - min(x))
        else:
            rez = min(x) + norm_x * (max(x) - min(x))
        return rez

    def plot_function(self, x, y):
        self.update_data()
        self.x = x.reshape(len(x), 1)
        self.y = y.reshape(len(y), 1)
        self.draw(self.ln1, x, y, self.axs[0])

    def draw(self, ln, x, y, axs):
        ln.set_data(x, y)
        axs.relim(visible_only=True)
        axs.autoscale_view(True)
        self.canvas.draw()
        self.canvas.flush_events()

    def start_train(self):
        nn_index = self.listWidgetNNs.row(self.listWidgetNNs.currentItem())

        if not self.nn_thread.is_running and nn_index is not -1:
            if self.x is None or self.y is None:
                return

            self.update_data()
            cur_nn = copy(self.nns_list[nn_index])
            epochs = int(self.spinEpochs.text())
            lr = self.doubleSpinLr.text()
            lr = float(lr.replace(',', '.'))
            try:
                new_x = self.normalize_data(self.x)
                new_y = self.normalize_data(self.y)
                self.axs[0].legend(('Origin', 'Approximate'), loc='upper right')
                self.axs[1].legend('E', loc='upper right')
                self.draw(self.ln1, new_x, new_y, self.axs[0])

                self.nn_thread.run(cur_nn, new_x.reshape(len(new_x), 1),
                                   new_y.reshape(len(new_y), 1), lr, epochs)
                predict_y = cur_nn.predict(new_x)
                norm_y = self.normalize_data(self.y, predict_y)
                norm_x = self.normalize_data(self.x, new_x)
                self.draw(self.ln1, self.x, self.y, self.axs[0])
                self.draw(self.ln2, norm_x, norm_y, self.axs[0])
                self.plainTextErrors.appendPlainText(
                "Time = %f AMSE = %f" % (cur_nn.time, cur_nn.accuracy(norm_y, self.y))
                )

            except: return

    def plot_animation(self, x, y, error):
        try:
            self.errors.append(error)
            self.plainTextErrors.appendPlainText(
                'Epoch: #%s, MSE: %f' % (
                    self.iter, float(error))
            )
            self.iters.append(self.iter)
            self.ln2.set_data(x, y)
            self.ln3.set_data(self.iters, self.errors)
            self.iter += 10
            self.axs[1].set_xlabel("MSE {}".format(error))
            self.axs[1].relim(visible_only=True)
            self.axs[1].autoscale_view(True)
            self.canvas.draw()
            self.canvas.flush_events()
        except:
            return
