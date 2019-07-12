from nn.layer import Layer
from nn.neurol_network import NeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from multiprocessing import Process, Queue, Pipe


def update(frame, q, ln1):
    ln1.set_ydata(q.get())
    return ln1,


def animation(q, ln, fig):
    ani = FuncAnimation(fig, update, frames=1000, fargs=(q, ln), interval=2, repeat=True)



def myplot(x, y, q, mses):
    plt.ion()
    fig, axs = plt.subplots(2, 1)
    count = 0
    ydate = np.empty(0)
    axs[0].plot(x, y)
    line1, = axs[0].plot(x, y)
    line2, = axs[1].plot(x, y)
    axs[1].set_ylim(-0.2, 1)
    axs[1].grid()
    axs[0].grid()
    while not q.empty() or not mses.empty():
        #l.acquire()
        line1.set_ydata(q.get())
        data = mses.get()
        ydate = np.append(ydate, data)
        #l.release()
        plt.xlabel('error ' + str(data))
        axs[1].relim(visible_only=True)
        axs[1].autoscale_view(True)
        axs[1].set_xlim(count, count + 10)
        line2.set_data(np.linspace(count, count + 10, len(ydate)), ydate)
        fig.canvas.draw()
        fig.canvas.flush_events()
        count += 10


if __name__ == '__main__':
    nn = NeuralNetwork()
    nn.add_layer(Layer(1, 12, 'sigmoid'))
    nn.add_layer(Layer(12, 12, 'sigmoid'))
    nn.add_layer(Layer(12, 1, 'sigmoid'))
    # Define dataset
    que = Queue()
    mses = Queue()
    x = np.linspace(0, 5, 100)
    x = x.reshape(100, 1)
    y = np.cos(x * np.pi)
    new_x = (x - min(x)) / (max(x) - min(x))
    new_y = (y - min(y)) / (max(y) - min(y))
    #que.put(new_y)
    #fig = plt.figure()
    #ln, = plt.plot([], [])
    process = Process(target=nn.train, args=(new_x, new_y, 1, 5000, que, mses))
    process.start()
    #ani = FuncAnimation(fig, update, 1000, fargs=(que, ln), repeat=True)
    myplot(new_x,new_y,que, mses)
    #myplot(new_x,new_y,que,mses)
    #plt.show()
    # Train the neural network
    #errors = nn.train(new_x, new_y, 1, 5000, que, mses)
    process.join()
    z = nn.predict(new_x)
    rez = min(y) + z * (max(y) - min(y))
    rex = min(x) + new_x * (max(x) - min(x))
    plt.plot(rex, rez)
    plt.plot(x, y)
    #plt.show()
