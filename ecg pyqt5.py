from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import math
from matplotlib.animation import FuncAnimation
from itertools import count
import pyqtgraph as pg
from random import randint
import pandas as pd
import utils as u
import scipy


def filtrar_señal(señal, fs, correct_signal = False, moving_average = False, asd = True):
    if asd ==False:
        return señal
    
    elif correct_signal:
        return u.integrador(u.correct_signal(u.pasabanda(u.med_filt(señal, fs), fs= fs, lowcut= 0.5, highcut= 50),fs),fs, largo_ventana= 0.01)
    else:

        if moving_average:
            return u.integrador(u.pasabanda(u.med_filt(señal, fs), fs= fs, lowcut= 0.5, highcut= 50),fs, largo_ventana= 0.05)
        else:
            return u.pasabanda(u.med_filt(señal, fs), fs= fs, lowcut= 0.5, highcut= 50)
    



class MyWindow(QMainWindow): #Hereda de QMainWindow
    def __init__(self):
        super(MyWindow, self).__init__() #llamo al constructor de la clase padre
        self.fs = 500
        self.setGeometry(300,200,1000,600)
        self.setWindowTitle('Simulador de ECG')

        self.initUI()
               
        
        #chart = Canvas(self)
        
        # READ DATA--------------------------------------------------------------
        
        data = pd.read_csv('coorteeqsrafva.csv', sep=';', header=0, index_col=0)
        data = list(data['diagnosi'])

        signals = np.load('ecgeq-500hzsrfava.npy')

        data_unique = set(data)
        dict_by_rythm = {}
        for rythm in data_unique:
            dict_by_rythm[rythm] = []

        for i in range(len(signals)):
            dict_by_rythm[data[i]].append(np.array(signals[i,:,:]))

        # dict_by_rythm['AFLT'][0][:,0]

        dato = list(data_unique)[1]
        print(data_unique)
        X = dict_by_rythm['SR']

        plt.figure(figsize=(10, 2))
        self.X_filt_real = list(filtrar_señal(X[10][:,1],self.fs,moving_average=True))
        
        secs = int(len(self.X_filt_real)/self.fs)# Number of seconds in signal X
        self.fs = 150
        samps = secs*150     # Number of samples to downsample
        self.X_filt_real = list(scipy.signal.resample(self.X_filt_real, samps))
           
        #y = np.arange(0,len(X_filt_real)/fs,1/fs)
        
        # --------------------------------------------------------------
        
        self.flag = False
        self.plot_graph = pg.PlotWidget()
        self.setCentralWidget(self.plot_graph)
        self.plot_graph
        self.plot_graph.setBackground("w")
        pen = pg.mkPen(color=(255, 0, 0), width=2)
        self.plot_graph.setTitle("ECG", color="b", size="20pt")
        styles = {"color": "red", "font-size": "18px"}
        self.plot_graph.setLabel("left", "Amplitude", **styles)
        self.plot_graph.setLabel("bottom", "S (min)", **styles)
        self.plot_graph.addLegend()
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setXRange(0, 5)
        self.plot_graph.setYRange(-1, 1)
#        self.time = list(np.arange(0, 5, 1/self.fs))
#        self.ECG_SIGNAL = list(np.zeros(5*self.fs))
        self.time = [0]
        self.ECG_SIGNAL = [0]
        # Get a line reference
        self.line = self.plot_graph.plot(
            self.time,
            self.ECG_SIGNAL,
#            name="ECG PLOT",
            pen=pen,
#            symbol="+",
#            symbolSize=15,
#            symbolBrush="b",
        )
        # Add a timer to simulate new temperature measurements
        self.timer = QtCore.QTimer()
        self.timer.setInterval(int(1/self.fs*1000))
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        array_aux = [np.nan]
        if self.time[-1] > 5 and self.ECG_SIGNAL[-1] != 0 and self.flag == False:
            self.flag = True
            self.time = array_aux + self.time
            self.ECG_SIGNAL = array_aux + self.ECG_SIGNAL
            
        if self.flag == True :
            
            self.time.append(self.time[0])
            self.time = self.time[1:]
            self.ECG_SIGNAL = self.ECG_SIGNAL[2:]
            self.ECG_SIGNAL = array_aux + self.ECG_SIGNAL     

        else:
            self.time.append(self.time[-1] + 1/self.fs)
        
        
        self.ECG_SIGNAL.append(self.X_filt_real[0])
        self.line.setData(self.time, self.ECG_SIGNAL)
        self.X_filt_real.append(self.X_filt_real[0])
        self.X_filt_real = self.X_filt_real[1:]
        
    def initUI(self):
        
        #LABELS (TEXTO)
        self.label = QtWidgets.QLabel(self)
        self.label.setText('mi primer label')
        self.label.move(50,50)
        
        #BOTONES
        
        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText('Soy un boton!')
        self.b1.clicked.connect(self.clickear)
        
        #LISTA DESPLEGABLE
        
        self.drop_list = QtWidgets.QComboBox()
        self.drop_list.addItems(['One', 'Two', 'Three', 'Four'])
        
    def clickear(self):
        self.label.setText('apretaste el boton')
        self.update()
    
    def update(self):
        self.label.adjustSize()
        

class Canvas(FigureCanvas):
    def __init__(self, parent):
        fig ,self.ax = plt.subplots(figsize = (5,4), dpi = 200)
        
        '''
        super().__init__(fig)
        self.setParent(parent)
        #plt.style.use('fiverthirtyeight')
        #ani = FuncAnimation(plt.gcf(), animate, interval = 1000)
        
        plt.ylim(-2,2)
        plt.xlim(0,100)
        x = list()
        y = list()

        i = 0

        inicio = 0
        

        while i < 100:
            t = list(np.arange(inicio, inicio + 0.5, 0.05))
            y = y + t
            for time in t:
                x.append(math.sin(time))
            plt.cla()
            self.ax.plot(y, x,color="blue")
            plt.pause(0.05)
            i += 1
            inicio = inicio + 0.5

        plt.close()
        
        index = count()
        
        def animate(i):
            x.append((next(index)))
            y.append()
        
        '''

def window():
    
    #INICIALIZAR VENTANA Y SUS PARAMETROS
    app = QApplication(sys.argv)
    win = MyWindow()
    #CODIGO PARA MOSTRAR VENTANA
    win.show()
    #CODIGO PARA SALIR
    sys.exit(app.exec_())
    
window()
