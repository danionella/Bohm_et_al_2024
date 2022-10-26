from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QLineEdit, \
    QVBoxLayout, QHBoxLayout, QLabel, QFormLayout, QSlider, \
        QDoubleSpinBox, QGridLayout, QCheckBox, QSpinBox, QFileDialog, QTabWidget
from PyQt5.QtGui import QDoubleValidator
import numpy as np
from scipy import signal
import pyqtgraph as pg
from PyQt5.QtCore import QThread, Qt, QTimer
from waveforms import Waveforms
from glob import glob
import re
# from time import sleep

class StartWindow(QMainWindow):
    def __init__(self, card):
        super().__init__()
        self.pos1 = 0.0
        self.pos2 = 0.0
        self.daq = card
        self.wf = Waveforms(3.5588, 5.7279, self.daq.sample_freq,
                            0.09126277240684635,
                            2.029975214187555e-06)
        self.generate_layout()
        self.connect_signals()
        
        
      
        self.ls_position.setValue(-0.122*self.wf.ls_scale + self.wf.ls_intercept)
        self.vc_position.setValue(-0.122)
        # self.thread = QThread()
        # self.daq.moveToThread(self.thread)
        # self.thread.started.connect(self.daq.continous_aq)
        # self.daq.progress.connect(self.live_plot)
        # self.daq.done.connect(self.thread.quit)
        self.liveTimer = QTimer()
        self.liveTimer.timeout.connect(self.live_out)
        self.liveTimer.timeout.connect(self.live_plot)
        self.xpeak = 0
        self.isLive = False

    
    def start_button_click(self):
        
        self.awf, self.dwf = self.wf.fast_waveform(self.scan_freq.value(), self.stack_size.value(),
                                         self.vc_position.value(), self.exp_time.value()/1000,
                                         self.duration_input.value(), self.delay.value(),
                                         self.cam_delay.value())
        # try:
        #     self.daq.setup_aquisition(self.awf, self.dwf)
        # # self.daq.test_aquisition()
        
        #     self.daq.start_aquisition()
        # finally:
        #     self.daq.stop_aquisition()
		
		if not self.daq.isOnShutter:
            self.shutter_clicked()
        
        self.daq.fast_aquisition_camera_leader(self.awf,
                                               1/(self.scan_freq.value()*self.exp_time.value()/1000))
        self.shutter_clicked()
		
        self.plotData()
        
    def live_view(self):
        if not self.isLive:
            # self.thread.start()
            self.isLiveStart = True
            self.isLive = True
            self.StartButton.setEnabled(False)
            self.CalibrateBtn.setEnabled(False)
            self.daq.continous_aq_setup()
            self.liveTimer.start(20)
        else:
            # self.thread.requestInterruption()
            self.liveTimer.stop()
            self.daq.continous_aq_cleanup()
            self.isLive = False
            self.StartButton.setEnabled(True)
            self.CalibrateBtn.setEnabled(True)
        
    def plotData(self):
        self.plot.clear()
        plotData1 = self.daq.adata[0,:]
        plotData2 = self.daq.adata[1,:]
        plotData3 = self.daq.ddata.astype(int)
        time = np.linspace(1/self.daq.sample_freq,
                           len(plotData1)/self.daq.sample_freq,
                           len(plotData1))
        self.plot.plot(time, plotData1, pen='r')
        self.plot.plot(time, plotData2, pen='g')
        self.plot.plot(time, self.awf[0,:], pen='y')
        self.plot.plot(time, self.awf[1,:], pen='m')
        self.plot.plot(time, plotData3, pen='b')
        
        if self.SaveDataChk.checkState():
            saveData = np.vstack((time, plotData1, plotData2, plotData3,
                                         self.awf[0,:], self.awf[1,:]))
            saveData = np.transpose(saveData)
            self.save_data(saveData)
        
    def live_plot(self):
        if self.isLiveStart:
            self.plot.clear()
            self.dispData = self.daq.inData
            xDat = np.arange(np.shape(self.dispData)[1])
            self.line1 = self.plot.plot(xDat, self.dispData[0,:], pen='g')
            self.line2 = self.plot.plot(xDat, self.dispData[1,:], pen='r')
            self.isLiveStart = False
        else:
            self.dispData = np.append(self.dispData,self.daq.inData, axis=1)
            if np.shape(self.dispData)[1]>1000:
                self.dispData = self.dispData[:,20:]
            xDat = np.arange(np.shape(self.dispData)[1])
            # print(len(self.dispData))
            self.line1.setData(xDat, self.dispData[0,:])
            self.line2.setData(xDat, self.dispData[1,:])
    
    def live_out(self):
        outDat = np.zeros([2,1])
        outDat[1,:] = self.ls_position.value()
        outDat[0,:] = self.vc_position.value()
        self.daq.continous_aq(outDat)
        
    def combine_pos(self):
        if self.combi_position_slider_label.checkState():
            self.combi_position.setEnabled(True)
            self.combi_position_slider.setEnabled(True)
            self.ls_position.setEnabled(False)
            self.ls_position_slider.setEnabled(False)
            self.vc_position.setEnabled(False)
            self.vc_position_slider.setEnabled(False)
            self.ls_scale.setEnabled(False)
            self.init_combine()
        else:
            self.combi_position.setEnabled(False)
            self.combi_position_slider.setEnabled(False)
            self.ls_position.setEnabled(True)
            self.ls_position_slider.setEnabled(True)
            self.vc_position.setEnabled(True)
            self.vc_position_slider.setEnabled(True)
            self.ls_scale.setEnabled(True)
            
    def init_combine(self):
        vc_uprange = self.vc_position.maximum() - self.vc_position.value()
        vc_downrange = self.vc_position.minimum() - self.vc_position.value()
        ls_uprange = self.ls_position.maximum() - self.ls_position.value()
        ls_downrange = self.ls_position.minimum() - self.ls_position.value()
        
        min_up = np.min([vc_uprange, ls_uprange/self.ls_scale.value()])
        min_down = np.max([vc_downrange, ls_downrange/self.ls_scale.value()])
        
        self.tot_range = min_up - min_down
        self.vc_abs_min = self.vc_position.value() + min_down
        self.ls_abs_min = self.ls_position.value() + min_down*self.ls_scale.value()
        curr_pos = (self.tot_range-min_up) / self.tot_range
        self.combi_position.blockSignals(True)
        self.combi_position.setValue(curr_pos)
        self.combi_position_slider.setValue(curr_pos*1000)
        self.combi_position.blockSignals(False)
        
    def move_combine(self):
        curr_pos = self.combi_position.value()
        new_pos = self.tot_range - (1-curr_pos)*self.tot_range
        self.vc_position.setValue(self.vc_abs_min + new_pos)
        self.ls_position.setValue(self.ls_abs_min + new_pos*self.ls_scale.value())    
            
    def shutter_clicked(self):
        self.daq.shutter()
        if self.daq.isOnShutter:
            self.ShutterButton.setStyleSheet("color: red")
        else:
            self.ShutterButton.setStyleSheet("color: black")
            
    def stack_clicked(self):
        if self.isLive:
            self.live_view()
            
        awf, dwf = self.wf.stack(self.vc_position.value(), 
                            self.stack_size.value(),
                            self.n_steps.value(),
                            self.exp_time.value())
        if not self.daq.isOnShutter:
            self.shutter_clicked()
        self.daq.acquire_stack(awf, dwf)
        self.shutter_clicked()
        
    def psf_stack_clicked(self):
        if self.isLive:
            self.live_view()
            
        awf, dwf = self.wf.psf_stack(self.vc_position.value(), 
                            self.stack_size.value(),
                            self.n_steps.value(),
                            self.exp_time.value())
        if not self.daq.isOnShutter:
            self.shutter_clicked()
        self.daq.acquire_stack(awf, dwf)
        self.shutter_clicked()
        
    def get_pos1_clicked(self):
        self.pos1 = (self.vc_position.value(),
                     self.ls_position.value())
        self.GetPos1Button.setText('Pos1:' + str(self.pos1))
        
    def get_pos2_clicked(self):
        self.pos2 = (self.vc_position.value(),
                     self.ls_position.value())
        self.GetPos2Button.setText('Pos1:' + str(self.pos2))
    
    def calibration_stack_clicked(self):
        if self.isLive:
            self.live_view()
            
        awf, dwf = self.wf.calibration_stack(self.pos1, 
                            self.pos2,
                            self.n_steps.value(),
                            self.exp_time.value())
        if not self.daq.isOnShutter:
            self.shutter_clicked()
        self.daq.acquire_stack(awf, dwf)
        self.shutter_clicked()
        
        self.plot.clear()
 
        time = np.linspace(1/self.daq.sample_freq,
                           len(awf[0,:])/self.daq.sample_freq,
                           len(awf[0,:]))

        self.plot.plot(time, awf[0,:], pen='y')
        self.plot.plot(time, awf[1,:], pen='m')
        
        saveData = np.vstack((time, awf[0,:], awf[1,:]))
        saveData = np.transpose(saveData)
        self.save_data(saveData)
        
    def calibrate_lightsheet_clicked(self):
        self.autofocuswind = AutoFocusWindow()
        self.autofocuswind.show()
        
    def calibrate_waveform(self):
        # xcorr = signal.correlate(self.daq.adata[0,:]-np.mean(self.daq.adata[0,:]),
        #                          self.daq.adata[1,:]-np.mean(self.daq.adata[1,:]))
        # lags = signal.correlation_lags(len(self.daq.adata[0,:]),
        #                                len(self.daq.adata[1,:]))
        # self.xpeak = lags[xcorr.argmax()]
        # self.delay.setValue(self.xpeak)
        self.wf.calibrate_fft(self.awf, self.daq.adata[0:2,:])
        
    def clear_calibration_clicked(self):
        self.wf.kernel_ls = None
        self.wf.kernel_vc = None
        
    def pulse_response_clicked(self):
        self.awf = self.wf.pulses(10, 1, 0.2, self.vc_position.value())
        self.dwf = np.full(self.awf.shape[1], False)
        try:
            self.daq.setup_aquisition(self.awf, self.dwf)
        # self.daq.test_aquisition()
        
            self.daq.start_aquisition()
        finally:
            self.daq.stop_aquisition()
        
        self.plotData()
    
    def select_path_clicked(self):
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.DataPath.setText(file)
        
    def save_data(self, data):
        files = glob(self.DataPath.text() + '\data*.csv')
        if files :
            matches = re.search('(?<=data)[0-9]+', files[-1])
            new_iter = int(matches[0]) + 1
            iter_str = str(new_iter).zfill(4)
        else:
            iter_str = '0000'
        complete_path = self.DataPath.text() + '\data' + iter_str +'.csv' 
        np.savetxt(complete_path, data, delimiter=",")
        
    def start_chirp_clicked(self):
        self.awf, self.dwf = self.wf.chirp_waveform(self.start_freq.value(), self.end_freq.value(),
                                                    self.stack_size.value(),
                                         self.vc_position.value(),
                                         self.duration_input.value())
        try:
            self.daq.setup_aquisition(self.awf, self.dwf)
        # self.daq.test_aquisition()
        
            self.daq.start_aquisition()
        finally:
            self.daq.stop_aquisition()
        
        self.plotData()

    def manual_calibration_values_clicked(self):
        self.wf.ls_scale = self.ls_scale.value()
        self.wf.ls_intercept = self.ls_intercept.value()
        
    def generate_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # buttons
        self.LiveButton = QPushButton('Live', self.central_widget)
        self.StartButton = QPushButton('Start', self.central_widget)
        self.CalibrateBtn = QPushButton('Calibrate', self.central_widget)
        self.ShutterButton = QPushButton('Shutter', self.central_widget)
        self.StackButton = QPushButton('acquire stack', self.central_widget)
        self.PsfStackButton = QPushButton('acquire PSF stack', self.central_widget)
        # self.ulseResponse = QPushButton('get pulse response', self.central_widget)
        self.DataPathButton = QPushButton('path', self.central_widget)
        self.SaveDataChk = QCheckBox('save data', self.central_widget)
        self.ClearCalibrationButton = QPushButton('clear calibration', self.central_widget)
        self.GetPos1Button = QPushButton('Pos1: ' + str(self.pos1), self.central_widget)
        self.GetPos2Button = QPushButton('Pos2: ' + str(self.pos2), self.central_widget)
        self.CalibrationStackButton = QPushButton('acquire calibration stack', self.central_widget)
        self.StartChirpButton = QPushButton('start chirp', self.central_widget)
        self.CalibrateLsButton = QPushButton('Calibrate Lightsheet', self.central_widget)
        self.ManualCalibrationValuesButton = QPushButton('use manual values', self.central_widget)
        
        # sliders
        self.ls_position_slider = QSlider(orientation=Qt.Horizontal, minimum=-10000,
                                          maximum=10000, parent=self.central_widget)
        self.ls_position_slider_label = QLabel('light sheet position')
        
        self.vc_position_slider = QSlider(orientation=Qt.Horizontal, minimum=-3330,
                                          maximum=3330, parent=self.central_widget)
        self.vc_position_slider_label = QLabel('voice coil position')
        
        self.combi_position_slider = QSlider(orientation=Qt.Horizontal, minimum=0,
                                          maximum=1000, parent=self.central_widget)
        self.combi_position_slider_label = QCheckBox('combined position')
        
        # input fields
        self.duration_input = QDoubleSpinBox(self.central_widget)
        self.duration_input_label = QLabel('Duration')
        
        self.scan_freq = QDoubleSpinBox(self.central_widget)
        self.scan_freq_label = QLabel('Scan Frequency')
        
        self.cam_freq = QDoubleSpinBox(self.central_widget)
        self.cam_freq_label = QLabel('Camera Frequency')
        
        self.delay = QSpinBox(self.central_widget, minimum=0, maximum=10000)
        self.delay_label = QLabel('Delay')
        
        self.cam_delay = QDoubleSpinBox(self.central_widget, minimum=0.0, maximum=1.0)
        self.cam_delay_label = QLabel('cam delay (x 2Pi)')
        
        self.stack_size = QDoubleSpinBox(self.central_widget, singleStep=0.001, decimals=3)
        self.stack_size_label = QLabel('Stack size (V)')
        self.stack_size.setValue(0.07)
        
        self.n_steps = QSpinBox(minimum=1, maximum=500, parent=self.central_widget)
        self.n_steps_label = QLabel('n steps')
        self.n_steps.setValue(70)
        
        self.exp_time = QDoubleSpinBox(minimum=0.1, maximum=10000, 
                                       singleStep=0.1, decimals=2, 
                                       parent=self.central_widget)
        self.exp_time_label = QLabel('exposure time')
        self.exp_time.setValue(50.0)
        # self.DelayTxt.setReadOnly(True)
        
        self.ls_position = QDoubleSpinBox(singleStep=0.001, maximum=10,
                                          minimum=-10, decimals=3, 
                                          parent=self.central_widget)
        
        self.vc_position = QDoubleSpinBox(singleStep=0.001, maximum=3.33,
                                          minimum=-3.33, decimals=3, 
                                          parent=self.central_widget)
        
        self.combi_position = QDoubleSpinBox(singleStep=0.001, maximum=1,
                                             minimum=0, decimals=3, 
                                             parent=self.central_widget)
        
        self.ls_scale = QDoubleSpinBox(singleStep=0.001, maximum=100, minimum=0,
                                       decimals=3, parent=self.central_widget)
        self.ls_scale.setValue(self.wf.ls_scale)
        self.ls_scale_label = QLabel('light Sheet scale')
        
        self.ls_intercept = QDoubleSpinBox(singleStep=0.001, decimals=3, parent=self.central_widget)
        self.ls_intercept_label = QLabel('ls intercept')
        
        self.DataPath = QLineEdit('data path')
        
        self.start_freq = QDoubleSpinBox(minimum=1.0, maximum=1000.0, parent=self.central_widget)
        self.start_freq_label = QLabel('start freq')
        self.end_freq = QDoubleSpinBox(minimum=1.0, maximum=1000.0, parent=self.central_widget)
        self.end_freq_label = QLabel('end freq')
        
      

        # plots
        self.plot = pg.PlotWidget()
        
        # layout
        
        self.sub_widget1 = QWidget(self.central_widget)
        self.layout1 = QFormLayout(self.sub_widget1)
        self.layout1.addRow(self.duration_input_label, self.duration_input)
        self.layout1.addRow(self.scan_freq_label, self.scan_freq)
        self.layout1.addRow(self.cam_freq_label, self.cam_freq)
        self.layout1.addRow(self.delay_label, self.delay)
        self.layout1.addRow(self.cam_delay, self.cam_delay_label)
        
        self.sub_widget2 = QWidget(self.central_widget)
        self.layout2 = QHBoxLayout(self.sub_widget2)
        self.layout2.addWidget(self.StartButton)
        self.layout2.addWidget(self.CalibrateBtn)
        self.layout2.addWidget(self.LiveButton)
        self.layout2.addWidget(self.ShutterButton)
        self.layout2.addWidget(self.ClearCalibrationButton)
        
        self.sub_widget3 = QWidget(self.central_widget)
        self.layout3 = QGridLayout(self.sub_widget3)
        self.layout3.addWidget(self.ls_position_slider_label, 1, 1)
        self.layout3.addWidget(self.ls_position_slider, 2, 1)
        self.layout3.addWidget(self.ls_position, 2, 2)
        self.layout3.addWidget(self.vc_position_slider_label, 3, 1)
        self.layout3.addWidget(self.vc_position_slider, 4, 1)
        self.layout3.addWidget(self.vc_position, 4, 2)
        self.layout3.addWidget(self.combi_position_slider_label, 5, 1)
        self.layout3.addWidget(self.combi_position_slider, 6, 1)
        self.layout3.addWidget(self.combi_position, 6, 2)
        
        self.sub_widget4 = QWidget(self.central_widget)
        self.layout4 = QFormLayout(self.sub_widget4)
        self.layout4.addRow(self.ls_scale_label, self.ls_scale)
        self.layout4.addRow(self.ls_intercept_label, self.ls_intercept)
        self.layout4.addRow('',self.ManualCalibrationValuesButton)
        
        self.sub_widget5 = QWidget(self.central_widget)
        self.layout5 = QHBoxLayout(self.sub_widget5)
        self.layout5.addWidget(self.stack_size_label)
        self.layout5.addWidget(self.stack_size)
        self.layout5.addWidget(self.n_steps_label)
        self.layout5.addWidget(self.n_steps)
        self.layout5.addWidget(self.exp_time_label)
        self.layout5.addWidget(self.exp_time)
        self.layout5.addWidget(self.StackButton)
        self.layout5.addWidget(self.PsfStackButton)
        
        self.sub_widget6 = QWidget(self.central_widget)
        self.layout6 = QHBoxLayout(self.sub_widget6)
        self.layout6.addWidget(self.SaveDataChk)
        self.layout6.addWidget(self.DataPath)
        self.layout6.addWidget(self.DataPathButton)
        
        self.sub_widget7 = QWidget(self.central_widget)
        self.layout7 = QHBoxLayout(self.sub_widget7)
        self.layout7.addWidget(self.GetPos1Button)
        self.layout7.addWidget(self.GetPos2Button)
        self.layout7.addWidget(self.CalibrationStackButton)
        self.layout7.addWidget(self.CalibrateLsButton)
        
        self.sub_widget8 = QWidget(self.central_widget)
        self.layout8 = QHBoxLayout(self.sub_widget8)
        self.layout8.addWidget(self.start_freq_label)
        self.layout8.addWidget(self.start_freq)
        self.layout8.addWidget(self.end_freq_label)
        self.layout8.addWidget(self.end_freq)
        self.layout8.addWidget(self.StartChirpButton)
        
        self.tab1 = QWidget(self.central_widget)
        self.layout0 = QVBoxLayout(self.tab1)
        self.layout0.addWidget(self.sub_widget1)
        self.layout0.addWidget(self.plot)
        self.layout0.addWidget(self.sub_widget2)
        self.layout0.addWidget(self.sub_widget3)
        self.layout0.addWidget(self.sub_widget4)
        # self.layout0.addWidget(self.sub_widget5)
        self.layout0.addWidget(self.sub_widget6)
        # self.layout0.addWidget(self.ls_position_slider_label)
        # self.layout0.addWidget(self.ls_position_slider)
        # self.layout0.addWidget(self.ls_position)
        # self.layout0.addWidget(self.vc_position_slider_label)
        # self.layout0.addWidget(self.vc_position_slider)
        # self.layout0.addWidget(self.vc_position)

        self.tab2 = QWidget(self.central_widget)
        self.layoutTab2 = QVBoxLayout(self.tab2)
        self.layoutTab2.addWidget(self.sub_widget5)
        self.layoutTab2.addWidget(self.sub_widget7)
        self.layoutTab2.addWidget(self.sub_widget8)

        # self.tabs = QTabWidget(self.central_widget)
        # self.tabs.addTab(self.tab1, 'main')
        # self.tabs.addTab(self.tab2, 'z-calibration')
        
        self.parentLayout = QHBoxLayout(self.central_widget)
        # self.parentLayout.addWidget(self.tabs)
        self.parentLayout.addWidget(self.tab1)
        self.parentLayout.addWidget(self.tab2)
        
    def connect_signals(self):
      self.StartButton.clicked.connect(self.start_button_click)
      self.CalibrateBtn.clicked.connect(self.calibrate_waveform)
      self.LiveButton.clicked.connect(self.live_view)
      self.ShutterButton.clicked.connect(self.shutter_clicked)
      self.StackButton.clicked.connect(self.stack_clicked)
      self.PsfStackButton.clicked.connect(self.psf_stack_clicked)
      # self.PulseResponse.clicked.connect(self.pulse_response_clicked)
      self.DataPathButton.clicked.connect(self.select_path_clicked)
      self.ClearCalibrationButton.clicked.connect(self.clear_calibration_clicked)
      self.GetPos1Button.clicked.connect(self.get_pos1_clicked)
      self.GetPos2Button.clicked.connect(self.get_pos2_clicked)
      self.CalibrationStackButton.clicked.connect(self.calibration_stack_clicked)
      self.StartChirpButton.clicked.connect(self.start_chirp_clicked)
      self.CalibrateLsButton.clicked.connect(self.calibrate_lightsheet_clicked)
      self.ManualCalibrationValuesButton.clicked.connect(self.manual_calibration_values_clicked)
      
      self.ls_position_slider.valueChanged.connect(lambda x: self.ls_position.setValue(x/1000))
      self.ls_position.valueChanged.connect(lambda x: self.ls_position_slider.setValue(x*1000))
        
      self.vc_position_slider.valueChanged.connect(lambda x: self.vc_position.setValue(x/1000))
      self.vc_position.valueChanged.connect(lambda x: self.vc_position_slider.setValue(int(x*1000)))
      
      self.combi_position_slider.valueChanged.connect(lambda x: self.combi_position.setValue(x/1000))
      self.combi_position.valueChanged.connect(lambda x: self.combi_position_slider.setValue(int(x*1000)))
      self.combi_position.valueChanged.connect(self.move_combine)
      
      self.combi_position_slider_label.stateChanged.connect(self.combine_pos)
      self.combi_position_slider_label.setCheckState(False)
      self.combi_position.setEnabled(False)
      self.combi_position_slider.setEnabled(False)
      
class AutoFocusWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.generate_layout()
        self.connect_signals()
        
    def generate_layout(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.slope = QDoubleSpinBox(self.central_widget)
        self.slope_label = QLabel('Slope')
        self.intercept = QDoubleSpinBox(self.central_widget)
        self.intercept_label = QLabel('Intercept')
        
        self.applyButton = QPushButton('Apply')
        
        self.sub_widget1 = QWidget(self.central_widget)
        self.layout1 = QFormLayout(self.sub_widget1)
        self.layout1.addRow(self.slope_label, self.slope)
        self.layout1.addRow(self.intercept_label, self.intercept)
        self.layout1.addRow('apply', self.applyButton)
    def connect_signals(self):
        pass
        