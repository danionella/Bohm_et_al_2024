from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QLineEdit, \
    QVBoxLayout, QHBoxLayout, QLabel, QFormLayout, QSlider, \
        QDoubleSpinBox, QGridLayout, QCheckBox, QSpinBox, QFileDialog, QTabWidget
from PyQt5.QtGui import QDoubleValidator
import numpy as np
from scipy import signal
import pyqtgraph as pg
from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
from waveforms import Waveforms
from autofocus import Autofocus
from glob import glob
import re
from QLed import QLed

# from time import sleep

class StartWindow(QMainWindow):
    def __init__(self, card):
        """
         Initialize the widget. 
         
         @param card - The NI card to use
        """
        super().__init__()
        self.pos1 = 0.0
        self.pos2 = 0.0
        self.daq = card
        # initialize waveform class
        self.wf = Waveforms(3.5588, 5.7279, self.daq.sample_freq,
                            0.09126277240684635,
                            2.029975214187555e-06)
        # generate the layout and connect signals
        self.generate_layout()
        self.connect_signals()
        self.CalibrateLed.value = False
        
        
      
        self.ls_position.setValue(-0.122*self.wf.ls_scale + self.wf.ls_intercept)
        self.vc_position.setValue(-0.122)
        
        # self.thread = QThread()
        # self.daq.moveToThread(self.thread)
        # self.thread.started.connect(self.daq.continous_aq)
        # self.daq.progress.connect(self.live_plot)
        # self.daq.done.connect(self.thread.quit)

        #setup interrupt for live view
        self.liveTimer = QTimer()
        self.liveTimer.timeout.connect(self.live_out)
        self.liveTimer.timeout.connect(self.live_plot)
        self.xpeak = 0
        self.isLive = False

    
    def start_button_click(self):
        """start fast data acquisition
        """        
        # generate waveforms for fast acquisition
        self.awf, self.dwf = self.wf.fast_waveform(self.scan_freq.value(), self.stack_size.value(),
                                         self.vc_position.value(), self.exp_time.value()/1000,
                                         self.duration_input.value(), self.delay.value(),
                                         self.cam_delay.value(), self.stBox.checkState(),
                                         self.stim_position.value(), self.stim_duration.value(),
                                         self.stim_frequency.value(), self.stim_amplitude.value())
        # try:
        #     self.daq.setup_aquisition(self.awf, self.dwf)
        # # self.daq.test_aquisition()
        
        #     self.daq.start_aquisition()
        # finally:
        #     self.daq.stop_aquisition()
        
        # self.daq.fast_aquisition_camera_leader(self.awf,
        #                                        1/(self.scan_freq.value()*self.exp_time.value()/1000))

        # acquire the data
        self.daq.fast_aquisition_camera_follower(self.awf, self.dwf)
        # plot the data
        self.plotData()
        
    def live_view(self):
        """start live view
        """        
        if not self.isLive:
            # self.thread.start()
            self.isLiveStart = True
            self.isLive = True
            self.StartButton.setEnabled(False)
            self.CalibrateBtn.setEnabled(False)
            self.daq.continous_aq_setup()
            # start the interrupt
            self.liveTimer.start(20)
        else:
            # self.thread.requestInterruption()
            self.liveTimer.stop()
            self.daq.continous_aq_cleanup()
            self.isLive = False
            self.StartButton.setEnabled(True)
            self.CalibrateBtn.setEnabled(True)
        
    def plotData(self):
        """
         plot acquired data in main window
        """
        self.plot.clear()
        plotData1 = self.daq.adata[0,:]
        plotData2 = self.daq.adata[1,:]
        plotData3 = self.daq.ddata.astype(int)
        # setup xaxis for analog and digital data
        time = np.linspace(1/self.daq.sample_freq,
                           len(plotData1)/self.daq.sample_freq,
                           len(plotData1))
        dtime = np.linspace(1/1e6,
                           len(plotData3)/1e6,
                           len(plotData3))
        self.plot.plot(time, plotData1, pen='r')
        self.plot.plot(time, plotData2, pen='g')
        self.plot.plot(time, self.awf[0,:], pen='y')
        self.plot.plot(time, self.awf[1,:], pen='m')
        self.plot.plot(time, self.awf[2,:], pen='c')
        self.plot.plot(dtime, plotData3, pen='b')
        
        # plot2Data1 = (plotData1 - self.wf.vc_lin_intercept)/self.wf.vc_lin_slope
        # plot2Data2 = (plotData2 - self.wf.ls_lin_intercept)/self.wf.ls_lin_slope
        # plot2Data2 = (plot2Data2 - self.wf.ls_intercept)/self.wf.ls_scale
        plot2Data1 = plotData1
        plot2Data2 = (plotData2 - self.wf.ls_out_intercept)/self.wf.ls_out_scale
        
        self.plot2.clear()
        self.plot2.plot(time, plot2Data1, pen='r')
        self.plot2.plot(time, plot2Data2, pen='g')
        
        # if desired, save data
        if self.SaveDataChk.checkState():
            saveData = np.vstack((time, plotData1, plotData2,
                                         self.awf[0,:], self.awf[1,:]))
            saveData = np.transpose(saveData)
            dsaveData = np.transpose(np.vstack((dtime, plotData3)))
            self.save_data(analogData=saveData, digitalData=dsaveData)
        
    def live_plot(self):
        """this function is called to plot data during live view
        """        
        if self.isLiveStart:
            # clear canvas when live view is started
            self.plot.clear()
            self.dispData = self.daq.inData
            xDat = np.arange(np.shape(self.dispData)[1])
            self.line1 = self.plot.plot(xDat, self.dispData[0,:], pen='g')
            self.line2 = self.plot.plot(xDat, self.dispData[1,:], pen='r')
            self.isLiveStart = False
        else:
            # append new data
            self.dispData = np.append(self.dispData,self.daq.inData, axis=1)
            # distData behaves like a FIFO buffer after reaching 1000 samples
            if np.shape(self.dispData)[1]>1000:
                self.dispData = self.dispData[:,20:]
            xDat = np.arange(np.shape(self.dispData)[1])
            # print(len(self.dispData))

            #display the data
            self.line1.setData(xDat, self.dispData[0,:])
            self.line2.setData(xDat, self.dispData[1,:])
    
    def live_out(self):
        """this gets called whenever the ls or vc position values change. Used to set new galvo or voice coil position
        """        
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
        """
         Called when shutter button is clicked
        """

        self.daq.shutter()
        # Set button color to red or black
        if self.daq.isOnShutter:
            self.ShutterButton.setStyleSheet("color: red")
        else:
            self.ShutterButton.setStyleSheet("color: black")
            
    def stack_clicked(self):
        """
         Called when the acquire stack button is clicked
        """
        if self.isLive:
            self.live_view()

        # generate waveforms for stack acquisition    
        awf, dwf = self.wf.stack(self.vc_position.value(), 
                            self.stack_size.value(),
                            self.n_steps.value(),
                            self.exp_time.value())
        # open shutter
        if not self.daq.isOnShutter:
            self.shutter_clicked()
        # acquire stack
        self.daq.acquire_stack(awf, dwf)
        # close shutter
        self.shutter_clicked()
        
    def psf_stack_clicked(self):
        """
         Called when the acquire psf stack button is clicked

        """
        if self.isLive:
            self.live_view()
        # calculate waveform    
        awf, dwf = self.wf.psf_stack(self.vc_position.value(), 
                            self.stack_size.value(),
                            self.n_steps.value(),
                            self.exp_time.value(),
                            self.step_size.value())
        # open shutter
        if not self.daq.isOnShutter:
            self.shutter_clicked()
        # acquire data
        self.daq.acquire_stack(awf, dwf)
        # close shutter
        self.shutter_clicked()
        
    def get_pos1_clicked(self):
        """called when pos1 is clicked, saves the current position of galvo and voice coil
        """        
        self.pos1 = (self.vc_position.value(),
                     self.ls_position.value())
        self.GetPos1Button.setText('Pos1:' + str(self.pos1))
        
    def get_pos2_clicked(self):
        """called when pos2 is clicked, saves the current position of galvo and voice coil
        """
        self.pos2 = (self.vc_position.value(),
                     self.ls_position.value())
        self.GetPos2Button.setText('Pos2:' + str(self.pos2))
    
    def calibration_stack_clicked(self):
        """called when calibration stack button is clicked, triggers camera and outputs waveform for galvo and voice coil for calibration stack
        """        
        if self.isLive:
            self.live_view()
        # calculate calibration waveform, pos1 and pos2 are the top and bottom positions    
        awf, dwf = self.wf.calibration_stack(self.pos1, 
                            self.pos2,
                            self.n_steps.value(),
                            self.exp_time.value(),
                            self.step_size.value())
        # open shutter
        if not self.daq.isOnShutter:
            self.shutter_clicked()
        # acquire data
        data = self.daq.acquire_stack(awf, dwf)
        # close shutter
        self.shutter_clicked()
        # plot data
        self.plot.clear()
 
        time = np.linspace(1/self.daq.sample_freq,
                           len(awf[0,:])/self.daq.sample_freq,
                           len(awf[0,:]))

        self.plot.plot(time, awf[0,:], pen='y')
        self.plot.plot(time, awf[1,:], pen='m')
        # save data
        saveData = np.vstack((time, awf[0,:], awf[1,:], data))
        saveData = np.transpose(saveData)
        self.save_data(data=saveData)
        
    def calibrate_lightsheet_clicked(self):
        """
         opens new window to load and calculate light sheet focus calibration
        """
        self.autofocuswind = AutoFocusWindow()
        self.autofocuswind.applyClicked.connect(self.on_calibrate_lightsheet_apply)
        self.autofocuswind.show()
        
    def on_calibrate_lightsheet_apply(self, values):
        """
         Called when user clicks on the apply button in the autofocus window
         
         @param values - list of calibration values
        """
        # set values
        self.manual_calibration_values_clicked(values[0], values[1])
        self.wf.ls_out_intercept = values[3]
        self.wf.ls_out_scale = values[2]
        
    def calibrate_waveform(self):
        """
         calibrate galvo and light sheet, used after one fast acquisition
        """
        # xcorr = signal.correlate(self.daq.adata[0,:]-np.mean(self.daq.adata[0,:]),
        #                          self.daq.adata[1,:]-np.mean(self.daq.adata[1,:]))
        # lags = signal.correlation_lags(len(self.daq.adata[0,:]),
        #                                len(self.daq.adata[1,:]))
        # self.xpeak = lags[xcorr.argmax()]
        # self.delay.setValue(self.xpeak)
        linawf = self.wf.ramp()
        lindata = self.daq.acquire_linear_calibration(linawf)
        self.wf.calibrate_fft(self.awf, self.daq.adata[0:2,:], linawf, lindata)
        if self.wf.kernel_vc is not None and self.wf.kernel_ls is not None:
            self.CalibrateLed.value = True
        else:
            self.CalibrateLed.value = False
            
        
    def clear_calibration_clicked(self):
        """clear fast acquisition calibration
        """
        self.wf.kernel_ls = None
        self.wf.kernel_vc = None
        self.CalibrateLed.value = False
        
    def beat_calibration_clicked(self):
        self.awf, self.dwf = self.wf.beat_waveform(self.scan_freq.value(), self.stack_size.value(),
                                         self.vc_position.value(), self.exp_time.value()/1000,
                                         self.duration_input.value(), self.delay.value(),
                                         self.cam_delay.value(), self.stBox.checkState(),
                                         self.stim_position.value(), self.stim_duration.value(),
                                         self.stim_frequency.value(), self.stim_amplitude.value())
        self.daq.fast_aquisition_camera_follower(self.awf, self.dwf)
        self.plotData()
        
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
        """called when select path is clicked, set directory to save data
        """
        file = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.DataPath.setText(file)
        
    def save_data(self, **data):
        """save data as *.npz to selected folder
        """
        files = glob(self.DataPath.text() + '\data*.npz')
        # increment data name by +1
        if files :
            matches = re.search('(?<=data)[0-9]+', files[-1])
            new_iter = int(matches[0]) + 1
            iter_str = str(new_iter).zfill(4)
        else:
            iter_str = '0000'
        complete_path = self.DataPath.text() + '\data' + iter_str +'.npz' 
        # np.savetxt(complete_path, data, delimiter=",")
        np.savez(complete_path, **data)
        
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
        
    def debug_button_clicked(self):
        """called when debug button is clicked, placeholder function to put debug code
        """
        awf = self.wf.multi_freq_sawtooth(1, 200, 5, 5, 0.4, -0.7)
        awf = np.vstack((awf, np.zeros(awf.shape)))
        
        lindata = self.daq.acquire_linear_calibration(awf)
        
        time = np.linspace(1/self.daq.sample_freq,
                           len(awf[0,:])/self.daq.sample_freq,
                           len(awf[0,:]))

        self.plot.plot(time, awf[0,:], pen='y')
        self.plot.plot(time, awf[1,:], pen='m')
        
        saveData = np.vstack((time, lindata[0,:], lindata[1,:], awf[0,:], awf[1,:]))
        saveData = np.transpose(saveData)
        self.save_data(data=saveData)
        
    def manual_calibration_values_clicked(self, ls_scale, ls_intercept):
        """called when manual calibration is clicked to use manual z-calibration values

        Args:
            ls_scale (float): slace of galvo / vc calibration
            ls_intercept (float): intercept of galvo / vc calibration
        """
        self.wf.ls_scale = ls_scale
        self.ls_scale.setValue(ls_scale)
        self.wf.ls_intercept = ls_intercept
        self.ls_intercept.setValue(ls_intercept)
        
    def generate_layout(self):
        """hardcoded layout with all controlls in the main window
        """        
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
        self.DebugButton = QPushButton('Debug', self.central_widget)
        self.BeatCalibrationButton = QPushButton('Beat Calibration', self.central_widget)
        
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
        self.duration_input.setValue(1.0)
        
        self.scan_freq = QDoubleSpinBox(self.central_widget)
        self.scan_freq_label = QLabel('Scan Frequency')
        self.scan_freq.setValue(8)
        
        self.cam_freq = QDoubleSpinBox(self.central_widget)
        self.cam_freq_label = QLabel('Camera Frequency')
        
        self.delay = QSpinBox(self.central_widget, minimum=0, maximum=10000)
        self.delay_label = QLabel('Delay')
        
        self.stim_label = QLabel('Stimulus:')
        self.stim_duration = QDoubleSpinBox(self.central_widget, minimum=0, singleStep=1)
        self.stim_duration_label = QLabel('duration (ms):')
        self.stim_position = QDoubleSpinBox(self.central_widget, minimum=0, singleStep=0.001, decimals=3)
        self.stim_position_label = QLabel('position (s):')
        self.stim_frequency = QDoubleSpinBox(self.central_widget, minimum=0, maximum=self.daq.sample_freq/2)
        self.stim_frequency_label = QLabel('frequency (Hz):') 
        self.stim_amplitude = QDoubleSpinBox(self.central_widget, minimum=0, singleStep=0.001, decimals=3)
        self.stim_amplitude_label = QLabel('amplitude (V):')
        
        self.cam_delay = QSpinBox(self.central_widget, minimum=0, maximum=1e6)
        self.cam_delay_label = QLabel('cam delay (in us)')
        
        self.stack_size = QDoubleSpinBox(self.central_widget, singleStep=0.001, decimals=3)
        self.stack_size_label = QLabel('Stack size (V)')
        self.stack_size.setValue(0.07)
        
        self.n_steps = QSpinBox(minimum=1, maximum=2000, parent=self.central_widget)
        self.n_steps_label = QLabel('n steps')
        self.n_steps.setValue(70)
        
        self.exp_time = QDoubleSpinBox(minimum=0.001, maximum=10000, 
                                       singleStep=0.001, decimals=3, 
                                       parent=self.central_widget)
        self.exp_time_label = QLabel('exposure time')
        self.exp_time.setValue(0.25)
        # self.DelayTxt.setReadOnly(True)
        
        self.step_size = QDoubleSpinBox(minimum=0, maximum=100, singleStep=0.001, decimals=4,
                                        parent=self.central_widget)
        self.step_size_label = QLabel('step size (V)')
        
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
        
        self.stBox = QCheckBox("sawtooth",self)
        
      

        # plots
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot2 = self.glw.addPlot(row=1, col=0)
        self.plot.setXLink(self.plot2)
        # self.plot = pg.PlotWidget()
        
        # indicators
        self.CalibrateLed=QLed(self, onColour=QLed.Green, shape=QLed.Circle)
        
        # layout
        
        self.sub_widget1 = QWidget(self.central_widget)
        self.layout1 = QFormLayout(self.sub_widget1)
        self.layout1.addRow(self.duration_input_label, self.duration_input)
        self.layout1.addRow(self.scan_freq_label, self.scan_freq)
        self.layout1.addRow(self.stBox)
        self.layout1.addRow(self.cam_freq_label, self.cam_freq)
        self.layout1.addRow(self.delay_label, self.delay)
        self.layout1.addRow(self.cam_delay, self.cam_delay_label)
        
        self.sub_widget1b = QWidget(self.central_widget)
        self.layout1b = QHBoxLayout(self.sub_widget1b)
        self.layout1b.addWidget(self.stim_position_label)
        self.layout1b.addWidget(self.stim_position)
        self.layout1b.addWidget(self.stim_duration_label)
        self.layout1b.addWidget(self.stim_duration)
        self.layout1b.addWidget(self.stim_frequency_label)
        self.layout1b.addWidget(self.stim_frequency)
        self.layout1b.addWidget(self.stim_amplitude_label)
        self.layout1b.addWidget(self.stim_amplitude)
        
        self.sub_widget2 = QWidget(self.central_widget)
        self.layout2 = QHBoxLayout(self.sub_widget2)
        self.layout2.addWidget(self.StartButton)
        self.layout2.addWidget(self.CalibrateBtn)
        self.layout2.addWidget(self.CalibrateLed)
        self.layout2.addWidget(self.LiveButton)
        self.layout2.addWidget(self.ShutterButton)
        self.layout2.addWidget(self.ClearCalibrationButton)
        self.layout2.addWidget(self.BeatCalibrationButton)
        
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
        self.layout5.addWidget(self.step_size)
        self.layout5.addWidget(self.step_size_label)
        
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
        self.layout8.addWidget(self.DebugButton)
        
        self.tab1 = QWidget(self.central_widget)
        self.layout0 = QVBoxLayout(self.tab1)
        self.layout0.addWidget(self.sub_widget1)
        self.layout0.addWidget(self.stim_label)
        self.layout0.addWidget(self.sub_widget1b)
        self.layout0.addWidget(self.glw)
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
        """connect all the controlls to their respective functions
        """      
        self.StartButton.clicked.connect(self.start_button_click)
        self.CalibrateBtn.clicked.connect(self.calibrate_waveform)
        self.LiveButton.clicked.connect(self.live_view)
        self.ShutterButton.clicked.connect(self.shutter_clicked)
        self.StackButton.clicked.connect(self.stack_clicked)
        self.PsfStackButton.clicked.connect(self.psf_stack_clicked)
        # self.PulseResponse.clicked.connect(self.pulse_response_clicked)
        self.DataPathButton.clicked.connect(self.select_path_clicked)
        self.ClearCalibrationButton.clicked.connect(self.clear_calibration_clicked)
        self.BeatCalibrationButton.clicked.connect(self.beat_calibration_clicked)
        self.GetPos1Button.clicked.connect(self.get_pos1_clicked)
        self.GetPos2Button.clicked.connect(self.get_pos2_clicked)
        self.CalibrationStackButton.clicked.connect(self.calibration_stack_clicked)
        self.StartChirpButton.clicked.connect(self.start_chirp_clicked)
        self.DebugButton.clicked.connect(self.debug_button_clicked)
        self.CalibrateLsButton.clicked.connect(self.calibrate_lightsheet_clicked)
        self.ManualCalibrationValuesButton.clicked.connect(lambda: \
                                                            self.manual_calibration_values_clicked(self.ls_scale.value(), self.ls_intercept.value()))
        
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
      
      
class AutoFocusWindow(QWidget):
    """autofocus window
    """

    applyClicked = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.generate_layout()
        self.connect_signals()
        self.af = Autofocus()
        
    def generate_layout(self):
        # self.central_widget = QWidget()
        # self.setCentralWidget(self.central_widget)
        
        # self.slope = QDoubleSpinBox(self.central_widget)
        self.slope = QDoubleSpinBox()
        self.slope_label = QLabel('Slope')
        # self.intercept = QDoubleSpinBox(self.central_widget)
        self.intercept = QDoubleSpinBox()
        self.intercept_label = QLabel('Intercept')
        
        self.applyButton = QPushButton('Apply')
        self.autofocusButton = QPushButton('run autofocus')
        self.imagePathButton = QPushButton('image path')
        self.csvPathButton = QPushButton('csv path')
        self.imagePath = QLineEdit('image path')
        self.csvPath = QLineEdit('csv path')
        
        # self.sub_widget1 = QWidget(self.central_widget)
        self.sub_widget1 = QWidget()
        self.layout1 = QFormLayout(self.sub_widget1)
        self.layout1.addRow(self.slope_label, self.slope)
        self.layout1.addRow(self.intercept_label, self.intercept)
        self.layout1.addRow('apply', self.applyButton)
        
        self.glw = pg.GraphicsLayoutWidget()
        self.plt = self.glw.addPlot()
        
        self.sub_widget2 = QWidget()
        self.layout2 = QHBoxLayout(self.sub_widget2)
        self.layout2.addWidget(self.imagePath)
        self.layout2.addWidget(self.imagePathButton)
        
        self.sub_widget3 = QWidget()
        self.layout3 = QHBoxLayout(self.sub_widget3)
        self.layout3.addWidget(self.csvPath)
        self.layout3.addWidget(self.csvPathButton)
        
        self.layout0 = QVBoxLayout(self)
        self.layout0.addWidget(self.sub_widget1)
        self.layout0.addWidget(self.sub_widget2)
        self.layout0.addWidget(self.sub_widget3)
        self.layout0.addWidget(self.autofocusButton)
        self.layout0.addWidget(self.glw)
        
        # self.setLayout(self.layout0)
    def connect_signals(self):
        self.applyButton.clicked.connect(self.apply)
        self.imagePathButton.clicked.connect(self.select_path_clicked)
        self.csvPathButton.clicked.connect(self.select_path_clicked)
        self.autofocusButton.clicked.connect(self.run_autofocus)
        
    def apply(self):
        """called when the apply button is clicked, sends list with values to main window
        """
        self.applyClicked.emit([self.slope.value(), self.intercept.value(),
                                self.af.measured_slope_out, self.af.measured_intercept_out])
        
    def select_path_clicked(self):
        snd = self.sender()
        nm = snd.text()
        file = str(QFileDialog.getOpenFileName(self, "Select Directory")[0])
        if 'image' in nm:
            self.imagePath.setText(file)
        elif 'csv' in nm:
            self.csvPath.setText(file)
        
    def run_autofocus(self):
        """run the autofocus the get z-calibration
        """
        # load data
        self.af.load_data(self.imagePath.text(), self.csvPath.text())
        # run autofocus
        self.af.run_af()
        # display results
        self.slope.setValue(self.af.measured_slope)
        self.intercept.setValue(self.af.measured_intercept)
        self.plt.clear()
        self.plt.plot(self.af.vc_values, self.af.ls_values, pen=None, symbol='o')
        self.plt.plot(self.af.vc_values, self.af.vc_values*self.af.measured_slope + self.af.measured_intercept)
        
if __name__=="__main__":
    pass