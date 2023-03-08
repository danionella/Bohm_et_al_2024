import nidaqmx
from nidaqmx.constants import Edge, AcquisitionType
from nidaqmx.stream_readers import AnalogMultiChannelReader
from nidaqmx.stream_writers import AnalogMultiChannelWriter
import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject
from time import sleep

class Daq(QObject):
    
    progress = pyqtSignal(object)
    done = pyqtSignal()
    
    def __init__(self, daq_name, sample_freq):
        super().__init__()
        self.daq_name = daq_name
        self.sample_freq = sample_freq
        self.isOnShutter = False
    
    def setup_aquisition(self, awaveform, dwaveform):
        
        self.aintask = nidaqmx.Task()
        self.aouttask = nidaqmx.Task()
        self.dintask = nidaqmx.Task()
        self.douttask = nidaqmx.Task()
        self.n_samples = np.shape(awaveform)[1]
        self.duration = self.n_samples / self.sample_freq
        self.adata = np.zeros((self.n_samples,2))
        self.ddata = np.full((self.n_samples,1), False)
        self.aintask.ai_channels.add_ai_voltage_chan(self.daq_name + "ai0:1")
        self.aouttask.ao_channels.add_ao_voltage_chan(self.daq_name + "ao0:2", max_val=10, min_val=-10)
        self.dintask.di_channels.add_di_chan(self.daq_name + "port0/line1")
        self.douttask.do_channels.add_do_chan(self.daq_name + 'port0/line0')
        self.aintask.timing.cfg_samp_clk_timing(self.sample_freq, source="", active_edge=Edge.RISING,
                                               sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=self.n_samples)
        self.aouttask.timing.cfg_samp_clk_timing(
            self.sample_freq,
            source="",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self.n_samples)
        
        self.dintask.timing.cfg_samp_clk_timing(self.sample_freq, source="",
                                                active_edge=Edge.RISING,
                                                sample_mode=AcquisitionType.FINITE,
                                                samps_per_chan=self.n_samples)
        
        self.douttask.timing.cfg_samp_clk_timing(
            self.sample_freq,
            source="",
            active_edge=Edge.RISING,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=self.n_samples)
       
        self.aouttask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/ai/StartTrigger')
        self.dintask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/ai/StartTrigger')
        self.douttask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/ai/StartTrigger')
      
        

        self.aouttask.write(awaveform)
        self.douttask.write(dwaveform)
        
    def fast_aquisition_camera_follower(self, awaveform, dwaveform):
        self.n_samples = np.shape(awaveform)[1]
        self.duration = self.n_samples / self.sample_freq
        with nidaqmx.Task() as aouttask, nidaqmx.Task() as aintask, \
        nidaqmx.Task() as dintask, nidaqmx.Task() as douttask:
            
            aintask.ai_channels.add_ai_voltage_chan(self.daq_name + "ai0:1")
            aouttask.ao_channels.add_ao_voltage_chan(self.daq_name + "ao0:2", max_val=10, min_val=-10)
            dintask.di_channels.add_di_chan(self.daq_name + "port0/line0")
            douttask.do_channels.add_do_chan(self.daq_name + 'port0/line1')
            douttask.do_channels.add_do_chan(self.daq_name + 'port0/line7')
            
            alltasks = [aintask, aouttask, douttask, dintask]  
            for task in alltasks:
                task.timing.cfg_samp_clk_timing(self.sample_freq,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=self.n_samples)
           
            for task in alltasks[1:]:
                task.triggers.start_trigger.cfg_dig_edge_start_trig(
                    trigger_source='/Dev1/ai/StartTrigger')
            
            aouttask.write(awaveform)
            douttask.write(dwaveform)
            
            aouttask.start()
            douttask.start()
            dintask.start()
            aintask.start()
            self.adata = np.asarray(aintask.read(self.n_samples, 
                                                    timeout=self.duration + 10))
            self.ddata = np.asarray(dintask.read(self.n_samples,
                                                      timeout=self.duration + 10))
        
    def fast_aquisition_camera_leader(self, awaveform, wf_freq):
        self.n_samples = np.shape(awaveform)[1]
        self.duration = self.n_samples / self.sample_freq
        samplesPerTrigger = self.sample_freq/wf_freq
        with nidaqmx.Task() as outtask, nidaqmx.Task() as aintask, \
        nidaqmx.Task() as dintask, nidaqmx.Task() as countertask, nidaqmx.Task() as shutterTask:

            countertask.co_channels.add_co_pulse_chan_freq("Dev1/ctr0", freq=self.sample_freq)
            countertask.timing.cfg_implicit_timing(sample_mode=AcquisitionType.FINITE, samps_per_chan=int(samplesPerTrigger))
            countertask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev1/PFI1")
            countertask.triggers.start_trigger.retriggerable = True
                
                
            outtask.ao_channels.add_ao_voltage_chan(self.daq_name + "ao0:1", max_val=10, min_val=-10)
            outtask.timing.cfg_samp_clk_timing(
                self.sample_freq,
                source="/Dev1/Ctr0InternalOutput",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=self.n_samples)
            outtask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev1/PFI1")
            
            shutterTask.do_channels.add_do_chan(self.daq_name + 'port0/line7')
            shutterTask.timing.cfg_samp_clk_timing(self.sample_freq, source="/Dev1/Ctr0InternalOutput",
                                              active_edge=Edge.RISING,
                                               sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=self.n_samples)
            
            shutterTask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev1/PFI1")
            
            aintask.ai_channels.add_ai_voltage_chan(self.daq_name + "ai0:1")
            aintask.timing.cfg_samp_clk_timing(self.sample_freq, source="/Dev1/Ctr0InternalOutput",
                                              active_edge=Edge.RISING,
                                               sample_mode=AcquisitionType.FINITE,
                                               samps_per_chan=self.n_samples)
            aintask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev1/PFI1")

            
            dintask.di_channels.add_di_chan(self.daq_name + "port0/line1")
            dintask.timing.cfg_samp_clk_timing(self.sample_freq, source="/Dev1/Ctr0InternalOutput",
                                                    active_edge=Edge.RISING,
                                                    sample_mode=AcquisitionType.FINITE,
                                                    samps_per_chan=self.n_samples)
            dintask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source="/Dev1/PFI1")
            
            shutterwave = np.full(self.n_samples, True)
            shutterwave[-1] = False
            outtask.write(awaveform)
            shutterTask.write(shutterwave)
            outtask.start()
            shutterTask.start()
            aintask.start()
            dintask.start()
            countertask.start()
            self.adata = np.asarray(aintask.read(self.n_samples, 
                                                    timeout=self.duration + 10))
            self.ddata = np.asarray(dintask.read(self.n_samples,
                                                      timeout=self.duration + 10))
        
            
    def start_aquisition(self):
        # self.outtask.start()
        self.aouttask.start()
        self.douttask.start()
        self.dintask.start()
        self.aintask.start()
        
        
        self.adata = np.asarray(self.aintask.read(self.n_samples, 
                                                timeout=self.duration + 2))
        self.ddata = np.asarray(self.dintask.read(self.n_samples,
                                                  timeout=self.duration + 2))
        self.douttask.wait_until_done()
        
        
    
    def stop_aquisition(self):
        self.aintask.stop()
        self.aouttask.stop()
        self.dintask.stop()
        self.douttask.stop()
        self.aintask.close()
        self.aouttask.close()
        self.dintask.close()
        self.douttask.close()
        
    # def test_aquisition(self):
    #     test_dat = np.zeros(1,)
    #     for n in range(int(self.duration)):
    #         test_dat[0] = np.random.rand()
    #         self.progress.emit(test_dat)
    #         sleep(1)
    
    def continous_aq_setup(self):
        self.liveIntask = nidaqmx.Task()
        self.liveOuttask = nidaqmx.Task()
        self.liveIntask.ai_channels.add_ai_voltage_chan(self.daq_name + "ai0:1")
        self.liveOuttask.ao_channels.add_ao_voltage_chan(self.daq_name + "ao0:1",
                                                    max_val=10, min_val=-10)
        self.liveIntask.timing.cfg_samp_clk_timing(400, source="", active_edge=Edge.RISING,
                                               sample_mode=AcquisitionType.CONTINUOUS,
                                               samps_per_chan=20)
        # self.liveOuttask.timing.cfg_samp_clk_timing(200,
        #                                        sample_mode=AcquisitionType.CONTINUOUS,
        #                                        samps_per_chan=2000)
        self.reader = AnalogMultiChannelReader(self.liveIntask.in_stream)
        self.writer = AnalogMultiChannelWriter(self.liveOuttask.out_stream)
        self.liveIntask.start()
        self.liveOuttask.start()
        self.inData = np.zeros([2, 20])
        
    def continous_aq(self, writeData):
        
        # while not self.thread().isInterruptionRequested():
        # try:
            self.reader.read_many_sample(data = self.inData,
                                    number_of_samples_per_channel = 20)
            self.writer.write_many_sample(writeData)
        # finally:
            # print('no data acquired')
            
    
    def continous_aq_cleanup(self):
        self.liveIntask.stop()
        self.liveOuttask.stop
        self.liveIntask.close()
        self.liveOuttask.close()
        
    def shutter(self):
        with nidaqmx.Task() as shutterTask:
            shutterTask.do_channels.add_do_chan(self.daq_name + 'port0/line8')
            if not self.isOnShutter:
                shutterTask.write(True)
                shutterTask.start()
                self.isOnShutter = True
            else:
                shutterTask.write(False)
                shutterTask.start()
                self.isOnShutter = False
                
    def acquire_stack(self, awf, dwf):
        with nidaqmx.Task() as aouttask, nidaqmx.Task() as douttask, nidaqmx.Task() as aintask:
            aouttask.ao_channels.add_ao_voltage_chan(self.daq_name + "ao0:1", max_val=10, min_val=-10)
            douttask.do_channels.add_do_chan(self.daq_name + 'port0/line1')
            aintask.ai_channels.add_ai_voltage_chan(self.daq_name + "ai0:1")
            
            aouttask.timing.cfg_samp_clk_timing(
                self.sample_freq,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=awf.shape[1])
            
            douttask.timing.cfg_samp_clk_timing(
                self.sample_freq,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=dwf.shape[0])
            
            aintask.timing.cfg_samp_clk_timing(
                self.sample_freq,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=awf.shape[1])
            
            
            douttask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/ao/StartTrigger')
            aintask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/ao/StartTrigger')
            
            aouttask.write(awf)
            douttask.write(dwf)

            douttask.start()
            aintask.start()
            aouttask.start()   
            adata = np.asarray(aintask.read(awf.shape[1], 
                                                    timeout=np.shape(awf)[1]/self.sample_freq + 2))
            aouttask.wait_until_done(np.shape(awf)[1]/self.sample_freq + 2)
            return adata
        
    def acquire_linear_calibration(self, awf):
        with nidaqmx.Task() as aouttask, nidaqmx.Task() as aintask:
            aouttask.ao_channels.add_ao_voltage_chan(self.daq_name + "ao0:1", max_val=10, min_val=-10)
            aintask.ai_channels.add_ai_voltage_chan(self.daq_name + "ai0:1")
            aintask.timing.cfg_samp_clk_timing(self.sample_freq, source="", active_edge=Edge.RISING,
                                                   sample_mode=AcquisitionType.FINITE,
                                                   samps_per_chan=np.shape(awf)[1])
            aouttask.timing.cfg_samp_clk_timing(
                self.sample_freq,
                source="",
                active_edge=Edge.RISING,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=awf.shape[1])
            
            aouttask.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/ai/StartTrigger')
            
            aouttask.write(awf)
            
            aouttask.start()   
            adata = np.asarray(aintask.read(np.shape(awf)[1], 
                                                    timeout=np.shape(awf)[1]/self.sample_freq + 2))
            aouttask.wait_until_done(np.shape(awf)[1]/self.sample_freq + 2)
        
        return adata