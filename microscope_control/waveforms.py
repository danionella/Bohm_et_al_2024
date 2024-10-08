import numpy as np
from scipy import signal

class Waveforms:
    """waveform class to generate all the controll waveforms for galvo, voice coil and camera trigger
    """
    def __init__(self, ls_scale, ls_intercept, sample_freq,
                 vc_vd_slope, vc_vd_intercept):
        self.ls_scale = ls_scale
        self.ls_intercept = ls_intercept
        self.sample_freq = sample_freq
        self.kernel_vc = None
        self.kernel_ls = None
        self.vc_vd_slope = vc_vd_slope
        self.vc_vd_intercept = vc_vd_intercept
        self.ls_lin_slope = 1
        self.ls_lin_intercept = 0
        self.vc_lin_slope = 1
        self.vc_lin_intercept = 0
        self.ls_out_scale = 1
        self.ls_out_intercept = 1
    
    def fast_waveform(self, freq, vc_amp, vc_pos, exp_time, duration, phase, cam_phase, st,
                      stim_pos, stim_duration, stim_freq, stim_amp):
        """generate waveform for fast acquisition

        Args:
            freq (int): frequncy of the volume acquisition
            vc_amp (float): amplitude (z-range)
            vc_pos (float): voice coil zero position
            exp_time (float): exposure time in s
            duration (float): duration of the waveform in s
            phase (float): phase of the voice coil signal relative to the galvo signal s
            cam_phase (int): camera phase in samples
            st (bool): if TRUE output is sawtooth instead of sine
            stim_pos (float): time point of the stimulation in s
            stim_duration (float): duration of the stimulation in s
            stim_freq (float): frequency of the stimulation
            stim_amp (float): amplitude of the stimulation

        Returns:
            awf, dwf (numpy array): analog and digital waveform
        """        
     
        wf_freq = 1/(freq*exp_time) # in Hz
        vc_amp = vc_amp/2    
        n_samples = int(duration * self.sample_freq)
        time1 = np.linspace(1/self.sample_freq, duration, n_samples) + phase*(1/self.sample_freq)
        time2 = np.linspace(1/self.sample_freq, duration, n_samples)
        # if sawtooth, filter at 2kHz to smooth edges
        if st:
            wf_saw = signal.sawtooth(wf_freq*time1*(2*np.pi),0.5)*vc_amp + vc_pos
            sos = signal.butter(2, 2000/(self.sample_freq/2), btype='low', output='sos')
            wf_saw = signal.sosfiltfilt(sos, wf_saw)
            wf_sin_vc = wf_saw.copy(order='C')
            wf_saw = signal.sawtooth(wf_freq*time2*(2*np.pi),0.5)*vc_amp + vc_pos
            sos = signal.butter(2, 2000/(self.sample_freq/2), btype='low', output='sos')
            wf_saw = signal.sosfiltfilt(sos, wf_saw)
            wf_sin_ls = wf_saw.copy(order='C')
            # wf_sin_ls = wf_saw*self.ls_scale + self.ls_intercept
        else:
            wf_sin_vc = np.sin(wf_freq*time1*(2*np.pi))*vc_amp + vc_pos
            wf_sin_ls = np.sin(wf_freq*time2*(2*np.pi))*vc_amp + vc_pos
            # wf_sin_ls = wf_sin_ls*self.ls_scale + self.ls_intercept

        # if a calibration kernel exists, use it
        if self.kernel_vc is not None and self.kernel_ls is not None:
            # fs = np.fft.fft(wf_sin_vc)
            # k = self.kernel_vc
            # dcfs = (np.conj(k)*fs)/(np.conj(k)*k+.0001)
            # wf_sin_vc = np.real(np.fft.ifft(dcfs))
            fs_vc = np.fft.fft(wf_sin_vc)
            fr_vc = fs_vc*self.kernel_vc
            
            response_vc = np.real(np.fft.ifft(fr_vc))
            response_ls = response_vc*self.ls_out_scale + self.ls_out_intercept
            
            fs = np.fft.fft(response_ls)
            k = self.kernel_ls
            dcfs = (np.conj(k)*fs)/(np.conj(k)*k+.0001)
            wf_sin_ls = np.real(np.fft.ifft(dcfs))
        else:
            wf_sin_ls = wf_sin_ls*self.ls_scale + self.ls_intercept
        
        #check the resulting waveform doesn't exceed the limits
        self.check_limits(wf_sin_vc)
        
        exp_time_us = int(exp_time*1e6)
        period = np.full(exp_time_us, True)
        period[-8:] = False
        dwf = np.tile(period, int(duration*1e6/exp_time_us))
        dwf = np.roll(dwf, cam_phase)
        # dwf = np.full((n_samples,), False)     
        # dwf[0:10] = True
        # dc = (exp_time-8e-6)/exp_time
        # dc = int(dc*1000)/1000
        # time3 = np.linspace(1/1000000, duration, int(duration * 1000000)) + phase*(1/1000000)
        # dwf = signal.square(freq*wf_freq*time3*(2*np.pi) + cam_phase*2*np.pi, dc)>0
        
        # shutterwave = np.full(n_samples, True)
        # shutterwave[-1] = False
        
        # get simulation waveform
        stimwf = self.stimulus_waveform(stim_pos, stim_duration, stim_freq, stim_amp, n_samples)
        
        # dwf = np.vstack((dwf, shutterwave))
        
        # wf_saw = signal.sawtooth(wf_freq*time*(2*np.pi), 0.5)*wf_amp
        # b, a = signal.butter(2, 1000/(sample_freq/2), btype='low')
        # sos = signal.butter(2, 2000/(sample_freq/2), btype='low', output='sos')
        # wf_saw = signal.sosfiltfilt(sos, wf_saw)
        # wf_saw = wf_saw.copy(order='C')

        awf = np.vstack((wf_sin_vc, wf_sin_ls, stimwf))
        
        return awf, dwf
    
    def calibrate_fft(self, awf, data, linawf, lindata):
        """generate the calibration kernel, don't use because voice coil position signal is not reliable

        Args:
            awf (numpy array): analog waveform
            data (numpy array): data from uncalibrated acquisition run
            linawf (numpy array): waveform for linear calibration
            lindata (numby array): data from linear calibration
        """
        slope_ls, intercept_ls = np.polyfit(linawf[1,1000:-1], lindata[1,1000:-1], 1)
        # ls_data = (data[1,:] - intercept_ls) / slope_ls
        ls_data = data[1,:]
        print('ls lin slope is: ' + str(slope_ls) + ', ls intercept is: ' + str(intercept_ls))
        slope_vc, intercept_vc = np.polyfit(linawf[0,1000:-1], lindata[0,1000:-1], 1)
        # vc_data = (data[0,:] - intercept_vc) / slope_vc
        vc_data = data[0,:]
        print('vc lin slope is: ' + str(slope_vc) + ', vc intercept is: ' + str(intercept_vc))
        
        self.ls_lin_slope = slope_ls
        self.ls_lin_intercept = intercept_ls
        self.vc_lin_slope = slope_vc
        self.vc_lin_intercept = intercept_vc
        # manual hack to correct for laggin vc signal at 500 Hz
        shift = 6
        bias = 0.0009538558075735791
        fs_vc = np.fft.fft(awf[0,:])
        fr_vc = np.fft.fft(np.roll(vc_data, -shift)-bias)
        
        fs_ls = np.fft.fft(awf[1,:])
        fr_ls = np.fft.fft(ls_data)
        
        self.kernel_vc = (np.conj(fs_vc)*fr_vc)/(np.conj(fs_vc)*fs_vc+.0001)
        self.kernel_ls = (np.conj(fs_ls)*fr_ls)/(np.conj(fs_ls)*fs_ls+.0001)
        
    def beat_waveform(self, freq, vc_amp, vc_pos, exp_time, duration, phase, cam_phase, st,
                      stim_pos, stim_duration, stim_freq, stim_amp):
        awf, dwf = self.fast_waveform(freq, vc_amp, vc_pos, exp_time,
                                      duration, phase, cam_phase, st,
                                      stim_pos, stim_duration, stim_freq, stim_amp)
        vc_amp = vc_amp/2    
        n_samples = int(duration * self.sample_freq)
        time1 = np.linspace(1/self.sample_freq, duration, n_samples) + phase*(1/self.sample_freq)
        wf_freq = 2/duration # two times faster that the duration of the entire recording
        wf_saw = signal.sawtooth(wf_freq*time1*(2*np.pi),0.5)*vc_amp*1.4 + vc_pos
        wf_ls = wf_saw*self.ls_scale + self.ls_intercept
        
        
        # 1 sample phase shift hack for 500 Hz volumes and 50000 sample rate
        
        # wf_ls = np.zeros(np.shape(awf[1,:]))
        # for n, idx in enumerate(np.arange(0,n_samples, 100)):
        #     wf_ls[idx:idx+100] = awf[1,idx-n:idx-n+100]
        
        awf[1,:] = wf_ls
        
        return awf, dwf
        
    def stack(self, mid_pos, size, n_steps, exp_time):
        """generate waveform to acquire a slow stack

        Args:
            mid_pos (float): middle position of the stack
            size (float): z extend
            n_steps (int): number of slices
            exp_time (float): exposure time of each camera frame in ms

        Returns:
            numpy array: driving waveform
        """
        exp_time = exp_time/1000;
        buffer_samples = 0.05 * self.sample_freq;
        step_samples = int(exp_time*self.sample_freq + buffer_samples)
        tot_samples = step_samples*n_steps
        awf = np.zeros([2,tot_samples])
        dwf = np.full([tot_samples], False)
        vc_zpos = np.linspace(mid_pos - size/2, mid_pos + size/2, n_steps)
        ls_zpos = vc_zpos*self.ls_scale + self.ls_intercept
        for z in range(n_steps):
            awf[0,step_samples*z:step_samples*z + step_samples] = vc_zpos[z]
            awf[1,step_samples*z:step_samples*z + step_samples] = ls_zpos[z]
            dwf[step_samples*z + int(0.025*self.sample_freq):\
                step_samples*z + int(0.035*self.sample_freq)] = True
        return awf, dwf
    
    def psf_stack(self, mid_pos, size, n_steps, exp_time, step_size):
        """generate waveform to take a stack with multiple positions for the lightsheet and for each
        lightsheet position 31 additional defocus steps of the voice coil

        Args:
            mid_pos (float): center position of the stack
            size (float): z-extend
            n_steps (int): number of light sheet position
            exp_time (float): camera exposure time
            step_size (float): step size of each voice coil defocus step

        Returns:
            numpy array: driving wavefom
        """
        exp_time = exp_time/1000;
        buffer_samples = 0.05 * self.sample_freq;
        step_samples = int(exp_time*self.sample_freq + buffer_samples)
        tot_pictures = n_steps*21
        tot_samples = step_samples*tot_pictures
        awf = np.zeros([2,tot_samples])
        dwf = np.full([tot_samples], False)
        
        vc_zpos1 = np.linspace(mid_pos - size/2, mid_pos + size/2, n_steps)
        vc_zpos2 = np.linspace(-1*step_size*10,step_size*10,21)
        ls_zpos1 = vc_zpos1*self.ls_scale + self.ls_intercept
        
        
        it = 0
        for z1 in range(n_steps):
            for z2 in range(len(vc_zpos2)):
                awf[0,step_samples*it:step_samples*it + step_samples] = vc_zpos1[z1] + vc_zpos2[z2]
                awf[1,step_samples*it:step_samples*it + step_samples] = ls_zpos1[z1] 
                dwf[step_samples*it + int(0.025*self.sample_freq):\
                    step_samples*it + int(0.035*self.sample_freq)] = True
                it += 1
        return awf, dwf
    
    def calibration_stack(self, pos1, pos2, n_steps, exp_time, step_size):
        """waveform to generate calibration stack by keeping voice coil at multiple position and varying light sheet
        (inverse to psf stack)

        Args:
            pos1 (list): top end of the stack
            pos2 (list): bottom end of the stack
            n_steps (int): number of slices
            exp_time (float): camera exposure time
            step_size (float): z step size of each defocus step

        Returns:
            numpy array: driving waveform
        """
        bottom_pos = (np.min((pos1[0], pos2[0])), np.min((pos1[1], pos2[1])))
        top_pos = (np.max((pos1[0], pos2[0])), np.max((pos1[1], pos2[1])))
        
        vc_zpos1 = np.linspace(bottom_pos[0], top_pos[0], n_steps)
        ls_zpos1 = np.linspace(bottom_pos[1], top_pos[1], n_steps)
        ls_zpos2 = np.linspace(-1*step_size*15,step_size*15,31)
        
        exp_time = exp_time/1000;
        buffer_samples = 0.05 * self.sample_freq;
        step_samples = int(exp_time*self.sample_freq + buffer_samples)
        tot_pictures = n_steps*len(ls_zpos2)
        tot_samples = step_samples*tot_pictures
        awf = np.zeros([2,tot_samples])
        dwf = np.full([tot_samples], False)
        
        it = 0
        for z1 in range(n_steps):
            for z2 in range(len(ls_zpos2)):
                awf[0,step_samples*it:step_samples*it + step_samples] = vc_zpos1[z1] 
                awf[1,step_samples*it:step_samples*it + step_samples] = ls_zpos1[z1] + ls_zpos2[z2]
                dwf[step_samples*it + int(0.025*self.sample_freq):\
                    step_samples*it + int(0.035*self.sample_freq)] = True
                it += 1

        self.check_limits(awf[0,:])
        return awf, dwf
        
    def pulses(self, n_pulses, duration, amp, zero_pos):
        n_samples = duration*self.sample_freq
        vc_pos = np.full(n_samples, zero_pos)
        ls_pos = vc_pos*self.ls_scale + self.ls_intercept
        pulse_pos = np.arange(n_samples/n_pulses,n_samples,n_samples/n_pulses, dtype=int)
        vc_pos[pulse_pos] = vc_pos[0] + amp
        ls_pos[pulse_pos] = ls_pos[0] + amp
        self.check_limits(vc_pos)
        
        awf = np.vstack((vc_pos, ls_pos))     
        return awf
    
    def chirp_waveform(self, start_freq, end_freq, size, vc_pos, duration):
        vc_amp = size/2    
        n_samples = int(duration * self.sample_freq)
        time1 = np.linspace(1/self.sample_freq, duration, n_samples)
        wf_sin_vc = signal.chirp(time1, start_freq, time1[-1], end_freq,
                                 method='logarithmic')*vc_amp + vc_pos
        wf_sin_ls = wf_sin_vc*self.ls_scale + self.ls_intercept
        
        if self.kernel_vc is not None:
            fs = np.fft.fft(wf_sin_vc)
            k = self.kernel_vc
            dcfs = (np.conj(k)*fs)/(np.conj(k)*k+.0001)
            wf_sin_vc = np.real(np.fft.ifft(dcfs))
            fs = np.fft.fft(wf_sin_ls)
            k = self.kernel_ls
            dcfs = (np.conj(k)*fs)/(np.conj(k)*k+.0001)
            wf_sin_ls = np.real(np.fft.ifft(dcfs))
        self.check_limits(wf_sin_vc)

        off = signal.chirp(time1, start_freq, time1[-1], end_freq,
                         method='logarithmic', phi=90)
        idx = np.where(np.diff(np.signbit(off)))[0]
        dwf = np.full(off.shape, False)
        for n in range(5):
            dwf[idx[1:-1:2] + n] = True

        awf = np.vstack((wf_sin_vc, wf_sin_ls))
        
        return awf, dwf
    
    def multi_freq_sawtooth(self, start_freq, end_freq, n_cycles, n_steps, amp, offset):
        freqs = np.linspace(start_freq, end_freq, n_steps)
        awf = np.zeros([1, int(np.ceil(np.sum(n_cycles*self.sample_freq/freqs)))])
        pos = 0
        for n, f in enumerate(freqs):
            t = np.linspace(1/self.sample_freq, n_cycles/f, int(self.sample_freq*n_cycles/f))
            a = signal.sawtooth(f*t*(2*np.pi),0.5)*amp
            awf[0,pos:pos+a.shape[0]] = a
            pos = pos + a.shape[0]
        awf = awf + offset    
        return awf
    
    def ramp(self):
        awf = np.linspace(-3, 3, 10000)
        awf = np.hstack((np.ones(1000)*-3, awf))
        awf = np.vstack((awf, awf))
        return awf
    
    def stimulus_waveform(self, start_pos, duration, frequency, amplitude, n_samples):
        """waveform for sine auditory stimulus

        Args:
            start_pos (float): start of the simulus in s
            duration (float): duration of the stimulus in s
            frequency (float): frequency
            amplitude (float): amplitude
            n_samples (int): total duration of the waveform in number of samples

        Returns:
            numpy array: stimulus waveform
        """
        stim_wf = np.zeros(n_samples)
        duration = duration/1000
        time1 = np.linspace(1/self.sample_freq, duration, int(duration*self.sample_freq))
        wave = np.sin(frequency*time1*(2*np.pi))*amplitude
        stim_wf[int(start_pos*self.sample_freq):int(start_pos*self.sample_freq) + np.shape(wave)[0]] = wave
        return stim_wf
    
    def check_limits(self,wf):
        """check that voice coil waveform stays within limits

        Args:
            wf (numpy array): voice coil driving waveform

        Raises:
            ValueError: raise error if outside of bounds
        """
        if np.any((wf > ((0.333/self.vc_vd_slope)-self.vc_vd_intercept)) | 
                  (wf < ((0.333/self.vc_vd_slope)-self.vc_vd_intercept)*-1)):
            raise ValueError('analog out value for voice coil must be within the limit -0.333 t0 0.333 V') 
# debuging    
if __name__=="__main__":  
    wf = Waveforms(13.08, 15.63, 50000,
                    0.09126277240684635,
                    2.029975214187555e-06)
    # ta, td = wf.stack(-0.132,0.07,70,50.0)
    # ta, td = wf.psf_stack(-0.122,0.442,10,50.0, 0.0044)
    ta = wf.multi_freq_sawtooth(1, 200, 5, 5, 0.4, 0)
    # ta, td = wf.calibration_stack((-1.9, -1.0), (-1.2, 1.0), 10, 5.0)
    # ta, td = wf.fast_waveform(8, 0, 0, 0.25, 1, 0, 0, True, 0.5, 20, 1000, 1)
    # ta, td = wf.chirp_waveform(1, 500, 0.02, -0.122, 5)
    # ta = wf.pulses(10, 1, 0.01, -0.122)
    # ta = wf.ramp()
    p = np.transpose(ta)
    print('pause')
