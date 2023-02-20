import numpy as np
from scipy import signal

class Waveforms:
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
    
    def fast_waveform(self, freq, vc_amp, vc_pos, exp_time, duration, phase, cam_phase, st,
                      stim_pos, stim_duration, stim_freq, stim_amp):
     
        wf_freq = 1/(freq*exp_time) # in Hz
        vc_amp = vc_amp/2    
        n_samples = int(duration * self.sample_freq)
        time1 = np.linspace(1/self.sample_freq, duration, n_samples) + phase*(1/self.sample_freq)
        time2 = np.linspace(1/self.sample_freq, duration, n_samples) + phase*(1/self.sample_freq)
        if st:
            wf_saw = signal.sawtooth(wf_freq*time1*(2*np.pi),0.5)*vc_amp + vc_pos
            sos = signal.butter(2, 2000/(self.sample_freq/2), btype='low', output='sos')
            wf_saw = signal.sosfiltfilt(sos, wf_saw)
            wf_saw = wf_saw.copy(order='C')
            wf_sin_vc = wf_saw
            wf_sin_ls = wf_saw*self.ls_scale + self.ls_intercept
        else:
            wf_sin_vc = np.sin(wf_freq*time1*(2*np.pi))*vc_amp + vc_pos
            wf_sin_ls = np.sin(wf_freq*time2*(2*np.pi))*vc_amp + vc_pos
            wf_sin_ls = wf_sin_ls*self.ls_scale + self.ls_intercept
        
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
        # dwf = np.full((n_samples,), False)     
        # dwf[0:10] = True
        dwf = signal.square(freq*wf_freq*time1*(2*np.pi) + cam_phase*2*np.pi, 0.2)>0
        
        shutterwave = np.full(n_samples, True)
        shutterwave[-1] = False
        
        stimwf = self.stimulus_waveform(stim_pos, stim_duration, stim_freq, stim_amp, n_samples)
        
        dwf = np.vstack((dwf, shutterwave))
        
        # wf_saw = signal.sawtooth(wf_freq*time*(2*np.pi), 0.5)*wf_amp
        # b, a = signal.butter(2, 1000/(sample_freq/2), btype='low')
        # sos = signal.butter(2, 2000/(sample_freq/2), btype='low', output='sos')
        # wf_saw = signal.sosfiltfilt(sos, wf_saw)
        # wf_saw = wf_saw.copy(order='C')

        awf = np.vstack((wf_sin_vc, wf_sin_ls, stimwf))
        
        return awf, dwf
    
    def calibrate_fft(self, awf, data, linawf, lindata):
        slope_ls, intercept_ls = np.polyfit(linawf[1,1000:-1], lindata[1,1000:-1], 1)
        ls_data = (data[1,:] - intercept_ls) / slope_ls
        print('ls lin slope is: ' + str(slope_ls) + ', ls intercept is: ' + str(intercept_ls))
        slope_vc, intercept_vc = np.polyfit(linawf[0,1000:-1], lindata[0,1000:-1], 1)
        vc_data = (data[0,:] - intercept_vc) / slope_vc
        print('vc lin slope is: ' + str(slope_vc) + ', vc intercept is: ' + str(intercept_vc))
        
        self.ls_lin_slope = slope_ls
        self.ls_lin_intercept = intercept_ls
        self.vc_lin_slope = slope_vc
        self.vc_lin_intercept = intercept_vc
        
        fs_vc = np.fft.fft(awf[0,:])
        fr_vc = np.fft.fft(vc_data)
        
        fs_ls = np.fft.fft(awf[1,:])
        fr_ls = np.fft.fft(ls_data)
        
        self.kernel_vc = (np.conj(fs_vc)*fr_vc)/(np.conj(fs_vc)*fs_vc+.0001)
        self.kernel_ls = (np.conj(fs_ls)*fr_ls)/(np.conj(fs_ls)*fs_ls+.0001)
        
    
    def stack(self, mid_pos, size, n_steps, exp_time):
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
    
    def ramp(self):
        awf = np.linspace(-3, 3, 10000)
        awf = np.hstack((np.ones(1000)*-3, awf))
        awf = np.vstack((awf, awf))
        return awf
    
    def stimulus_waveform(self, start_pos, duration, frequency, amplitude, n_samples):
        stim_wf = np.zeros(n_samples)
        duration = duration/1000
        time1 = np.linspace(1/self.sample_freq, duration, int(duration*self.sample_freq))
        wave = np.sin(frequency*time1*(2*np.pi))*amplitude
        stim_wf[int(start_pos*self.sample_freq):int(start_pos*self.sample_freq) + np.shape(wave)[0]] = wave
        return stim_wf
    
    def check_limits(self,wf):
        if np.any((wf > ((0.333/self.vc_vd_slope)-self.vc_vd_intercept)) | 
                  (wf < ((0.333/self.vc_vd_slope)-self.vc_vd_intercept)*-1)):
            raise ValueError('analog out value for voice coil must be within the limit -0.333 t0 0.333 V') 
# debuging    
if __name__=="__main__":  
    wf = Waveforms(13.08, 15.63, 50000,
                    0.09126277240684635,
                    2.029975214187555e-06)
    # ta, td = wf.stack(-0.132,0.07,70,50.0)
    ta, td = wf.psf_stack(-0.122,0.442,10,50.0, 0.0044)
    # ta, td = wf.calibration_stack((-1.9, -1.0), (-1.2, 1.0), 10, 5.0)
    # ta, td = wf.fast_waveform(8, 0, 0, 0.25, 1, 0, 0, True, 0.5, 20, 1000, 1)
    # ta, td = wf.chirp_waveform(1, 500, 0.02, -0.122, 5)
    # ta = wf.pulses(10, 1, 0.01, -0.122)
    # ta = wf.ramp()
    p = np.transpose(ta)
    print('pause')
