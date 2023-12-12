import numpy as np
from glob import glob
from skimage import io
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, warp
from skimage.morphology import dilation
from skimage.morphology import disk
from scipy.sparse.linalg import svds
from sklearn.linear_model import Ridge
import os
from scipy import signal
from scipy import stats
import cv2
import psutil
import multiprocessing as mp
import gc

# define helper functions:

def transform_back(mask):
    # transform the mask stack backwards
    maskTrans = [np.zeros(np.shape(im)) for im in imsTrans]
    for n in range(16):
        maskTrans[idx2[n]][:c2,:] = mask[n,:c2,:]
        maskTrans[idx1[n]][c2:,:] = mask[n,c2:,:]

    maskPad = [maskTrans[idx1[0]]]
    for n in range(15):
        s = (np.sum(shifts[0:n+1,1], axis=0), np.sum(shifts[0:n+1,0], axis=0))
        tf = AffineTransform(translation=s)
        imWarp = warp(maskTrans[idx1[n+1]], tf)
        maskPad.append(imWarp)

    maskPad = [maskPad[i] for i in np.argsort(idx1)]

    #unpad images
    maskOrig = [im[starty:endy, startx:endx] for im in maskPad]
    return maskOrig

def get_index():
    r = 0
    planes = np.roll(np.arange(0,16),r)
    up = planes[0:8]
    down = planes[-1:-9:-1]

    idx1 = np.zeros(16, dtype='int')

    for n in range(8):
        idx1[n*2] = up[n]
        idx1[2*n+1] = down[n]

    planes = np.roll(np.arange(0,16),r-1)
    up = planes[0:8]
    down = planes[-1:-9:-1]    
    idx2 = np.zeros(16, dtype='int')
    for n in range(8):
        idx2[n*2] = down[n]
        idx2[2*n+1] = up[n]
    return idx1, idx2

def load_memmap(filenames):
    Yr = []
    for filename in filenames:
        fn_without_path = os.path.split(filename)[-1]
        fn_without_path = fn_without_path.split('.')[0]
        fpart = fn_without_path.split('_')[1:]  # The filename encodes the structure of the map
        d1, d2, d3, T, order = int(fpart[-9]), int(fpart[-7]), int(fpart[-5]), int(fpart[-1]), fpart[-3]
        myshape = tuple(map(lambda x: np.uint64(x), (d1 * d2 * d3, T)))
        Yr.append(np.memmap(filename, mode='r', shape=myshape, dtype=np.float32, order=order))
    if d3 == 1:
        return (Yr, (d1, d2), T)
    else:
        return (Yr, (d1, d2, d3), T)

def signal_filter(sg, freq, fr, order=3, mode='high'):
    """
    Function for high/low passing the signal with butterworth filter
    
    Args:
        sg: 1-d array
            input signal
            
        freq: float
            cutoff frequency
        
        order: int
            order of the filter
        
        mode: str
            'high' for high-pass filtering, 'low' for low-pass filtering
            
    Returns:
        sg: 1-d array
            signal after filtering            
    """
    normFreq = freq / (fr / 2)
    b, a = signal.butter(order, normFreq, mode)
    sg = np.single(signal.filtfilt(b, a, sg, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1)))
    return sg

def denoise_spikes(data, window_length, fr=400,  hp_freq=1,  clip=100, threshold_method='adaptive_threshold', 
                   min_spikes=10, pnorm=0.5, threshold=3,  do_plot=True):
    """ Function for finding spikes and the temporal filter given one dimensional signals.
        Use function whitened_matched_filter to denoise spikes. Two thresholding methods can be 
        chosen, simple or 'adaptive thresholding'.

    Args:
        data: 1-d array
            one dimensional signal

        window_length: int
            length of window size for temporal filter

        fr: int
            number of samples per second in the video
            
        hp_freq: float
            high-pass cutoff frequency to filter the signal after computing the trace
            
        clip: int
            maximum number of spikes for producing templates

        threshold_method: str
            adaptive_threshold or simple method for thresholding signals
            adaptive_threshold method threshold based on estimated peak distribution
            simple method threshold based on estimated noise level 
            
        min_spikes: int
            minimal number of spikes to be detected
            
        pnorm: float
            a variable deciding the amount of spikes chosen for adaptive threshold method

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level

        do_plot: boolean
            if Ture, will plot trace of signals and spiketimes, peak triggered
            average, histogram of heights
            
    Returns:
        datafilt: 1-d array
            signals after whitened matched filter

        spikes: 1-d array
            record of time of spikes

        t_rec: 1-d array
            recovery of original signals

        templates: 1-d array
            temporal filter which is the peak triggered average

        low_spikes: boolean
            True if number of spikes is smaller than 30
            
        thresh2: float
            real threshold in second round of spike detection 
    """
    # high-pass filter the signal for spike detection
    data = signal_filter(data, hp_freq, fr, order=5)
    data = data - np.median(data)
    pks = data[signal.find_peaks(data, height=None)[0]]

    # first round of spike detection    
    if threshold_method == 'adaptive_threshold':
        thresh, _, _, low_spikes = adaptive_thresh(pks, clip, 0.25, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    elif threshold_method == 'simple':
        thresh, low_spikes = simple_thresh(data, pks, clip, 3.5, min_spikes)
        locs = signal.find_peaks(data, height=thresh)[0]
    else:
        # logging.warning("Error: threshold_method not found")
        raise Exception('Threshold_method not found!')

    # spike template
    window = np.int64(np.arange(-window_length, window_length + 1, 1))
    locs = locs[np.logical_and(locs > (-window[0]), locs < (len(data) - window[-1]))]
    PTD = data[(locs[:, np.newaxis] + window)]
    PTA = np.median(PTD, 0)
    PTA = PTA - np.min(PTA)
    templates = PTA

    # whitened matched filtering based on spike times detected in the first round of spike detection
    datafilt = whitened_matched_filter(data, locs, window)    
    datafilt = datafilt - np.median(datafilt)

    # second round of spike detection on the whitened matched filtered trace
    pks2 = datafilt[signal.find_peaks(datafilt, height=None)[0]]
    if threshold_method == 'adaptive_threshold':
        thresh2, falsePosRate, detectionRate, low_spikes = adaptive_thresh(pks2, clip=0, pnorm=pnorm, min_spikes=min_spikes)  # clip=0 means no clipping
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    elif threshold_method == 'simple':
        thresh2, low_spikes = simple_thresh(datafilt, pks2, 0, threshold, min_spikes)
        spikes = signal.find_peaks(datafilt, height=thresh2)[0]
    
    # compute reconstructed signals and adjust shrinkage
    t_rec = np.zeros(datafilt.shape)
    t_rec[spikes] = 1
    t_rec = np.convolve(t_rec, PTA, 'same')   
    factor = np.mean(data[spikes]) / np.mean(datafilt[spikes])
    datafilt = datafilt * factor
    thresh2_normalized = thresh2 * factor
        
    if do_plot:
        plt.figure()
        plt.subplot(211)
        plt.hist(pks, 500)
        plt.axvline(x=thresh, c='r')
        plt.title('raw data')
        plt.subplot(212)
        plt.hist(pks2, 500)
        plt.axvline(x=thresh2, c='r')
        plt.title('after matched filter')
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(np.transpose(PTD), c=[0.5, 0.5, 0.5])
        plt.plot(PTA, c='black', linewidth=2)
        plt.title('Peak-triggered average')
        plt.show()

        plt.figure()
        plt.subplot(211)
        plt.plot(data)
        plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), color='r', marker='o', fillstyle='none',
                 linestyle='none')
        plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), color='g', marker='o', fillstyle='none',
                 linestyle='none')
        plt.subplot(212)
        plt.plot(datafilt)
        plt.plot(locs, np.max(datafilt) * 1.1 * np.ones(locs.shape), color='r', marker='o', fillstyle='none',
                 linestyle='none')
        plt.plot(spikes, np.max(datafilt) * 1 * np.ones(spikes.shape), color='g', marker='o', fillstyle='none',
                 linestyle='none')
        plt.show()

    return datafilt, spikes, t_rec, templates, low_spikes, thresh2_normalized

def simple_thresh(data, pks, clip, threshold=3.5, min_spikes=10):
    """ Simple threshold method for deciding threshold based on estimated noise level.

    Args:
        data: 1-d array
            the input trace
            
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        threshold: float
            threshold for spike detection in simple threshold method 
            The real threshold is the value multiply estimated noise level
    
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    low_spikes = False
    ff1 = -data * (data < 0)
    Ns = np.sum(ff1 > 0)
    std = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
    thresh = threshold * std
    locs = signal.find_peaks(data, height=thresh)[0]
    if len(locs) < min_spikes:
        # logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
        low_spikes = True
    elif ((len(locs) > clip) & (clip > 0)):
        # logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))    
    return thresh, low_spikes

def adaptive_thresh(pks, clip, pnorm=0.5, min_spikes=10):
    """ Adaptive threshold method for deciding threshold given heights of all peaks.

    Args:
        pks: 1-d array
            height of all peaks

        clip: int
            maximum number of spikes for producing templates

        pnorm: float, between 0 and 1, default is 0.5
            a variable deciding the amount of spikes chosen for adaptive threshold method
            
        min_spikes: int
            minimal number of spikes to be detected

    Returns:
        thresh: float
            threshold for choosing spikes

        falsePosRate: float
            possibility of misclassify noise as real spikes

        detectionRate: float
            possibility of real spikes being detected

        low_spikes: boolean
            true if number of spikes is smaller than minimal value
    """
    # find median of the kernel density estimation of peak heights
    spread = np.array([pks.min(), pks.max()])
    spread = spread + np.diff(spread) * np.array([-0.05, 0.05])
    low_spikes = False
    pts = np.linspace(spread[0], spread[1], 2001)
    kde = stats.gaussian_kde(pks)
    f = kde(pts)    
    xi = pts
    center = np.where(xi > np.median(pks))[0][0]

    fmodel = np.concatenate([f[0:center + 1], np.flipud(f[0:center])])
    if len(fmodel) < len(f):
        fmodel = np.append(fmodel, np.ones(len(f) - len(fmodel)) * min(fmodel))
    else:
        fmodel = fmodel[0:len(f)]

    # adjust the model so it doesn't exceed the data:
    csf = np.cumsum(f) / np.sum(f)
    csmodel = np.cumsum(fmodel) / np.max([np.sum(f), np.sum(fmodel)])
    lastpt = np.where(np.logical_and(csf[0:-1] > csmodel[0:-1] + np.spacing(1), csf[1:] < csmodel[1:]))[0]
    if not lastpt.size:
        lastpt = center
    else:
        lastpt = lastpt[0]
    fmodel[0:lastpt + 1] = f[0:lastpt + 1]
    fmodel[lastpt:] = np.minimum(fmodel[lastpt:], f[lastpt:])

    # find threshold
    csf = np.cumsum(f)
    csmodel = np.cumsum(fmodel)
    csf2 = csf[-1] - csf
    csmodel2 = csmodel[-1] - csmodel
    obj = csf2 ** pnorm - csmodel2 ** pnorm
    maxind = np.argmax(obj)
    thresh = xi[maxind]

    if np.sum(pks > thresh) < min_spikes:
        low_spikes = True
        # logging.warning(f'Few spikes were detected. Adjusting threshold to take {min_spikes} largest spikes')
        thresh = np.percentile(pks, 100 * (1 - min_spikes / len(pks)))
    elif ((np.sum(pks > thresh) > clip) & (clip > 0)):
        # logging.warning(f'Selecting top {clip} spikes for template')
        thresh = np.percentile(pks, 100 * (1 - clip / len(pks)))

    ix = np.argmin(np.abs(xi - thresh))
    falsePosRate = csmodel2[ix] / csf2[ix]
    detectionRate = (csf2[ix] - csmodel2[ix]) / np.max(csf2 - csmodel2)
    return thresh, falsePosRate, detectionRate, low_spikes

def whitened_matched_filter(data, locs, window):
    """
    Function for using whitened matched filter to the original signal for better
    SNR. Use welch method to approximate the spectral density of the signal.
    Rescale the signal in frequency domain. After scaling, convolve the signal with
    peak-triggered-average to make spikes more prominent.
    
    Args:
        data: 1-d array
            input signal

        locs: 1-d array
            spike times

        window: 1-d array
            window with size of temporal filter

    Returns:
        datafilt: 1-d array
            signal processed after whitened matched filter
    
    """
    N = np.ceil(np.log2(len(data)))
    censor = np.zeros(len(data))
    censor[locs] = 1
    censor = np.int16(np.convolve(censor.flatten(), np.ones([1, len(window)]).flatten(), 'same'))
    censor = (censor < 0.5)
    noise = data[censor]
    hamm_wind_length = np.min([len(noise), 1000])
    _, pxx = signal.welch(noise, fs=2 * np.pi, window=signal.get_window('hamming', hamm_wind_length), nfft=2 ** N, detrend=False,
                          nperseg=hamm_wind_length)
    Nf2 = np.concatenate([pxx, np.flipud(pxx[1:-1])])
    scaling_vector = 1 / np.sqrt(Nf2)

    cc = np.pad(data.copy(),(0,np.int32(2**N-len(data))),'constant')    
    dd = (cv2.dft(cc,flags=cv2.DFT_SCALE+cv2.DFT_COMPLEX_OUTPUT)[:,0,:]*scaling_vector[:,np.newaxis])[:,np.newaxis,:]
    dataScaled = cv2.idft(dd)[:,0,0]
    PTDscaled = dataScaled[(locs[:, np.newaxis] + window)]
    PTAscaled = np.mean(PTDscaled, 0)
    datafilt = np.convolve(dataScaled, np.flipud(PTAscaled), 'same')
    datafilt = datafilt[:len(data)]
    return datafilt

def blur_per_plane(data, shapes, kernel_size_x=7, kernel_size_y=7, kernel_std_x=1.5,
                            kernel_std_y=1.5, borderType=cv2.BORDER_REPLICATE):
    '''
    Function to apply gaussian blur to every frame of every plane in the context region
    inputs are the 2D data matrix (time x space) and the xy sizes of the context region per frame
    '''
    # initialize results
    dataBlur = np.empty_like(data)
    pid = 0
    # iterate over planes
    for n in range(len(shapes)):
        d1 = shapes[n][0]
        d2 = shapes[n][1]
        if d1*d2 > 0:
            # iterate over frames
            for k in range(np.shape(data)[0]):
                frame = np.reshape(data[k,pid:pid+d1*d2],(d1, d2))
                frame = cv2.GaussianBlur(frame,ksize=(kernel_size_x, kernel_size_y),
                             sigmaX=kernel_std_x,
                             sigmaY=kernel_std_y,
                             borderType=borderType)
                dataBlur[k,pid:pid+d1*d2] = np.reshape(frame, d1*d2)
        pid += d1*d2
    return dataBlur

def run_spikepursuit(cell_n):
    """
    the follwing code is mostly based on spikepursuit.py from the volpy package, originally written by
    @caichangjia adapt based on Matlab code provided by Kaspar Podgorski and Amrita Singh
    """
    try:
        print('running cell number: ' + str(cell_n)) 

        bw = np.where(mask==cell_n, 1,0)

        bwexp = dilation(bw, np.ones(args['context_size']), shift_x=True, shift_y=True)

        d1inds = np.where(np.any(bwexp > 0, axis=(1,2)) > 0)[0]
        d2inds = np.where(np.any(bwexp > 0, axis=(0,2)) > 0)[0]
        d3inds = np.where(np.any(bwexp > 0, axis=(0,1)) > 0)[0]

        contextInds = np.full(bw.shape, 0)
        contextInds[d1inds[0]:d1inds[-1] + 1, d2inds[0]:d2inds[-1] + 1, d3inds[0]:d3inds[-1] + 1] = 1
        contextBack = np.asarray(transform_back(contextInds))
        tmpstack = np.empty_like(np.swapaxes(contextBack, 1, 2))
        for n in range(contextBack.shape[0]):
            tmpstack[n,:,:] = contextBack[n,:,:].T
        contextBack = tmpstack

        shapes = []
        corner = []
        for cb in contextBack:
            d2 = np.where(np.any(cb == 1, axis=0) > 0)[0]
            d1 = np.where(np.any(cb == 1, axis=1) > 0)[0]
            shapes.append((len(d1),len(d2)))
            x = d1[0] if d1.size > 0 else 0
            y = d2[0] if d2.size > 0 else 0
            corner.append((x, y))

        bwBack = np.where(np.asarray(transform_back(bw))==1,1,0)
        # bwBack = np.swapaxes(bwBack, 1, 2)
        tmpstack = np.empty_like(np.swapaxes(bwBack, 1, 2))
        for n in range(bwBack.shape[0]):
            tmpstack[n,:,:] = bwBack[n,:,:].T
        bwBack = tmpstack

        notbw = 1 - dilation(bw, np.tile(disk(args['censor_size'][1]), (2*args['censor_size'][0] + 1, 1, 1)))
        notbwBack = np.where(np.asarray(transform_back(notbw))==1,1,0)
        tmpstack = np.empty_like(np.swapaxes(notbwBack, 1, 2))
        for n in range(notbwBack.shape[0]):
            tmpstack[n,:,:] = notbwBack[n,:,:].T
        notbwBack = tmpstack
        # notbwBack = np.swapaxes(notbwBack, 1, 2)

        bw = bwBack[contextBack==1]
        notbw = notbwBack[contextBack==1]
        # print('loading data...')
        data = [images[n][np.ravel(contextBack[n,:,:]==1),:] for n in range(len(images))]
        data = np.concatenate(data, axis=0).T
        # print('...done')
        # print('RAM memory % used:', psutil.virtual_memory()[2])
        # print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
        data = -data
        bw = (bw > 0)
        notbw = (notbw > 0)

        output = {}
        output['mean_im'] = np.mean(data, axis=0)
        data = data - np.mean(data, 0)
        data = data - np.mean(data, 0)   #do again because of numeric issues
        data_hp = signal_filter(data.T, args['hp_freq_pb'], fr).T  
        data_lp = data - data_hp

        t0 = np.nanmean(data_hp[:, bw], 1)

        # remove any variance in trace that can be predicted from the background principal components
        data_svd = data_hp[:, notbw]
        if data_svd.shape[1] < args['nPC_bg'] + 1:
            raise Exception(f'Too few pixels ({data_svd.shape[1]}) for background extraction (at least {args["nPC_bg"]} needed);'
                            f'please decrease context_size and censor_size')
        Ub, Sb, Vb = svds(data_svd, args['nPC_bg'])
        alpha = args['nPC_bg'] * args['ridge_bg']    # square of F-norm of Ub is equal to number of principal components
        reg = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr').fit(Ub, t0)
        t0 = np.double(t0 - np.matmul(Ub, reg.coef_))

        ts, spikes, t_rec, templates, low_spikes, thresh = denoise_spikes(t0, 
                                            window_length, fr, hp_freq=args['hp_freq'], clip=args['clip'],
                                            threshold_method=args['threshold_method'], 
                                            pnorm=args['pnorm'], threshold=args['threshold'], 
                                            min_spikes=args['min_spikes'], do_plot=False)
        
        output['rawROI'] = {}
        output['rawROI']['t'] = t0.copy()
        output['rawROI']['ts'] = ts.copy()
        output['rawROI']['spikes'] = spikes.copy()
        if weights_init is None:
            output['rawROI']['weights'] = bw.copy()
        else:
            output['rawROI']['weights'] = weights_init.copy()
        output['rawROI']['t'] = output['rawROI']['t'] * np.mean(t0[output['rawROI']['spikes']]) / np.mean(
            output['rawROI']['t'][output['rawROI']['spikes']])  # correct shrinkage
        output['rawROI']['templates'] = templates
        num_spikes = [spikes.shape[0]]

        # prebuild the regression matrix generate a predictor for ridge regression
        pred = np.empty_like(data_hp)
        pred[:] = data_hp
        pred = np.hstack((np.ones((data_hp.shape[0], 1), dtype=np.single), blur_per_plane(pred, shapes)))

        # cross-validation of regularized regression parameters
        lambdamax = np.single(np.linalg.norm(pred[:, 1:], ord='fro') ** 2)
        lambdas = lambdamax * np.logspace(-4, -2, 3)

        s_max = 1
        l_max = 2
        sigma = args['sigmas'][s_max]

        recon = np.empty_like(data_hp)
        recon[:] = data_hp
        recon = np.hstack((np.ones((data_hp.shape[0], 1), dtype=np.single), blur_per_plane(recon, shapes,
                                kernel_size_x=np.int32(2 * np.ceil(2 * sigma) + 1),
                                kernel_size_y=np.int32(2 * np.ceil(2 * sigma) + 1),
                                kernel_std_x=sigma, kernel_std_y=sigma,
                                borderType=cv2.BORDER_REPLICATE)))

        # refine weights and estimate spike times for several iterations 
        # print('starting iterations for cell: ' + str(cell_n))
        for iteration in range(args['n_iter']):
            # update weights
            tr = np.single(t_rec.copy())

            Ri = Ridge(alpha=lambdas[l_max], fit_intercept=True, solver='lsqr')
            Ri.fit(recon, tr)
            weights = Ri.coef_
            weights[0] = Ri.intercept_

            # update the signal            
            t = np.matmul(recon, weights)
            t = t - np.mean(t)

            # ridge regression to remove background components
            b = Ridge(alpha=alpha, fit_intercept=False, solver='lsqr').fit(Ub, t).coef_
            t = t - np.matmul(Ub, b)

            # correct shrinkage
            weights = weights * np.mean(t0[spikes]) / np.mean(t[spikes])
            t = np.double(t * np.mean(t0[spikes]) / np.mean(t[spikes]))

            # estimate spike times
            ts, spikes, t_rec, templates, low_spikes, thresh = denoise_spikes(t, 
                        window_length, fr,  hp_freq=args['hp_freq'], clip=args['clip'],
                        threshold_method=args['threshold_method'], pnorm=args['pnorm'], 
                        threshold=args['threshold'], min_spikes=args['min_spikes'], do_plot=False)

            num_spikes.append(spikes.shape[0])
        # compute SNR 
        if len(spikes)>0:
            t = t - np.median(t)
            selectSpikes = np.zeros(t.shape)
            selectSpikes[spikes] = 1
            sgn = np.mean(t[selectSpikes > 0])
            ff1 = -t * (t < 0)
            Ns = np.sum(ff1 > 0)
            noise = np.sqrt(np.divide(np.sum(ff1**2), Ns)) 
            snr = sgn / noise
        else:
            snr = 0
        
        # locality test       
        matrix = np.matmul(np.transpose(pred[:, 1:]), t_rec)
        sigmax = np.sqrt(np.sum(np.multiply(pred[:, 1:], pred[:, 1:]), axis=0))
        sigmay = np.sqrt(np.dot(t_rec, t_rec))
        IMcorr = matrix / sigmax / sigmay
        maxCorrInROI = np.max(IMcorr[bw])
        if np.any(IMcorr[notbw] > maxCorrInROI):
            locality = False
        else:
            locality = True

        # weights in the FOV
        weights_FOV = np.zeros(contextBack.shape)
        weights_FOV[contextBack==1] = weights[1:]
        
        # subthreshold activity extraction    
        t_sub = t.copy() - t_rec
        t_sub = signal_filter(t_sub, args['sub_freq'], fr, order=5, mode='low') 

        # output
        output['cell_n'] = cell_n
        output['t'] = t
        output['ts'] = ts
        output['t_rec'] = t_rec        
        output['t_sub'] = t_sub
        output['spikes'] = spikes
        output['low_spikes'] = low_spikes
        output['num_spikes'] = num_spikes
        output['templates'] = templates
        output['snr'] = snr
        output['thresh'] = thresh
        output['weights'] = weights_FOV
        output['locality'] = locality    
        output['context_coord'] = {'shapes': shapes, 'corner': corner}
        output['F0'] = np.abs(np.nanmean(data_lp[:, bw] + output['mean_im'][bw][np.newaxis, :], 1))
        output['dFF'] = t / output['F0']
        output['rawROI']['dFF'] = output['rawROI']['t'] / output['F0']
        # print('finished cell: ' + str(cell_n))
        del data
        del data_hp
        del recon
        del pred
        gc.collect()
        return output
    except:
        print('problem with cell #: ' + str(cell_n))
        output = False
        return output
        

    
# run the analysis 

#load mean fluorescence for each plane
imfiles = sorted(glob('/sc-external/ag-judkewitz-hpc-rw/user/urs/20230425_fish_PTZ/rec01/rec01_plane_[0-9][0-9]_expcorr_motion_corrected_meanFrame.tif'))
print(imfiles)
ims = []
for f in imfiles:
    ims.append(io.imread(f))
    # print('loading image: ' + str(f))

# load the 3D segmentation
masks_3d = np.load('/sc-external/ag-judkewitz-hpc-rw/user/urs/20230425_fish_PTZ/rec01/y_shift_stack_seg.npy', allow_pickle=True)
mask = masks_3d[()]['masks']

idx1, idx2 = get_index()

# recalculate the shifts between the planes due to the lateral displacement of the voicecoil
shifts = []
# n = [k for k in range(16)]
n = idx1
# n.append(0)
for i in range(15):
    shift, error, diffphase = phase_cross_correlation(ims[n[i]][:62,:], ims[n[i+1]][:62,:], upsample_factor=20, normalization=None)
    shifts.append(shift)
# set x shift to 0 for this recording, it shifts it far away otherwise
shifts = [[n[0], 0.] for n in shifts]
shifts = np.asarray(shifts)
# print(shifts)

# register all planes to each other
# find min and max shift and pad input images
miny = np.floor(np.min(np.cumsum(shifts[:,0]))).astype('int')
maxy = np.ceil(np.max(np.cumsum(shifts[:,0]))).astype('int')
minx = np.floor(np.min(np.cumsum(shifts[:,1]))).astype('int')
maxx = np.ceil(np.max(np.cumsum(shifts[:,1]))).astype('int')
# print(str(minx) + ', ' + str(maxx))

#pad images
bigy = np.shape(ims[0])[0] + np.abs(miny) + np.abs(maxy)
bigx = np.shape(ims[0])[1] + np.abs(minx) + np.abs(maxx)
# print(str(bigy) + ', ' + str(bigx))
starty = np.abs(miny)
endy = np.shape(ims[0])[0] + np.abs(miny)
startx = np.abs(minx)
endx = np.shape(ims[0])[1] + np.abs(minx)
imsPad = []
for im in ims:
    bigim = np.zeros((bigy, bigx))
    bigim[starty:endy, startx:endx] = im
    imsPad.append(bigim)

imsTrans = [imsPad[idx1[0]]]
for n in range(15):
    s = (np.sum(shifts[0:n+1,1], axis=0), np.sum(shifts[0:n+1,0], axis=0))
    tf = AffineTransform(translation=s)
    imWarp = warp(imsPad[idx1[n+1]], tf.inverse)
    imsTrans.append(imWarp)
imsTrans = [imsTrans[i] for i in np.argsort(idx1)]
# print('imsTrans shape: ' + str(imsTrans[0].shape))

c2 = np.ceil(imsTrans[0].shape[0]/2).astype('int')

args ={'context_size': (4,35,35), 'censor_size': (1,12), 'hp_freq_pb': 2, 'nPC_bg': 8, 'ridge_bg': 0.01,
      'hp_freq': 1,
      'clip': 100,
      'threshold_method': 'adaptive_threshold',
      'pnorm': 0.5,
      'threshold': 4,
      'min_spikes': 10,
      'template_size': 0.01,
      'sigmas': np.array([1, 1.5, 2]),
      'n_iter': 2,
      'sub_freq': 20}
weights_init = None
fr = 500
window_length = int(fr * args['template_size'])
fnames = sorted(glob('/sc-external/ag-judkewitz-hpc-rw/user/urs/20230425_fish_PTZ/rec01/*.mmap'))

# load memory maped data
images, dims, T = load_memmap(fnames) 

pool = mp.Pool(10)

# run volpy
out = pool.map(run_spikepursuit, [cell_n for cell_n in range(1,np.max(mask)+1)])

# out = []
# for cell_n in range(1,np.max(mask)+1):
#     print('running cell number: ' + str(cell_n))
#     out.append(run_spikepursuit(cell_n))

# save results
np.save('/sc-external/ag-judkewitz-hpc-rw/user/urs/20230425_fish_PTZ/rec01/volpy_3D_results.npy', out)
pool.close()
print('finished')