import numpy as np
from skimage import io
from scipy import signal
from glob import glob
import pickle

# use to correct exposure and row shift
def correct_plane(planepk, glitchpks, plane):
    #get indices
    indices = np.zeros(np.shape(plane)[0])
    for n in range(np.shape(planepk)[0]):
        if planepk[n] > 50:
            if n<planepk.size-1:
                indices[planepk[n]+2:planepk[n+1]+2] = (n % 2) + 1
            else:
                indices[planepk[n]+2:] = (n % 2) + 1

    # generate ratio image
    im1 = np.mean(plane[indices==1,:,:], axis=0)
    im2 = np.mean(plane[indices==2,:,:], axis=0)
    ratimg = im2/im1

    # correct each frame
    corrplane = np.float32(plane)
    for n in range(np.shape(planepk)[0]):
        print(str(n) + '/' + str(np.shape(planepk)[0] -1), end='\r')
        if ((n % 2) == 0) & (planepk[n] > 50):
            if n<planepk.size-1:
                corrplane[planepk[n]+2:planepk[n+1]+2,:,:] *= ratimg
            else:
                corrplane[planepk[n]+2:,:,:] *=ratimg
    
    # now correct the row glitches
    # proj = np.mean(corrplane, axis=2)
    # ffdif = np.mean(np.sqrt((np.diff(proj[:,:], axis=0))**2), axis=1)
    # ffdif -= np.min(ffdif)
    # thr = np.max(ffdif[50:])/3
    # pks is now the index of the row glitches
    # pks = np.where(ffdif>thr)[0]
    pks = glitchpks

    #get indices
    indices = np.zeros(np.shape(plane)[0])
    for n in range(pks.size):
        if pks[n] > 50:
            if n<pks.size-1:
                indices[pks[n]+1:pks[n+1]+1] = (n % 2) + 1
            else:
                indices[pks[n]+1:] = (n % 2) + 1

    # generate difference image
    im1 = np.mean(corrplane[indices==1,:,:], axis=0)
    im2 = np.mean(corrplane[indices==2,:,:], axis=0)
    imdiff = im2-im1

     # correct each frame
    for n in range(pks.size):
        if ((n % 2) == 0) & (pks[n] > 50):
            if n<pks.size-1:
                corrplane[pks[n]+1:pks[n+1]+1,:,:] += 0.5*imdiff
            else:
                corrplane[pks[n]+1:,:,:] += 0.5*imdiff
        else:
            if n<pks.size-1:
                corrplane[pks[n]+1:pks[n+1]+1,:,:] -= 0.5*imdiff
            else:
                corrplane[pks[n]+1:,:,:] -= 0.5*imdiff
    return corrplane.astype(np.float32)

def correct_line_noise(plane):
    samp_freq = 500  # Sample frequency (Hz)
    quality_factor = 150.0  # Quality factor
    f = np.array([50, 150])
    # design notch filter with 50 and 150 Hz center freq
    coefs = [signal.iirnotch(notch, quality_factor, samp_freq) for notch in f]

    filteredMov = np.zeros(np.shape(plane))
    for i1 in range(np.shape(filteredMov)[1]):
        print(str(i1) + '/' + str(np.shape(filteredMov)[1]), end='\r')
        for i2 in range(np.shape(filteredMov)[2]):
            outputSignal = plane[:,i1,i2]
            for b_notch, a_notch in coefs:
                # Apply notch filter to the noisy signal using signal.filtfilt
                outputSignal = signal.filtfilt(b_notch, a_notch, outputSignal)
            filteredMov[:,i1,i2] = outputSignal
    print('')
    return filteredMov

if __name__ == '__main__':
    from pathlib import Path
    import sys    
    import re
    file = Path(sys.argv[1])
    plane = io.imread(file)
    with open('glitches.pickle', 'rb') as f:
        dat = pickle.load(f)
    planePks = dat[0]
    glitchpks = dat[1]
    match = re.search(r'plane_(\d+)', str(file))
    n = int(match.group(1))
    print('file number = ' +str(n))
    # load and correct file
    print('correcting shutter noise in plane ' + str(file))
    corrplane = correct_plane(planePks[n], glitchpks[n], plane)
    print('correcting line noise in plane ' + str(file))
    corrplane = correct_line_noise(corrplane)
    io.imsave(str(file).replace('.tif', '_expcorr.tif'), corrplane.astype(np.float32))
    io.imsave(str(file).replace('.tif', '_expcorr_proj.tif'), np.hstack((np.mean(plane[0:200,:,:].astype(np.float32), axis=2),
                                                                         np.mean(corrplane[0:200,:,:].astype(np.float32), axis=2))))
