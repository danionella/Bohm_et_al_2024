import numpy as np
from skimage import io
from scipy import signal
from glob import glob
import pickle

def correct_stack(files):
    # files = glob(r'H:\Urs_data\20230418_fish_PTZ\rec06_plane_[0-9][0-9].tif')
    # load all planes
    print('loading files..')
    planes = [io.imread(f) for f in files]
    print('..done')

    # find timepoints where shutter glitches
    alldif = []
    allpks = []
    for p in planes:
        # average along lines
        tmpproj = np.mean(p, axis=2)
        # remove photobleaching
        tmpproj = tmpproj - signal.medfilt(tmpproj, [201,1])
        # flip signal
        tmpproj[:,0::4] *= -1
        tmpproj[:,1::4] *= -1
        ffdif = np.sqrt(np.diff(np.mean(tmpproj, axis=1))**2)
        # # average difference signal over lines is high when glitch
        # idx = np.argsort(np.mean(tmpproj,axis=0))
        # # only use bottom 1/3 mean intensity lines
        # ffdif = np.mean(np.sqrt((np.diff(tmpproj[:,:], axis=0))**2)[:,idx[0:int(len(idx)/5)]], axis=1)
        # # ffdif = np.mean(np.sqrt((np.diff(tmpproj[:,:], axis=0))**2), axis=1)
        # ffdif -= np.min(ffdif)
        thr = np.mean(np.sort(ffdif[50:])[-30:])/2
        # get positions
        pks = np.where(ffdif>thr)[0]
        alldif.append(ffdif)
        allpks.append(pks)
    
    #combine into one vector
    difcomb = np.zeros(np.sum([s.shape[0] for s in alldif]))
    for n in range(len(alldif)):
        difcomb[n::len(alldif)] = alldif[n]

    pkscomb = np.hstack([(p*len(allpks))+n for n,p in enumerate(allpks)])

    # get the frame where a glitch first happens
    pkssort = np.sort(pkscomb)
    tdif = np.zeros(difcomb.shape)
    tdif[pkssort[pkssort>800]] = 1
    filttdif = signal.medfilt(tdif, 5)
    pkssort = np.where(filttdif>.5)[0]
    glitchpks = [np.where(filttdif[n::16]>.5)[0] for n in range(16)]

    # get first when gap is > 40
    firstpk = pkssort[np.hstack((True, np.diff(pkssort)>40))]

    # generate indices
    # totFrames = np.sum([p.shape[0] for p in planes])
    # framePerPlane = np.repeat(np.arange(0,np.shape(planes[0])[0]), len(planes))
    planePks = [np.ceil((firstpk-n)/16).astype(int)-1 for n in range(16)]
    # planePks = [framePerPlane[firstpk+n] for n in range(16)]
    print(len(planes))
    print(len(planePks))
    print(len(glitchpks))
    print(np.shape(glitchpks[0]))
    return planes, planePks, glitchpks

if __name__ == '__main__':
    files = sorted(glob('./rec[0-9][0-9]_plane_[0-9][0-9].tif'))
    # load and correct files
    planes, planePks, glitchpks = correct_stack(files)
    with open('glitches.pickle', 'wb') as f:
        pickle.dump([planePks, glitchpks], f)
    
