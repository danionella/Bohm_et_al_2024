from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.volpy.volparams import volparams
import caiman as cm
from skimage import io
import gc
import numpy as np
import os
import sys

def run_correction(file):
    fnames = file

   
    c, dview, n_processes = cm.cluster.setup_cluster(
             backend='local', n_processes=None, single_thread=False)
    print('running with ' + str(n_processes))

    fr = 500                                        # sample rate of the movie

    # motion correction parameters
    pw_rigid = False                                # flag for pw-rigid motion correction
    gSig_filt = (3, 3)                              # size of filter, in general gSig (see below),
                                                    # change this one if algorithm does not work
    max_shifts = (5, 5)                             # maximum allowed rigid shift
    strides = (48, 48)                              # start a new patch for pw-rigid motion correction every x pixels
    overlaps = (24, 24)                             # overlap between pathes (size of patch strides+overlaps)
    max_deviation_rigid = 3                         # maximum deviation allowed for patch with respect to rigid shifts
    border_nan = 'copy'

    opts_dict = {
        'fnames': fnames,
        'fr': fr,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'gSig_filt': gSig_filt,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': border_nan
    }

    opts = volparams(params_dict=opts_dict)


    mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    print('done..')
    print('loading mmap file' + str(mc.mmap_file[0]))
    ff = cm.load(mc.mmap_file[0])
    print('done..')
    ff = np.asarray(ff)

    #imgflat = np.reshape(ff, (ff.shape[0],-1))
    #print('calculate average frame')
    #av = np.mean(imgflat, axis=0)
    #imgsel = imgflat[:,av>30]
    #bg = np.mean(imgsel, axis=1)
    #bg /= np.mean(bg)
    #bg = bg[:, None, None]
    #print('correct background fluctuation')
    #ff = ff/bg
    print('saving images')
    #io.imsave(str(fnames) + '_motion_corrected.tif', ff.astype('uint8'), dtype='uint8')
    io.imsave(str(fnames).replace('.tif', '_motion_corrected.tif'), ff)
    io.imsave(str(fnames).replace('.tif', '_motion_corrected_meanFrame.tif'), np.mean(ff, axis=0))

    dview.terminate()
    # os.remove(mc.mmap_file[0])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("provide path to movie")
   #for arg in sys.argv:
   #    print(arg)
    from pathlib import Path    
    file = Path(sys.argv[1])
    print('processing file: ' + str(file))
    run_correction(file)
