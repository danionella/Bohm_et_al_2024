import numpy as np
from scipy.signal import medfilt
from scipy.optimize import minimize
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

import cupy as cp
import cupyx.scipy.ndimage as ndigpu

def cupy_fft_transform_warp_polar(self,image):
    
        def warp_polar_gpu(image, radius):
            
            # This funciton was developed by Niels Cautaerts
            
            cx, cy = image.shape[1] / 2, image.shape[0] / 2
            output_shape = (360, radius)
            T = cp.linspace(0, 2*np.pi, output_shape[0]).reshape(output_shape[0], 1)
            R = cp.arange(output_shape[1]).reshape(1, output_shape[1])
            X = R * cp.cos(T) + cx
            Y = R * cp.sin(T) + cy
            coordinates = cp.stack([Y, X])
            polar = ndigpu.map_coordinates(image, coordinates, order=1)
            
            return polar
        
        radius = int(np.ceil(np.sqrt((image.shape[0] / 2)**2 + (image.shape[1] / 2)**2)))
        img_polar = np.zeros((image.shape[0], 360, radius))    
        
        for i in range(image.shape[0]):
            
            tmp=cp.absolute(cp.fft.fftshift(cp.fft.fft2(cp.asarray(image[i]))))
            img_polar[i]= warp_polar_gpu(tmp,radius).get()
            
        return img_polar
    
    
    
    # def fft_transform(image): # function for CPU
    #     img_fft = np.zeros(image.shape)    
    #     for i in range(image.shape[0]):
    #         img_fft[i] = np.abs(fftshift(fft2(image[i])))
    #     return img_fft
    
    
    # def polar_tform(image, radius=600): # function for CPU
    #     # radius = 2000
    #     img_polar = np.zeros((image.shape[0], 360, radius))
    #     for i in range(image.shape[0]):
    #         img_polar[i] = warp_polar(image[i], radius=radius)  #, scaling='log')
    #     return img_polar
    
    
def projection_img(self,image):
    
    img_project = np.zeros((image.shape[0], image.shape[2]))
    for i in range(image.shape[0]):
        img_project[i] = np.log((np.sum(image[i], axis=0)))
    return img_project


def focus_measure(self,image):
    img_measure = np.zeros(image.shape[0])

    base_ind=int(image.shape[1]*0.5)
    for i in range(image.shape[0]):
        
        baseline=image[i][base_ind:]
        inds=np.where(medfilt(image[i][:base_ind],11)<(baseline.mean()+baseline.std()*3))[0]
        if isinstance(inds, np.ndarray):           
            img_measure[i] = inds.min()
        else:
            img_measure[i] = 0
        
    return img_measure

def detect_peak(self,trace):
    
    def func(x,args):
        
        target=args
        x_axis=np.arange(len(target))

        y=np.exp(-x[0]*(x_axis-x[1])**2)
        
        return np.sum((target-y)**2)
    
    trace_norm=(trace-trace.min())/(trace.max()-trace.min())
    x_init=[0.001,len(trace)/2]
    x_bound=[[0,0.01],[5,len(trace)-5]]
    result=minimize(func, x_init, args=(trace_norm), method='L-BFGS-B',bounds=x_bound)
    
    x=np.arange(len(trace_norm))
    
    self.subplot1.clear()
    self.panel1 = Figure(figsize=(3,3))
    self.axes1 = self.panel1.gca()
    self.axes1.set_position([0.2,0.15,0.75,0.8])
    self.canvas1 = FigureCanvas(self.panel1)
    self.subplot1.addWidget(self.canvas1)
    
    self.axes1.cla()
    self.axes1.plot(x-20, trace_norm,'o')
    self.axes1.plot(x-20,np.exp(-result.x[0]*(x-result.x[1])**2))
    self.axes1.plot([result.x[1]-20,result.x[1]-20],[0,1],'r:')
    self.axes1.set_ylabel('Resolution measure')
    self.axes1.set_xlabel('Searched plane (μm)')
    
    return int(result.x[1])
    

def detect_best_focus(self,stack):
    
    
    img_polar = self.cupy_fft_transform_warp_polar(stack) # for GPU
          
    # img_fft = fft_transform(img) # for CPU
    # img_polar = polar_tform(img_fft) # for CPU
    
    img_project = self.projection_img(img_polar)
    img_mea     = self.focus_measure(img_project)
    best_plane  = self.detect_peak(img_mea)
    
    return best_plane
    