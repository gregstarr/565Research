#%% 565 Research
from pyRinex import readRinexNav,readRinexObsHdf
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft,fftfreq,fftshift
from glob import glob
from pymap3d.coordconv3d import ecef2geodetic,ecef2aer,aer2geodetic
from os.path import isfile
from matplotlib.dates import DateFormatter
fmt = DateFormatter('%H:%M:%S')
#%% receiver array class
class receiverArray():
    # Satellite constants
    f1 = 1575420000
    c = 3.0E8
    wl = c/f1
    
    # read data, make convenient numpy array
    def __init__(self,interval=[np.datetime64('2015-10-07T06:10:00'),np.datetime64('2015-10-07T06:50:00')],
                                directory = 'C:/Users/fuck/Desktop/565 Research/rinex/',sat_num=23):
        
        self.interval_length = int((interval[1]-interval[0]).item().total_seconds())-1
        self.parameters = ['L1','S1','detrended','ACF','PSD']
        self.interval = interval
        self.sat_num = sat_num
        self.directory = directory
        self.flist = glob(self.directory+'*.h5')
        self.data = np.empty((len(self.parameters),self.interval_length,len(self.flist)),complex) #receiver,parameter(L1,S1,detrended,ACF),time
        self.rec_pos = np.empty((len(self.flist),3),float) #rec pos
        self.rec_names = [fn.split('\\')[-1][:4] for fn in self.flist]
        self.times = self.interval[0]+np.arange(self.interval_length)
        if isfile(self.directory+'arraydata.npy') and isfile(self.directory+'recpos.npy'):
            self.data = np.load(self.directory+'arraydata.npy')
            self.rec_pos = np.load(self.directory+'recpos.npy')
            return

        for fn in self.flist:
            print(fn)
            #data extraction
            header,data,svset,obstimes = readRinexObsHdf(fn)
            L1 = data['L1',sat_num,:,'data']
            S1 = data['S1',sat_num,:,'data']
            #interval bounds
            start = np.argmin(np.abs(self.interval[0]-L1.index))
            stop = np.argmin(np.abs(self.interval[1]-L1.index))
            #data grooming/calculation
            phase = np.array(L1.values,float)[start+1:stop]
            snr = np.array(S1.values,float)[start+1:stop]
            fit = np.poly1d(np.polyfit(np.arange(self.interval_length)-np.argmin(phase),phase,6))
            y = fit(np.arange(self.interval_length)-np.argmin(phase))
            detrended = phase-y    
            sigma_phi = np.var(detrended)
            ro_phi = np.correlate(detrended,detrended,mode='full')[self.interval_length//2:self.interval_length+self.interval_length//2]
            ro_phi/=(sigma_phi*np.concatenate((np.arange(self.interval_length//2+1,self.interval_length),
                                               np.arange(self.interval_length,self.interval_length//2,-1))))
            PSD = sigma_phi*ifft(ro_phi)
            #array assignment
            i = self.flist.index(fn)
            self.data[0,:,i] = phase
            self.data[1,:,i] = snr
            self.data[2,:,i] = detrended
            self.data[3,:,i] = ro_phi
            self.data[4,:,i] = PSD
            
            x,y,z = np.array(header['APPROX POSITION XYZ'],float)
            self.rec_pos[i] = ecef2geodetic(x,y,z)
        np.save(self.directory+'arraydata',self.data)
        np.save(self.directory+'recpos',self.rec_pos)
        
        
            
    def plotRecPos(self):
        
        cmap = get_cmap(len(self.rec_names)+1)
        fig = plt.figure()
        fig.add_subplot(111)
        plt.tight_layout(rect=[0,0,.8,1])
        for i in range(len(self.flist)):
            plt.plot(self.rec_pos[i,1],self.rec_pos[i,0],'x',c=cmap(i),ms=10,
                     mew=3,label=self.rec_names[i])
        plt.legend(numpoints=1,bbox_to_anchor=(1.3,1))
        plt.ylim([np.min(self.rec_pos[:,0])-1,np.max(self.rec_pos[:,0])+1])
        plt.xlim([np.min(self.rec_pos[:,1])-1,np.max(self.rec_pos[:,1])+1])
        plt.xlabel('Longitude')
        plt.ylabel('Lattitude')
        plt.title('Receiver Positions')
        plt.grid()
        
    def plotDetrended(self,recs):
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.xaxis.set_major_formatter(fmt)
        n = len(recs)
        cmap = get_cmap(n+1)
        for i in range(n):
            ax.plot(self.times,self.data[2,:,recs[i]],c=cmap(i),label=self.rec_names[recs[i]])
        plt.legend()
        
    def determineSpeed(self,rec):
        nrec = len(self.rec_names)
        self.xcor = np.empty((self.interval_length,nrec),float)
        self.distance = np.empty((nrec,2),float)
        for i in np.arange(rec,nrec+rec):
            self.xcor[:,i%nrec]=np.correlate(self.data[2,:,rec],self.data[2,:,i%nrec],
                                             mode='full')[self.interval_length//2:self.interval_length+self.interval_length//2]
            self.xcor[:,i%nrec]/=(np.std(self.data[2,:,rec])*np.std(self.data[2,:,i%nrec])*
                                          np.concatenate((np.arange(self.interval_length//2+1,self.interval_length),
                                          np.arange(self.interval_length,self.interval_length//2,-1))))
       
        self.delay = self.interval_length//2-np.argmax(self.xcor,0)
        # haversine formula for distance
        # http://www.movable-type.co.uk/scripts/latlong.html
        distance_finder = np.zeros((nrec,nrec))
        distance_finder[:,rec] = np.ones(nrec)*-1
        distance_finder+=np.eye(nrec)
        radians_away = (np.pi/180)*distance_finder@self.rec_pos[:,:2]
        a = (np.sin(radians_away[:,0]/2)**2+np.cos(np.ones(nrec)*
                    self.rec_pos[rec,0]*np.pi/180)*np.cos(radians_away[:,0]+
                    np.ones(nrec)*self.rec_pos[rec,0]*np.pi/180)*
                    np.sin(radians_away[:,1]/2)**2)
        c = 2*np.arctan2(np.sqrt(a),np.sqrt(1-a))
        d = 6371000*c
        theta = np.arctan2(radians_away[:,0],radians_away[:,1])
        self.distance = d[:,None]*np.hstack((np.cos(theta)[:,None],np.sin(theta)[:,None]))
        self.drift_velocity = self.distance/self.delay[:,None]
        
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = Normalize(vmin=0, vmax=N-1)
    scalar_map = ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color
        
if __name__=='__main__':
    # Ipython plotting
    from IPython import get_ipython
    ipy = get_ipython()
    ipy.magic("matplotlib qt")
    
    RA = receiverArray()
    RA.determineSpeed(7)