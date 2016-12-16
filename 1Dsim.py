#%% Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import fft,ifft,fftfreq,fftshift
from scipy.signal import blackman

"""
    def calcDiffraction(self):
        #angular spectrum
        #window = np.zeros(self.N)
        #window[950:1050] = 1
        outgoing_wave = np.exp(1j*self.phase_screen)
        #w=blackman(self.N)
        self.Ehat = ifft(outgoing_wave)
        M = np.concatenate((np.arange(0,self.N/2),np.arange(self.N/2,0,-1)))
        mdq2 = (self.dQ*M)**2
        
        self.field = np.empty((self.N,self.N),complex)
        for i in range(self.N):
            self.field[i] = fft(np.exp(-.5j*self.Z[i]*mdq2/self.k)*self.Ehat)
    
        self.intensity = np.abs(self.field)**2
        
        #focal_index = np.argmax(np.max(intensity,1))
        #plt.plot(self.x,np.ones(N)*Z[focal_index],'k')      
        #D = 2*np.sqrt(np.pi*Z) + 2*phi0*np.pi*Z
        #plt.plot((D+X)[D+X<0],Z[D+X<0],'k')
        #plt.plot(-1*(D+X)[D+X<0],Z[D+X<0],'k')        
        #print('Focal Length Z={}, z={}'.format(Z[focal_index],z[focal_index]))        
"""

#%% 1D Simulation Class
class Simulation1D():
    
    def __init__(self,l=512,N=65536,wavelength=1,phi0=10):
        
        self.l=l
        self.N=N
        self.wavelength=wavelength
        self.k = 2*np.pi/self.wavelength
        self.phi0 = phi0
        self.j = np.concatenate((np.arange(0,self.N/2),np.arange(self.N/2,0,-1))) #circular convolution
        
        self.x = np.arange(-self.N/2,self.N/2)
        
    def propagate(self,zeta):
        self.z = zeta*self.wavelength*self.l**2
        rg = self.wavelength**2*self.z/self.N**2 # ((lambda*sqrt(z))/length)^2
        w = np.exp(1j*np.pi*(.25-rg*self.j**2)) #propagation term
        self.A_z = ifft(w*fft(self.A))
        self.I = np.abs(self.A_z)**2
  
    #%% Gaussian Lens Phase Screen
    def generateGaussianScreen(self):
        self.phase_screen = self.phi0*np.exp(-(self.x/self.l)**2)
        self.A = np.exp(1j*self.phase_screen)
        
    def generateSinScreen(self):
        self.phase_screen = self.phi0*np.sin(2*np.pi*self.x/self.l)
        self.A = np.exp(1j*self.phase_screen)
        
    #%% Random Phase Screen                    
    def generateRandomScreen(self):

        self.row = np.exp(-1*self.X**2)
        self.S_k = np.abs(fft(self.row)*self.dX/(2*np.pi))
        self.S_k *= 1/(self.dQ*np.sum(self.S_k))
        self.w_k = self.phi0*np.sqrt(24*np.pi*self.S_k/self.dX)
        self.P = np.random.rand(self.N)-.5
        self.Phat = np.fft.fft(self.P)
        self.phi_hat = self.w_k*self.Phat
        
        self.phase_screen = np.real((1/self.N)*np.fft.fft(self.phi_hat))#imag basically 0

    def calcStats(self):
        #Autocorrelation, variance of screen
        self.sigma_phi = np.var(self.phase_screen)
        m = np.mean(self.phase_screen)
        self.ro_phi = np.correlate(self.phase_screen-m,self.phase_screen-m,mode='full')[-self.N:]
        self.ro_phi/=(self.sigma_phi*np.arange(self.N,0,-1))
        m = np.mean(self.I)
        self.M2 = np.correlate(self.I-m,self.I-m,mode='full')[-self.N:]
        self.M2/=np.arange(self.N,0,-1)
        self.m = self.M2[0]
        
        self.PSD_phi = ifft(self.ro_phi)*self.sigma_phi
        self.q = fftfreq(self.N)
        self.PSD_I = ifft(self.M2)

    def plotmvsz(self):
        m=[]
        for zeta in np.arange(0,2,.01):
            self.propagate(zeta)
            m.append(np.sqrt((np.mean(self.I**2)-np.mean(self.I)**2)/np.mean(self.I)**2))
        plt.plot(np.arange(0,2,.01),m)
        plt.ylim([0,2.5])
  
    #%% Diffraction Movie
    def exportAVI(self,save_file='ani.mp4',frames=200):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im, = ax.plot(self.X,self.intensity[0])
        ax.set_ylim([np.min(self.intensity)-5,np.max(self.intensity)+5])
        fig.set_size_inches([8,8])
        def update_img(frame):
            im.set_ydata(self.intensity[frame])
            ax.set_title('Z = %.3f, z = %.2fm'%(self.Z[frame],self.z[frame]))
            return im
        ani = animation.FuncAnimation(fig,update_img,frames,interval=100)
        writer = animation.writers['ffmpeg'](fps=5)
    
        ani.save(save_file,writer=writer,dpi=200)
        return ani
        
    def plotDiff(self):
        
        plt.figure()
        plt.plot(self.x,self.I)
        plt.xlim([-self.l,self.l])
        plt.ylim([0,10])
    
    def plotScreen(self):
        plt.figure()
        plt.plot(self.X,self.phase_screen)
        
    def plotHistCont(self,zi):
        plt.figure()
        for i in range(len(zi)):
            values,xedges,yedges = np.histogram2d(np.real(self.field[zi[i]]),
                                                  np.imag(self.field[zi[i]]),10)
            x,y = np.meshgrid(xedges[:-1],yedges[:-1])
            a=np.ceil(len(zi)/2)
            plt.subplot(a,np.ceil(len(zi)/a),i+1)
            plt.title('Z=%.4f'%self.Z[zi[i]])
            plt.contourf(x,y,values)
            plt.colorbar()
            plt.grid()
        
if __name__ == "__main__":
    from IPython import get_ipython
    ipy = get_ipython()
    ipy.magic("matplotlib qt")
    sim = Simulation1D()
    sim.generateSinScreen()
    sim.propagate(.025)
    sim.calcStats()
   
