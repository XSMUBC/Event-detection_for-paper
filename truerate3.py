print("The code is used for event detection by cusum method!")


import matplotlib.pyplot as plt

import math
import csv
import numpy as np
from numpy import ndarray
import matplotlib 
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi
import pylab 
from scipy.interpolate import make_interp_spline, BSpline
#from scipy.interpolate import spline
from matplotlib.ticker import MaxNLocator



dset1= [0.89,0.90, 0.95,0.96, 0.97,0.98, 0.99,0.99 ,1.00,1.00 ,1.00,1.00, 1.00,1.00, 1.00,1.00, 1.00,1.00,1.00,1.00]
dset2= [0.80,0.82, 0.90,0.93, 0.95,0.96, 0.97,0.98, 0.99,0.99, 1.00,1.00, 1.00,1.00, 1.00,1.00, 1.00,1.00,1.00,1.00]
dset3= [0.68,0.74 ,0.85,0.87 ,0.89,0.91, 0.92,0.93, 0.94,0.95 ,0.96,0.98 ,1.00,1.00, 1.00, 1.00,1.00,1.00,1.00,1.00]
dset4= [0.51,0.63, 0.70,0.75, 0.81, 0.82,0.83,0.85, 0.87,0.88, 0.90,0.94, 0.98,0.99 ,1.00,1.00, 1.00,1.00,1.00,1.00]
dset5= [0.70,0.77 ,0.87,0.90 ,0.91,0.93, 0.94,0.95, 0.96,0.97 ,0.98,0.99 ,1.00,1.00, 1.00, 1.00,1.00,1.00,1.00,1.00]
dset6= [0.89,0.92, 0.96,0.97, 0.98,0.99, 0.99,0.99 ,1.00,1.00 ,1.00,1.00, 1.00,1.00, 1.00,1.00, 1.00,1.00,1.00,1.00]

plt.figure(1)
#fig = plt.figure()
matplotlib.rc('axes', linewidth=10)
matplotlib.rc('xtick', labelsize=40) 
matplotlib.rc('ytick', labelsize=40) 

plt.ylim(0, 1)
plt.xlim(0, 1)


plt.yticks((0, 0.5, 1))
plt.tick_params(direction='out', pad=20)
plt.xticks((0, 0.5, 1))

'''
plt.xaxis.set_major_locator(MaxNLocator(5, prune='lower'))
plt.yaxis.set_major_locator(MaxNLocator(4))
'''
xnew = np.linspace(0,1,800) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(np.arange(0, 1.0, 0.05),dset1, k=1) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth, ':r',linewidth=10.0, label='EPCUSUM')


xnew = np.linspace(0,1,800) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(np.arange(0, 1.0, 0.05),dset2, k=1) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth, '--y',linewidth=10.0, label='BF+MBCUSUM')


xnew = np.linspace(0,1,800) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(np.arange(0, 1.0, 0.05),dset3, k=1) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth, ':b',linewidth=10.0, label='CUSUM')



xnew = np.linspace(0,1,800) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(np.arange(0, 1.0, 0.05),dset4, k=1) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth, '-g',linewidth=10.0, label='MB-GT')



xnew = np.linspace(0,1,800) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(np.arange(0, 1.0, 0.05),dset5, k=1) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth, '--h',linewidth=10.0, label='SEP')



xnew = np.linspace(0,1,800) #300 represents number of points to make between T.min and T.max
spl = make_interp_spline(np.arange(0, 1.0, 0.05),dset6, k=1) #BSpline object
power_smooth = spl(xnew)
plt.plot(xnew,power_smooth, '--b',linewidth=10.0, label='BF+EPCUSUM')


pylab.legend(loc='lower right', fontsize=40)


'''

x1 = [0, 0]
y1 = [0, 0.98] 
y2 = [0, 0.84] 
y3 = [0, 0.76] 
y4 = [0, 0.51]    

plt.plot(x1,y1, 'b',linewidth=10.0)
plt.plot(x1,y2, 'y',linewidth=10.0)
plt.plot(x1,y3, 'g',linewidth=10.0)
plt.plot(x1,y4, 'r',linewidth=10.0)


'''

#plt.gca().set_title('Runs=100,Len=300,Window=5,Dim=1', fontsize=70)
plt.xlabel('False positive rate', fontsize=40)
plt.ylabel('True positive rate', fontsize=40)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.show(1)


'''

plt.figure(2)
#fig = plt.figure()
matplotlib.rc('axes', linewidth=4)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 



plt.plot(np.arange(0, 1.0, 0.025),dset1, 'b',linewidth=4.0, label='MB-GT')
plt.plot(np.arange(0, 1.0, 0.025),dset2, 'y',linewidth=4.0, label='MB-CUSUM')
plt.plot(np.arange(0, 1.0, 0.025),dset3, 'g',linewidth=4.0, label='BF+MBCUSUM')
plt.plot(np.arange(0, 1.0, 0.025),dset4, 'r',linewidth=4.0, label='EPCUSUM')
pylab.legend(loc='lower right', fontsize=25)
plt.gca().set_title('Runs=100,Len=300,Window=5,Dim=1', fontsize=25)
plt.xlabel('False positive rate', fontsize=25)
plt.ylabel('True positive rate', fontsize=25)



plt.show(2)

'''

'''
#with open('tmp_file.txt', 'w') as f:
#    csv.writer(f, delimiter=' ').writerows(data)

plt.figure()
#fig = plt.figure()
matplotlib.rc('axes', linewidth=4)
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

plt.tight_layout()

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.title.set_text('First Plot')
ax2.title.set_text('Second Plot')
ax3.title.set_text('Third Plot')
ax4.title.set_text('Fourth Plot')

plt.subplot(511)
plt.plot(np.arange(1, 301, 1),dset1, 'b',linewidth=4.0)
plt.gca().set_title('Raw data', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Amp', fontsize=25)

plt.subplot(512)
plt.plot(np.arange(1, 301, 1),maxsum1, 'r',linewidth=4.0)
#plt.scatter(np.arange(1, 301, 1),maxsum)
plt.gca().set_title('Abrupt change score-MBGT', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Score', fontsize=25)

plt.subplot(513)
plt.plot(np.arange(1, 301, 1),16-maxsum2, 'r',linewidth=4.0)
plt.gca().set_title('Abrupt change score-CUSUM', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Score', fontsize=25)

plt.subplot(514)
plt.plot(np.arange(1, 301, 1),16-maxsum3, 'r',linewidth=4.0)
plt.gca().set_title('Abrupt change score-BF+CUSUM', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Score', fontsize=25)
#plt.scatter(np.arange(1, 301, 1),maxsum)


plt.subplot(515)
plt.plot(np.arange(1, 301, 1),16-maxsum4, 'r',linewidth=4.0)
plt.gca().set_title('Abrupt change score-EPCUSUM', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Score', fontsize=25)

plt.show()


#fig.savefig('Basic.png', dpi=300)


'''


