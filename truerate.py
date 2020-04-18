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


dset1= [0.98, 0.982, 0.984, 0.989, 0.99, 1, 1, 1, 1,1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dset2= [0.84, 0.856, 0.88, 0.90, 0.91, 0.923, 0.932, 0.94, 0.958,0.97, 0.99, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dset3= [0.76, 0.81, 0.85, 0.88, 0.90, 0.91, 0.923, 0.934, 0.94,0.943, 0.950, 0.952, 0.958, 0.964, 0.976, 0.978, 0.98, 0.99, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
dset4= [0.51, 0.60, 0.71, 0.78, 0.80, 0.802, 0.812, 0.818, 0.822,0.825, 0.831, 0.832, 0.855, 0.864, 0.872, 0.883, 0.891, 0.905, 0.91, 0.92, 0.939, 0.947, 0.952, 0.968, 0.973, 0.98, 0.98, 0.99, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

plt.figure(1)
#fig = plt.figure()
matplotlib.rc('axes', linewidth=10)
matplotlib.rc('xtick', labelsize=70) 
matplotlib.rc('ytick', labelsize=70) 

plt.ylim(0, 1.2)

plt.plot(np.arange(0, 1.0, 0.025),dset1, 'b',linewidth=10.0, label='EPCUSUM')
plt.plot(np.arange(0, 1.0, 0.025),dset2, 'y',linewidth=10.0, label='BF+MBCUSUM')
plt.plot(np.arange(0, 1.0, 0.025),dset3, 'g',linewidth=10.0, label='MB-CUSUM')
plt.plot(np.arange(0, 1.0, 0.025),dset4, 'r',linewidth=10.0, label='MB-GT')
pylab.legend(loc='lower right', fontsize=40)



x1 = [0, 0]
y1 = [0, 0.98] 
y2 = [0, 0.84] 
y3 = [0, 0.76] 
y4 = [0, 0.51]    

plt.plot(x1,y1, 'b',linewidth=10.0)
plt.plot(x1,y2, 'y',linewidth=10.0)
plt.plot(x1,y3, 'g',linewidth=10.0)
plt.plot(x1,y4, 'r',linewidth=10.0)




plt.gca().set_title('Runs=100,Len=300,Window=5,Dim=1', fontsize=70)
plt.xlabel('False positive rate', fontsize=70)
plt.ylabel('True positive rate', fontsize=70)
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


