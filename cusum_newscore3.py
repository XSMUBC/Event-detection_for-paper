print("The code is used for event detection by cusum method!")


import matplotlib.pyplot as plt

import math
import csv
import numpy as np
from numpy import ndarray
import matplotlib 
from scipy.ndimage import gaussian_filter
import scipy.ndimage as ndi










def bilateralFtr1D(y, sSpatial = 5, sIntensity = 5):
    '''
    version: 1.0
    author: akshay
    date: 10/03/2016
    
    The equation of the bilateral filter is
    
            (       dx ^ 2       )       (         dI ^2        )
    F = exp (- ----------------- ) * exp (- ------------------- )
            (  sigma_spatial ^ 2 )       (  sigma_Intensity ^ 2 )
        ~~~~~~~~~~~~~~~~~~~~~~~~~~
        This is a guassian filter!
        dx - The 'geometric' distance between the 'center pixel' and the pixel
         to sample
    dI - The difference between the intensity of the 'center pixel' and
         the pixel to sample
    sigma_spatial and sigma_Intesity are constants. Higher values mean
    that we 'tolerate more' higher value of the distances dx and dI.
    
    Dependencies: numpy, scipy.ndimage.gaussian_filter1d
    
    calc gaussian kernel size as: filterSize = (2 * radius) + 1; radius = floor (2 * sigma_spatial)
    y - input data
    '''

    # gaussian filter and parameters
    radius = np.floor (2 * sSpatial)
    filterSize = ((2 * radius) + 1)
    ftrArray = np.zeros (int(filterSize))
    ftrArray[int(radius)] = 1
    
    # Compute the Gaussian filter part of the Bilateral filter
    gauss = ndi.gaussian_filter1d(ftrArray, sSpatial)

    # 1d data dimensions
    width = y.size

    # 1d resulting data
    ret = np.zeros (width)

    for i in range(width):

        ## To prevent accessing values outside of the array
        # The left part of the lookup area, clamped to the boundary
        xmin = max (i - radius, 1);
        # How many columns were outside the image, on the left?
        dxmin = xmin - (i - radius);

        # The right part of the lookup area, clamped to the boundary
        xmax = min (i + radius, width);
        # How many columns were outside the image, on the right?
        dxmax = (i + radius) - xmax;

        # The actual range of the array we will look at
        area = y [int(xmin):int(xmax)]

        # The center position
        center = y [i]

        # The left expression in the bilateral filter equation
        # We take only the relevant parts of the matrix of the
        # Gaussian weights - we use dxmin, dxmax, dymin, dymax to
        # ignore the parts that are outside the image
        expS = gauss[(1+int(dxmin)):(int(filterSize)-int(dxmax))]

        # The right expression in the bilateral filter equation
        dy = y [int(xmin):int(xmax)] - y [i]
        dIsquare = (dy * dy)
        expI = np.exp (- dIsquare / (sIntensity * sIntensity))

        # The bilater filter (weights matrix)
        F = expI * expS

        # Normalized bilateral filter
        Fnormalized = F / sum(F)

        # Multiply the area by the filter
        tempY = y [int(xmin):int(xmax)] * Fnormalized

        # The resulting pixel is the sum of all the pixels in
        # the area, according to the weights of the filter
        # ret(i,j,R) = sum (tempR(:))
        ret[i] = sum (tempY)
    
    return ret



dset=np.zeros(300)


#  decline with time 
for t in range(0,100):
	dset[t]=2
#	print(",", dset[t])

for t in range(100,200):
	dset[t]=19


for t in range(200,300):
	dset[t]=46
'''

# increase with time
for t in range(0,100):
	dset[t]=46
#	print(",", dset[t])

for t in range(100,200):
	dset[t]=19


for t in range(200,300):
	dset[t]=2

'''

for t in range(0,len(dset)):
	dset[t]=dset[t]+2*math.sin(t*(math.pi)/5)
#	print(",", dset[t])


'''
# Gaussian noise
# Convert to linear Watt units
#target_noise_watts = 10** (target_noise_db / 10)
target_noise_watts = 5

# Generate noise samples
mean_noise = 0
noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(dset))

# Noise up the original signal (again) and plot
dset = dset + noise_volts
'''

'''

#1D random noise

randomnoise = np.random.normal(0,1,len(dset))



dset = dset + randomnoise*2

'''


#1-D  uniform and Laplace noise 



def fn_addnoise(data):
    i = len(data)
    # create 1D numpy data:
    npdata = np.asarray(data).reshape((i))
    # add uniform noise:
    u = npdata + np.random.uniform(size=npdata.shape)
    # add laplace noise:
    p = npdata + np.random.laplace(loc=0.0, scale=1.0, size=npdata.shape)

#    print npdata
#    print u
#    print p
    return p

dset = dset + fn_addnoise(dset)*0.9


















dset1=dset
dset2=dset
dset3=dset
dset4=dset



#filename = 'rea2.txt'
#dset  = np.loadtxt(filename, delimiter=',')



nl=int(len(dset)/60) #windows length
#print(",", dset[len(dset)-1])

#with open('tmp_file.txt', 'w') as f:
maxsum1=ndarray((len(dset),),int)

maxsum2=ndarray((len(dset),),int)

maxsum3=ndarray((len(dset),),int)
maxsum4=ndarray((len(dset),),int)
maxsum5=ndarray((len(dset),),int)
maxsum6=ndarray((len(dset),),int)



for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxsum1[m]=0


#print("The N is:",len(nums))
#print("The nums[0] is:",nums[0])
#print("The nums[N-1] is:",nums[len(nums)-1])

	for i in range(0,len(nums)-2):
#	print("The num is:",nums[i])
		for j in range(i+1,len(nums)-1):
#		print("The i,j is:",i,j)
			c_ij=0
			for k in range(i,j-1):
				for l in range(j,len(nums)-1):
#				print("The k,l is:",k,l)
#				if (k==j-1 and l==len(nums)-1):
#					c_ij=c_ij+abs(nums[l]-nums[k])

#					print("The num is:",c_ij)
#				else:
					c_ij=c_ij+abs(nums[l]-nums[k])
			c_ij=c_ij/(j-i)/(len(nums)-j+1)
#		print("The c_i,j is:",c_ij)
			if (c_ij> maxsum1[m]):
				maxsum1[m]=c_ij
#	print(",",maxsum)
#        csv.writer(f, delimiter=' ').writerows(maxsum)

#return maxmbgt
for m in range (len(dset)-nl,len(dset)):
	maxsum1[m]=maxsum1[len(dset)-nl-5]
















dset=dset2


for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxsum2[m]=0

	ss=nl*2

#print("The N is:",len(nums))
#print("The nums[0] is:",nums[0])
#print("The nums[N-1] is:",nums[len(nums)-1])

	for i in range(0,len(nums)-2):
#	print("The num is:",nums[i])
		for j in range(i+1,len(nums)-1):
#		print("The i,j is:",i,j)
			s_ij=0
			for l in range(j,len(nums)-1):
#				print("The b is:",nums[l]-nums[k2])
				a=0
				for k1 in range(j,len(nums)-1):
					a=a+math.exp(-pow(nums[l]-nums[k1],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
				a=a/(len(nums)-j+1)
				b=0
				for k2 in range(i,j-1):
					b=b+math.exp(-pow(nums[l]-nums[k2],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
#				print("The b is:",b)
				b=0.00001+b/(j-i)  # careful

#				print("The b is:",b)
#				print("The k,l is:",k,l)
#				if (k==j-1 and l==len(nums)-1):
#					c_ij=c_ij+abs(nums[l]-nums[k])

#					print("The num is:",c_ij)
#				else:
#				c_ij=c_ij+abs(nums[l]-nums[k])
				s_ij=s_ij+math.log(a/b)
#		print("The num is:",s_ij)
			if (s_ij> maxsum2[m]):
				maxsum2[m]=s_ij
#	print(",",maxsum)
#        csv.writer(f, delimiter=' ').writerows(maxsum)

#return maxmbgt
for m in range (len(dset)-nl,len(dset)):
	maxsum2[m]=maxsum2[len(dset)-nl-5]





dset=dset3


dset=bilateralFtr1D(dset, nl/2,  nl/2)


for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxsum3[m]=0

	ss=nl*2
	dd=nl

#print("The N is:",len(nums))
#print("The nums[0] is:",nums[0])
#print("The nums[N-1] is:",nums[len(nums)-1])

	for i in range(0,len(nums)-2):
#	print("The num is:",nums[i])
		for j in range(i+1,len(nums)-1):
#		print("The i,j is:",i,j)
			s_ij=0
			for l in range(j,len(nums)-1):
#				print("The b is:",nums[l]-nums[k2])
				a=0
				for k1 in range(j,len(nums)-1):
					a=a+math.exp(-pow(nums[l]-nums[k1],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))#/math.exp(-pow(l-k1,2))
				a=a/(len(nums)-j+1)
				b=0
				for k2 in range(i,j-1):
					b=b+math.exp(-pow(nums[l]-nums[k2],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))#/math.exp(-pow(l-k2,2))
#				print("The b is:",b)
				b=0.00001+b/(j-i)  # careful

#				print("The b is:",b)
#				print("The k,l is:",k,l)
#				if (k==j-1 and l==len(nums)-1):
#					c_ij=c_ij+abs(nums[l]-nums[k])

#					print("The num is:",c_ij)
#				else:
#				c_ij=c_ij+abs(nums[l]-nums[k])
				s_ij=s_ij+math.log(a/b)
#		print("The num is:",s_ij)
			if (s_ij> maxsum3[m]):
				maxsum3[m]=s_ij
#	print(",",maxsum)
#        csv.writer(f, delimiter=' ').writerows(maxsum)

#return maxmbgt
for m in range (len(dset)-nl,len(dset)):
	maxsum3[m]=maxsum3[len(dset)-nl-5]

#if (maxsum>abs(nums[len(nums)-1]-nums[0])/len(nums)):
#	print("Event detected")
#else:
#	print("Event undetected")
#print "\nAfter taxes the price is: ", a + base






dset=dset4

#dset=bilateralFtr1D(dset, nl/2,  nl/2)


for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxsum4[m]=0

	ss=nl*2
	dd=nl

#print("The N is:",len(nums))
#print("The nums[0] is:",nums[0])
#print("The nums[N-1] is:",nums[len(nums)-1])

	for i in range(0,len(nums)-2):
#	print("The num is:",nums[i])
		for j in range(i+1,len(nums)-1):
#		print("The i,j is:",i,j)
			s_ij=0
			for l in range(j,len(nums)-1):
#				print("The b is:",nums[l]-nums[k2])
				a=0
				for k1 in range(j,len(nums)-1):
					#a=a+math.exp(-pow(l-k1,2)/2/pow(ss,2))*math.exp(-pow(nums[l]-nums[k1],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))/(dd*math.sqrt(2*math.pi))
					aa1=math.exp(-pow(nums[l]-nums[k1],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
					aa2=math.exp(-pow(l-k1,2)/2/pow(dd,2))/(dd*math.sqrt(2*math.pi))
					a=a+aa1/aa2
				a=a/(len(nums)-j+1)
				b=0
				for k2 in range(i,j-1):
					#b=b+math.exp(-pow(l-k2,2)/2/pow(ss,2))*math.exp(-pow(nums[l]-nums[k2],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))/(dd*math.sqrt(2*math.pi))
					bb1=math.exp(-pow(nums[l]-nums[k2],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
					bb2=math.exp(-pow(l-k2,2)/2/pow(dd,2))/(dd*math.sqrt(2*math.pi))
					b=b+bb1/bb2
#				print("The b is:",b)
				b=0.00001+b/(j-i)  # careful

#				print("The b is:",b)
#				print("The k,l is:",k,l)
#				if (k==j-1 and l==len(nums)-1):
#					c_ij=c_ij+abs(nums[l]-nums[k])

#					print("The num is:",c_ij)
#				else:
#				c_ij=c_ij+abs(nums[l]-nums[k])
				s_ij=s_ij+math.log(a/b)
#		print("The num is:",s_ij)
			if (s_ij> maxsum4[m]):
				maxsum4[m]=s_ij
#	print(",",maxsum)
#        csv.writer(f, delimiter=' ').writerows(maxsum)

#return maxmbgt
for m in range (len(dset)-nl,len(dset)):
	maxsum4[m]=maxsum4[len(dset)-nl-5]








dset=dset4

dset=bilateralFtr1D(dset, nl/2,  nl/2)


for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxsum5[m]=0

	ss=nl*2
	dd=nl

#print("The N is:",len(nums))
#print("The nums[0] is:",nums[0])
#print("The nums[N-1] is:",nums[len(nums)-1])

	for i in range(0,len(nums)-2):
#	print("The num is:",nums[i])
		for j in range(i+1,len(nums)-1):
#		print("The i,j is:",i,j)
			s_ij=0
			for l in range(j,len(nums)-1):
#				print("The b is:",nums[l]-nums[k2])
				a=0
				for k1 in range(j,len(nums)-1):
					#a=a+math.exp(-pow(l-k1,2)/2/pow(ss,2))*math.exp(-pow(nums[l]-nums[k1],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))/(dd*math.sqrt(2*math.pi))
					aa1=math.exp(-pow(nums[l]-nums[k1],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
					aa2=math.exp(-pow(l-k1,2)/2/pow(dd,2))/(dd*math.sqrt(2*math.pi))
					a=a+aa1/aa2
				a=a/(len(nums)-j+1)
				b=0
				for k2 in range(i,j-1):
					#b=b+math.exp(-pow(l-k2,2)/2/pow(ss,2))*math.exp(-pow(nums[l]-nums[k2],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))/(dd*math.sqrt(2*math.pi))
					bb1=math.exp(-pow(nums[l]-nums[k2],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
					bb2=math.exp(-pow(l-k2,2)/2/pow(dd,2))/(dd*math.sqrt(2*math.pi))
					b=b+bb1/bb2
#				print("The b is:",b)
				b=0.00001+b/(j-i)  # careful

#				print("The b is:",b)
#				print("The k,l is:",k,l)
#				if (k==j-1 and l==len(nums)-1):
#					c_ij=c_ij+abs(nums[l]-nums[k])

#					print("The num is:",c_ij)
#				else:
#				c_ij=c_ij+abs(nums[l]-nums[k])
				s_ij=s_ij+math.log(a/b)
#		print("The num is:",s_ij)
			if (s_ij> maxsum5[m]):
				maxsum5[m]=s_ij
#	print(",",maxsum)
#        csv.writer(f, delimiter=' ').writerows(maxsum)

#return maxmbgt
for m in range (len(dset)-nl,len(dset)):
	maxsum5[m]=maxsum5[len(dset)-nl-5]



#sep method 
dset=dset4

sep=10
ss=5
lam=0.9
#0.8
#dset=bilateralFtr1D(dset, nl/2,  nl/2)

for m in range(len(dset)-2*sep):

    nums1=dset[m:m+sep-1]
    nums2=dset[m+sep:m+2*sep-1]

    maxsum6[m]=0

    '''
    sepc=ndarray((len(nums1),),int)



    for l in range(len(nums1)):
        ba=0
        for i in range(l):
            a=0
    #	print("The num is:",nums[i])
            for j in range(len(nums2)):
    #		print("The i,j is:",i,j)
			    #s_ij=0

                aa1=math.exp(-pow(nums1[i]-nums2[j],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
                a=a+aa1
            ba=ba+a
            
        sepc[l]=ba
        
    '''
    #abc=0

    
    ba=0
    for i in range(len(nums1)):
        a=0
#	print("The num is:",nums[i])
        for j in range(len(nums2)):
#		print("The i,j is:",i,j)
            #s_ij=0

            aa1=math.exp(-pow(nums1[i]-nums2[j],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
            a=a+aa1
        ba=ba+a
    
    
    

#print("The N is:",len(nums))
#print("The nums[0] is:",nums[0])
#print("The nums[N-1] is:",nums[len(nums)-1])


    sepg=ndarray((len(nums1),),int)
    #b=0\
    for l in range(len(nums1)):
        b=0
        for i in range(l):
    #	print("The num is:",nums[i])
            a=0
            for j in range(len(nums2)):
    #		print("The i,j is:",i,j)
	            #s_ij=0

                aa1=math.exp(-pow(nums1[i]-nums2[j],2)/2/pow(ss,2))/(ss*math.sqrt(2*math.pi))
                a=a+aa1

            b=b+a*(-1/lam)*(1/sep)*ba


        sepg[l]=b
        print("g(x)",b)


    if (0.5-ndarray.mean(sepg)>=0):
        maxsum6[m]=0.5-ndarray.mean(sepg)
    else:

        maxsum6[m]=0
       
      

#	print(",",maxsum)
#        csv.writer(f, delimiter=' ').writerows(maxsum)

#return maxmbgt


for m in range (len(dset)-2*sep,len(dset)):
	maxsum6[m]=maxsum6[len(dset)-2*sep-1]

maxsum6=maxsum6[::-1]
for m in range(1,len(dset)-1):
    if (maxsum6[m]>=maxsum6[m-1] and maxsum6[m+1]>=maxsum6[m]):
#        maxsum6[m]=0
        maxsum6[m-1]=0

        
        

for m in range (len(dset)-2*sep,len(dset)):
	maxsum6[m]=maxsum6[len(dset)-2*sep-1]






























#with open('tmp_file.txt', 'w') as f:
#    csv.writer(f, delimiter=' ').writerows(data)

plt.figure(1)
#fig = plt.figure()
fs=30
matplotlib.rc('axes', linewidth=5)
matplotlib.rc('xtick', labelsize=fs) 
matplotlib.rc('ytick', labelsize=fs) 


plt.tight_layout()
#fig.tight_layout(h_pad=5, w_pad=5)
plt.subplots_adjust(top = 0.97, bottom=0.01, hspace=0.9, wspace=0.1)
'''
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.title.set_text('First Plot')
ax2.title.set_text('Second Plot')
ax3.title.set_text('Third Plot')
ax4.title.set_text('Fourth Plot')
'''
plt.subplot(711)
plt.plot(np.arange(1, 301, 1),dset1, 'b',linewidth=5.0)
plt.gca().set_title('(a)', fontsize=fs,loc='left')
plt.xlabel('Time', fontsize=fs)
plt.ylabel('Amp', fontsize=fs)

plt.subplot(712)
plt.plot(np.arange(1, 301, 1),maxsum1, 'r',linewidth=5.0)
#plt.scatter(np.arange(1, 301, 1),maxsum)
plt.gca().set_title('(b)', fontsize=fs,loc='left')
#plt.xlabel('Time', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.xticks([])

plt.subplot(713)
plt.plot(np.arange(1, 301, 1),15-maxsum2, 'r',linewidth=5.0)
plt.gca().set_title('(c)', fontsize=fs,loc='left')
#plt.xlabel('Time', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.xticks([])

plt.subplot(714)
plt.plot(np.arange(1, 301, 1),15-maxsum3, 'r',linewidth=5.0)
plt.gca().set_title('(d)', fontsize=fs,loc='left')
#plt.xlabel('Time', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.xticks([])
#plt.scatter(np.arange(1, 301, 1),maxsum)


plt.subplot(715)
plt.plot(np.arange(1, 301, 1),20-maxsum4, 'r',linewidth=5.0)
plt.gca().set_title('(e)', fontsize=fs,loc='left')
#plt.xlabel('Time', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.xticks([])

plt.subplot(716)
plt.plot(np.arange(1, 301, 1),20-maxsum5, 'r',linewidth=5.0)
plt.gca().set_title('(f)', fontsize=fs,loc='left')
#plt.xlabel('Time', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.xticks([])



plt.subplot(717)
plt.plot(np.arange(1, 301, 1),maxsum6, 'r',linewidth=5.0)
plt.gca().set_title('(g)', fontsize=fs,loc='left')
#plt.xlabel('Time', fontsize=fs)
plt.ylabel('Score', fontsize=fs)
plt.xticks([])











plt.show()











'''



plt.figure(2)






matplotlib.rc('axes', linewidth=5)
matplotlib.rc('xtick', labelsize=25) 
matplotlib.rc('ytick', labelsize=25) 

plt.tight_layout()

plt.subplot(311)
plt.plot(np.arange(1, 301, 1),dset1, 'b',linewidth=5.0)
plt.gca().set_title('Raw data', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Amp', fontsize=25)

plt.subplot(312)
plt.plot(np.arange(1, 301, 1),maxsum1, 'r',linewidth=5.0)
#plt.scatter(np.arange(1, 301, 1),maxsum)
plt.gca().set_title('Abrupt change score before applying threshold', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Score', fontsize=25)




threshold=max(maxsum1)/3


for m in range (0,len(maxsum1)):
	if maxsum1[m]<threshold:
		maxsum1[m]=0
	








plt.subplot(313)
plt.plot(np.arange(1, 301, 1),maxsum1, 'r',linewidth=5.0)
plt.gca().set_title('Abrupt change score-CUSUM after  applying threshold', fontsize=25)
plt.xlabel('Time', fontsize=25)
plt.ylabel('Score', fontsize=25)















plt.show()


'''




#fig.savefig('Basic.png', dpi=300)





