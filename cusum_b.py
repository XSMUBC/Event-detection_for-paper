print("The code is used for event detection by cusum method!")





import math
import numpy as np


#dset= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]



import numpy as np

filename = 'rea2.txt'
dset  = np.loadtxt(filename, delimiter=',')

















nl=10 #windows length
#print(",", dset[len(dset)-1])





for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxsum=0

	ss=1

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
					a=a+(math.exp(-pow(nums[l]-nums[k1],2)/2*pow(ss,2))/(ss*math.sqrt(2*math.pi)))
				a=a/(len(nums)-j+1)
				b=0
				for k2 in range(i,j-1):
					b=b+(math.exp(-pow(nums[l]-nums[k2],2)/pow(ss,2))/(ss*math.sqrt(2*math.pi)))
#				print("The b is:",b)
				b=0.025+b/(j-i)  # careful

#				print("The b is:",b)
#				print("The k,l is:",k,l)
#				if (k==j-1 and l==len(nums)-1):
#					c_ij=c_ij+abs(nums[l]-nums[k])

#					print("The num is:",c_ij)
#				else:
#				c_ij=c_ij+abs(nums[l]-nums[k])
				s_ij=s_ij+math.log2(a/b)
#		print("The num is:",s_ij)
			if (s_ij> maxsum):
				maxsum=s_ij
	print(",",maxsum)

#return maxmbgt


#if (maxsum>abs(nums[len(nums)-1]-nums[0])/len(nums)):
#	print("Event detected")
#else:
#	print("Event undetected")
#print "\nAfter taxes the price is: ", a + base














