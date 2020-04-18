print("The code is used for event detection by mb-gt method!")



import math
import numpy as np







#dset= [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]

#dset= [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]

#dset=np.array(dset)*10





dset=np.zeros(300)


for t in range(0,100):
	dset[t]=16
#	print(",", dset[t])

for t in range(100,200):
	dset[t]=9


for t in range(200,300):
	dset[t]=2

#dset= [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,2, 2, 2, 2, 2, 2, 2, 2, 2, 2 ]






for t in range(0,len(dset)):
	dset[t]=dset[t]+0.5*math.sin(t*(math.pi)/6)
	print(",", dset[t])


print("here" )



















nl=10 #windows length
#print(",", dset[len(dset)-1])





for m in range (0,len(dset)-nl):

	nums=dset[m:m+nl-1]


	maxmbgt=0


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
			if (c_ij> maxmbgt):
				maxmbgt=c_ij
	print(",", maxmbgt)



#return maxmbgt


#if (maxmbgt>abs(nums[len(nums)-1]-nums[0])/len(nums)):#
#	print("Event detected")
#else:
#	print("Event undetected")
#print "\nAfter taxes the price is: ", a + base














