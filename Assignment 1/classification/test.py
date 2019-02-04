import cv2
import  numpy as np
import os

shapes= ['circle', 'square', 'star', 'triangle']
f = open("test.txt", "w")
for shape in shapes:
	f.write(shape+"\n")
	directory = 'data/D2/images/' + shape + '/'
	num_black = 0
	num_files = 0
	num_d = 0
	corners=0
	for filename in os.listdir(directory):
		num_files+=1
		im = cv2.imread(directory + filename, 0)
		im = im/255
		im = im.astype(int)
		d = (np.delete((np.delete(im, 49, 0) - np.delete(im, 0, 0))**2, 49 ,1) + np.delete((np.delete(im, 49, 1) - np.delete(im, 0, 1))**2, 49 ,0))
		# print(d.size)
		# for col in d:
		# 	for x in col:
		# 		f.write(str(int(x))+ " ")
		# 	f.write("\n")
		# cv2.imshow(d.tolist(), 0)
		num_d += sum(d.flatten() >= 1)
		num_black+=sum(im.flatten()<0.5)

		window=4
		for i in range(window, 50-window):
		    for j in range(window, 50-window):
		        if(im[i,j]==0 and (im[i-1,j]==1 or im[i+1,j]==1 or im[i,j-1]==1 or im[i,j+1]==1)):
		            im1= im[i-window:i+window+1, j-window:j+window+1]
		            if(1.0*sum(im1.flatten() < 0.5 )/((2*window+1)**2) < 0.35):
		                corners+=1
		# for i in range(1, 50-1):
		#     for j in range(1, 50-1):
		#     	x = im[i-1,j]+im[i+1,j]+im[i,j-1]+im[i,j+1]
		#         if(im[i,j]==0 and x<=3 and x>=2):
		#             corners+=1


	print(shape, num_d/num_files, num_black/num_files, corners/num_files)

