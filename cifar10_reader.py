import numpy as np
import matplotlib.pyplot as plt
import os

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		dict = pickle.load(fo, encoding='bytes')
	return dict

def get_all_data():
	x,y,names = get_data('data_batch_1')
	for i in range(2,6):
		x_t,y_t,names_t = get_data('data_batch_'+str(i))
		x = np.concatenate((x,x_t),axis = 1)
		y = np.concatenate((y,y_t),axis = 1)
		names = np.concatenate((names,names_t),axis = 0)
	return x,y,names

def get_data(file):
	absFile = os.path.abspath("data/"+file)
	dict = unpickle(absFile)
	#for key in dict.keys():
	#	print(key)
	#print("Unpacking {}".format(dict[b'batch_label']))
	X = np.asarray(dict[b'data'].T).astype("uint8")
	Yraw = np.asarray(dict[b'labels'])
	Y = np.zeros((10,10000))
	for i in range(10000):
		Y[Yraw[i],i] = 1
	names = np.asarray(dict[b'filenames'])
	return X,Y,names

def visualize_image(X,Y,names,id):
	rgb = X[:,id]
	#print(rgb.shape)
	img = rgb.reshape(3,32,32).transpose([1, 2, 0])
	#print(img.shape)
	plt.imshow(img)
	plt.title(names[id])
	#print(Y[id])
	#plt.show()
	dir = os.path.abspath("output/samples")
#	try:
#		os.makedirs(dir)
#	except OSError as exception:
#		if exception.errno != errno.EEXIST:
#			raise
	plt.savefig(dir+"/"+names[id].decode('ascii'))
	#plt.savefig(dir+"/image.jpg",format='jpg')