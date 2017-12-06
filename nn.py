import numpy as np
import activations as af
import datetime
import sys
import matplotlib.pyplot as plt
import os


def log(str):
	tm = datetime.datetime.now().strftime("%I:%M:%S %p")
	print("{} -> {}".format(tm,str))

def init_parameters(layer_dims,parameters_file):
	if parameters_file == None:
		return init_parameters_new(layer_dims)
	log("Reusing the parameter from {}".format(parameters_file))
	parameters = np.load(parameters_file)
	log(type(parameters))
	log(parameters[()]['W1'].shape)
	return parameters[()]
def init_parameters_new(layer_dims):
	parameters = {}
	L = len(layer_dims)
	for l in range(1, L):
		parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l-1])
		parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
	return parameters

def perform_activation(activation,Z):
	return getattr(sys.modules["activations"],activation)(Z)
'''
	if activation == 'tanh':
		return af.tanh(Z)
	elif activation == 'sigmoid':
		return af.sigmoid(Z)
	elif activation == 'relu':
		return af.relu(Z)
'''
def perform_activation_backwards(activation,Z,A):
	return getattr(sys.modules["activations"],"inverse_"+activation)(Z,A)
'''
	s = perform_activation(activation,z)
	dZ = dA * s * (1-s)
	return dZ
'''

def update_grads(grads,parameters,alpha):
	L = len(parameters) // 2
	for l in range(1,L+1):
		parameters['W' + str(l)] = parameters['W' + str(l)] - alpha * grads['dW' + str(l)]
		parameters['b' + str(l)] = parameters['b' + str(l)] - alpha * grads['db' + str(l)]
	return parameters #have to check whether this is really reqd

def back_prop(m,A_values,Z_values,Y,activation,parameters):
	grads = {}
	#dAL = - (np.divide(Y, A_values[-1]) - np.divide(1 - Y, 1 - A_values[-1]))
	#dZ = perform_activation_backwards(dAL,A_values[-1],activation[1])
	dZ = A_values[-1] - Y
	L = len(A_values)-1
	for l in reversed(range(L)):
		grads['dW' + str(l + 1)] = (1 / m) * np.dot(dZ,A_values[l].T)
		grads['db' + str(l + 1)] = (1 / m) * np.sum(dZ,axis=1,keepdims=True)
		#log(" W{} : {}, dZ : {}, A_Values{} :{}".format(str(l+1),np.asarray(parameters['W' + str(l+1)]).shape,np.asarray(dZ).shape,l,np.asarray(A_values[l]).shape))
		if l != 0:
			dZ = np.dot(parameters['W' + str(l+1)].T,dZ)# * (1-np.power(A_values[l],2))
			dZ *= perform_activation_backwards(activation[0],Z_values[l-1],A_values[l])
		#dA_prev = np.dot(parameters['W' + str(l)].T,dZ)
		# dZ = perform_activation_backwards(  #dint quite understand yet on how the relu inverse is done
	return grads

def forward_prop(X,activation,parameters):
	A_values = []
	Z_values = []
	L = len(parameters) // 2
	A = X
	A_values.append(A)
	for l in range(1,L):
		Z = np.dot(parameters['W' + str(l)],A) + parameters['b' + str(l)]
		A = perform_activation(activation[0],Z)
		A_values.append(A)
		#if activation[0] == 'relu': #Only relu requires this as for others we will be able to do with A
		Z_values.append(Z)
	Z = np.dot(parameters['W' + str(L)],A) + parameters['b' + str(L)]
	A = perform_activation(activation[1],Z)
	A_values.append(A)
	#if activation[0] == 'relu':
	Z_values.append(Z)
	return A_values,Z_values

def validate (Y,Y1,m):
	succ = 0
	for i in range(m):
		if(np.sum(Y[:,i] == Y1[:,i]) == 10):
			succ+=1
	return succ/m

def predict(m,A2):
	Y = np.zeros((10, m))
	for i in range(m):
		max_val = 0
		max_val_id = 0
		for j in range(10):
			if A2[j,i] > max_val :
				max_val_id = j
				max_val = A2[j,i]
			#Y[j, i] = 1 if A2[j, i] > 0.5 else 0
		Y[max_val_id,i] = 1
	return Y

def get_batch(X,Y,m,X_current_batch,Y_current_batch,batch_size,batch_cursor,epoch):
	#log("in get_batch with batch size : {} and {} ".format(batch_size,batch_cursor))
	X_current_batch[:,0:batch_size] = X[:,batch_cursor:batch_cursor+batch_size]
	Y_current_batch[:,0:batch_size] = Y[:,batch_cursor:batch_cursor+batch_size]
	if batch_cursor + 2*batch_size >= m:
		batch_cursor = 0
		epoch+=1
		#log("epoch increased : {}".format(epoch))
	else:
		batch_cursor += batch_size
	return X_current_batch,Y_current_batch,batch_cursor,epoch

def compute_cost(y_hat,Y,m,train_cost,train_accu):
	logprobs = np.multiply(np.log(y_hat), Y) + np.multiply((1 - Y), np.log(1 - y_hat))
	cost = - np.sum(logprobs) / m
	train_cost.append(cost)
	Y2 = predict(m,y_hat)
	accu = validate(Y,Y2,m)
	train_accu.append(accu)
	return cost
	#log("cost : {} - {}, train accu : {}".format(index*50,cost,accu))

def compute_dev_set(X,Y,m,activation,parameters,dev_accu):
	A_values,Z_values = forward_prop(X,activation,parameters)
	Y2 = predict(m,A_values[-1])
	accu = validate(Y,Y2,m)
	dev_accu.append(accu)
	return accu

def model(X,Y,**kwargs):
	log("Entered model with {}".format(kwargs))
	log("X size : {}, Y size : {}".format(X.shape,Y.shape))
	
	x_n,m = X.shape
	y_n = len(Y)
	
	alpha = kwargs.get('alpha',0.01)
	iter = kwargs.get('iter',3000)
	layer_dims = kwargs.get('hidden_layer_dims',[])
	activation = kwargs.get('activation',['tanh','sigmoid'])
	batch_size = kwargs.get('batch_size',m)
	dev_set_ratio = kwargs.get('dev_set_ratio',0.02)
	parameters_file = kwargs.get('parameters_file',None)
	
	layer_dims.insert(0,x_n)
	layer_dims.insert(len(layer_dims),y_n)

	parameters = init_parameters(layer_dims,parameters_file)
	log(len(parameters))
	iterations_capture_freq = 50
	capture_frequency = 500
	accu = 0
	train_cost = []#np.zeros(int(iter/iterations_capture_freq))
	train_accu = []
	dev_accu = []
	batch_cursor = 0
	epoch = 0
	X_current_batch = np.zeros([x_n,batch_size])
	Y_current_batch = np.zeros([y_n,batch_size])
	m_dev = int(m*dev_set_ratio)
	m = m - m_dev
	X,X_dev = np.split(X,[m],axis=1)
	Y,Y_dev = np.split(Y,[m],axis=1)
	log("Post splitting of train and dev set, shape of train : {} , dev : {}".format(X.shape,X_dev.shape))
	print("Training the model, please wait")
	print("00.00%  cost: 00.0000  accu: 0.0000",end="")
	for i in range(iter):
		X_current_batch,Y_current_batch,batch_cursor,epoch = get_batch(X,Y,m,X_current_batch,Y_current_batch,batch_size,batch_cursor,epoch)
		A_values,Z_values = forward_prop(X_current_batch,activation,parameters)
		if(i%iterations_capture_freq==0):
			cost = compute_cost(A_values[-1],Y_current_batch,batch_size,train_cost,train_accu)
			if m_dev >0:
				accu = compute_dev_set(X_dev,Y_dev,m_dev,activation,parameters,dev_accu)
			print("\b"*35,end="")
			print("{:05.2f}%  cost: {:07.4f}  accu: {:06.4f}".format((i/iter*100),cost,accu),end="",flush=True)
				#log('dev acc : {}'.format(accu))
		grads = back_prop(batch_size,A_values,Z_values,Y_current_batch,activation,parameters)
		parameters = update_grads(grads,parameters,alpha)
		if i%capture_frequency == 0 and i!=0:
			snapshot(train_cost,train_accu,dev_accu,parameters,i)
	#plot_cost_graph(train_cost)
	print("")
	if m_dev >0:
		accu = compute_dev_set(X_dev,Y_dev,m_dev,activation,parameters,dev_accu)
	snapshot(train_cost,train_accu,dev_accu,parameters,i)
	log("Model ready with accuracy : {}".format(accu))

def snapshot(train_cost,train_accu,dev_accu,parameters,i):
	plt.clf()
	#dir = os.path.abspath("output/snapshots/"+str(i))
	dir = os.path.abspath("output/snapshots")
	os.makedirs(os.path.dirname(dir), exist_ok=True)
	np.save(os.path.join(dir, 'parameters'+str(i)),parameters)
	#cost graph
	plt.subplot(3,1,1)
	plt.grid(True)
	ay = plt.gca()
	ay.set_yscale('log')
	plt.plot(train_cost)
	plt.title("Cost graph")
	
	#train accu
	plt.subplot(3,1,2)
	plt.grid(True)
	#ay = plt.gca()
	#ay.set_yscale('log')
	plt.plot(train_accu)
	plt.title("Training accuracy")
	
	#dev accu
	plt.subplot(3,1,3)
	plt.grid(True)
	#ay = plt.gca()
	#ay.set_yscale('log')
	plt.plot(dev_accu)
	plt.title("Dev set accuracy")
	plt.savefig(dir+"/graph"+str(i)+".png")
	plt.close()

def plot_cost_graph(values):
	plt.plot(values)
	plt.grid(True)
	ay = plt.gca()
	ay.set_yscale('log')
	#plt.ylim([0,2])
	plt.show()