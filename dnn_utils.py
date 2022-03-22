import numpy as np
import pickle as pkl


# PA-I-1 write data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
        file1 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/datasets/train_data_x.pkl', 'rb')
        file2 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/datasets/train_data_y.pkl', 'rb')
        file3 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/datasets/test_data_x.pkl', 'rb')
        file4 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/datasets/test_data_y.pkl', 'rb')
        trnx = pkl.load(file1,encoding='latin1')
        trny = pkl.load(file2,encoding='latin1')
        tstx = pkl.load(file3,encoding='latin1')
        tsty = pkl.load(file4,encoding='latin1')
        return(trnx,trny,tstx,tsty)



def sigmoid(x):
	y = 1.0 / (1 + np.exp(-x))
	
	return y, x


def relu(x):
	y = np.maximum(0, x)
	
	return y, x


def relu_backward(dy, x):
	dx = np.array(dy, copy = True) 
	
	dx[x <= 0] = 0
	
	return dx


def sigmoid_backward(dy, x):
	s = 1.0 / (1 + np.exp(-x))
	dx = dy * s * (1 - s)
	
	return dx


# PB-I-1 write initialize_pars_deep function for weight parameters w and bias scalar b
# w are initialized with standard normal function * 0.1, b are initialized with 0 
def initialize_pars_deep(layer_dims):
        #input: layer_dims --  the dimensions of each layer in our network
        # returns: parameters -- w1, b1, ..., wL, bL:
        #          wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        #          bl -- bias vector of shape (layer_dims[l], 1)
        np.random.seed(3) # fix seed
        L = len(layer_dims) #finding out the number of layers
        #print(L)
        parameters = {} # creating a dictionary as the dimensions change with each layer
        parameters['w' + str(0)] = np.random.randn(layer_dims[0], 12288) * 0.01 # first set of weights w0 have N1 x 12288 dimensions
        for i in range(1,L):
                parameters['w' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
                parameters['b' + str(i)] =  np.zeros((layer_dims[i], 1))
        return parameters

layers_dim = [12288, 32, 16, 1]
#print(layers_dim)
param = initialize_pars_deep(layers_dim)
#print(len(param))


# linear forward function
def linear_forward(x, w, b):
	y = np.dot(w, x) + b
	
	cache = (x, w, b)
	
	return y, cache


# PB-I-2 write linear_activation_forward function for computing activations of each layer
def linear_activation_forward(x, w, b, activation):
	# input:
	# x -- activations from previous layer (or input data)
	# w -- weights matrix, numpy array of shape (size of current layer, size of previous layer)
	# b -- bias vector, numpy array of shape (size of the current layer, 1)
	# activation: activation function: sigmoid or relu

	# return:
	# y -- the output of the activation function
	# cache -- tuple of (linear_cache, activation_cache), linear_cache and activation_cache: the second output of linear_forward and sigmoid/relu
	# cache is stored for computing the backward
        if activation == "sigmoid":
                Z, linear_cache  = linear_forward(x, w, b)
                A, activation_cache = sigmoid(Z)
        elif activation == "relu":
                Z, linear_cache = linear_forward(x, w, b)
                A, activation_cache = relu(Z)
        cache = (linear_cache, activation_cache)
        return A, cache



# L_model_forward function for computing forward of each layer
# hidden layer activation: relu, output layer activation: sigmoid
# use linear_activation_forward function
def L_model_forward(x, pars):
	# input: x -- input data, pars -- initialized pars
	# return: xL -- output values, caches -- stored values (the second output of linear_activation_forward) of all layers 
	caches = []
	L = len(pars) // 2 # getting the depth ceiling division by 2 as both w and b are present in the pars
	y = x
	for l in range(1, L):
		x_prev = y
		#print(x_prev.shape)
		y, cache = linear_activation_forward(x_prev, pars['w' + str(l)], pars['b' + str(l)], "relu")
		caches.append(cache)
	kk = L
	xL, cache = linear_activation_forward(y, pars['w' + str(kk)], pars['b' + str(kk)], "sigmoid")
	caches.append(cache)
	#print(caches[1])
			
	return xL, caches


# PB-I-3 write compute_loss function for computing loss
def compute_loss(y_hat, y):
	# input: y_hat -- prediction value, y -- groundtruth value of shape (1, number of data samples)
	# return: loss -- loss value
	m = y.shape[1]
	loss = np.multiply(np.log(y_hat),y) +  np.multiply(np.log(1-y_hat), (1-y))
	losst = -1/m*np.sum(loss)
	return losst


def linear_backward(dy, cache):
	x_prev, w, b = cache
	num = x_prev.shape[1]

	dw = 1.0 / num * np.dot(dy, x_prev.T)
	db =  1.0 / num * (np.sum(dy, axis = 1, keepdims = True))
	dy_prev = np.dot(w.T, dy)
	
	return dy_prev, dw, db


# linear_activation_backward function for computing backward gradients
# use relu_backward/sigmoid_backward and linear_backward functions
def linear_activation_backward(dy, cache, activation):
	# input:
	# dy -- post-activation gradient for current layer l 
	# cache -- tuple of (linear_cache, activation_cache) for computing backward propagation 
	# activation -- the activation to be used in this layer: "sigmoid" or "relu"
	
	# return:
	# dy_prev -- gradient of the objective wrt the activation of the previous layer l-1
	# dw -- gradient of the objective wrt w of current layer l
	# db -- gradient of the objective wrt b of current layer l

	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dx = relu_backward(dy, activation_cache)
		dy_prev, dw, db = linear_backward(dx, linear_cache)
		
	elif activation == "sigmoid":
		dx = sigmoid_backward(dy, activation_cache)
		dy_prev, dw, db = linear_backward(dx, linear_cache)
	
	return dy_prev, dw, db




# PB-I-4 write L_model_backward function for computing backward of each layer
# use linear_activation_backward function
def L_model_backward(xL, y, caches):
	# input:
	# xL -- prediction values
	# y -- groundtruth values of all data samples
	# caches -- stored values of all layers: the second output of L_model_forward

	# return: grads -- gradients of the objective wrt activation and parameters (dy, dw, db) of all layers 
        grads = {}
        L = len(caches) 
        m = xL.shape[1]
        y = y.reshape(xL.shape) 
        dxL = - (np.divide(y, xL) - np.divide(1 - y, 1 - xL))
        #print(caches)
        current_cache = caches[L-1]
        grads["dx" + str(L)], grads["dw" + str(L)], grads["db" + str(L)] = linear_activation_backward(dxL, current_cache, activation = "sigmoid")
    
        for l in reversed(range(L-1)):
                current_cache = caches[l]
                dx_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dx" + str(l + 2)],  current_cache, activation = "relu")
                grads["dx" + str(l + 1)] = dx_prev_temp
                grads["dw" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp
        return grads


def update_pars(pars, grads, learning_rate):	
	L = len(pars) // 2
	for i in range(1, L + 1):
		pars["w" + str(i)] = pars["w" + str(i)] - learning_rate * grads["dw"+str(i)]
		pars["b" + str(i)] = pars["b" + str(i)] - learning_rate * grads["db"+str(i)]
		
	return pars


# PB-I-5 write prediction function for model prediction
def predict(pars, x):
	# input: x -- input data, pars -- learned pars
	# return: y_predict -- prediction values of data samples 
        y_predict,cache = L_model_forward(x,pars)
        for i in range(len(y_predict)):
                y_predict[0,i] = 1 if y_predict[0,i] >= .5 else 0
        return y_predict
        
