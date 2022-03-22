import numpy as np
import pickle as pkl
import scipy.misc as smp
from scipy import special

from matplotlib import pyplot as plt


# PA-I-1 write data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    file1 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment1/datasets/train_data_x.pkl', 'rb')
    file2 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment1/datasets/train_data_y.pkl', 'rb')
    file3 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment1/datasets/test_data_x.pkl', 'rb')
    file4 = open('/Users/subhadramokashe/BrandeisCoursework/deeplearning/assignment1/datasets/test_data_y.pkl', 'rb')
    trnx = pkl.load(file1,encoding='latin1')
    trny = pkl.load(file2,encoding='latin1')
    tstx = pkl.load(file3,encoding='latin1')
    tsty = pkl.load(file4,encoding='latin1')
    return(trnx,trny,tstx,tsty)

def sigmoid(x):
    x =  np.array(x, dtype=np.float128)
    y = 1.0 / (1.0 + np.exp(- x))
    #y = - special.logsumexp([0, -x])
    return y

train_data_x, train_data_y, test_data_x, test_data_y= load_dataset()
#print(train_data_x.shape)
#print(train_data_y.shape)
#print(test_data_x.shape)
#print(test_data_y.shape)


plt.imshow(train_data_x[1,:,:,:] )
plt.show()

#train_data_x_vec = train_data_x.reshape(train_data_x.shape[0], -1).T
#print(train_data_x_vec.shape)
#test_data_x_vec = test_data_x.reshape(test_data_x.shape[0], -1).T
#print(test_data_x_vec.shape)
#print(max(train_data_x_vec[:,1]))
#print(train_data_x[19,:,:,:])

# PA-II-1 write initialize_zero function for weight parameters w of dimension (dim, 1) and bias scalar b
# initialize pars with all zeros
def initialize_zero(dim):
    # input: dim -- size of w
    # returns: initialized w, b
    w = np.zeros((dim,1))
    b = 0.0
    return w,b

def initialize_norm(dim):
    # input: dim -- size of w
    # returns: initialized w, b
    w = np.random.randn(dim,1)
    b = np.random.randn()
    return w,b

# PA-II-2 write forward function for computing loss and gradient
def forward(w, b, x, y):
    # input:
    # w -- weight parameters
    # b -- bias parameter
    # x -- input data (data_size, number of data samples)
    # y -- label of data: 1 or 0
    #x =  np.array(x, dtype=np.float128)
    #y =  np.array(y, dtype=np.float128)
    #print(y)
    n = y.shape[1]
    h = sigmoid(np.dot((w.T), x) + b)# compute activation
    #A =  np.array(A, dtype=np.float128)
    class1_cost = -y*np.log(h)
    class2_cost = (1-y)*np.log(1.0000-h)
    #cost = -1/n * np.sum(y * np.log(h) + (1.0-y) * np.log(1.0-h))
    #h = sigmoid(np.dot(w.T,x)+b)  # transformed weighted input
    #loss = -np.sum(y*np.log(h)+(1-h)*np.log(1-h))/len(y) # np.mean takes summation and divides by the length of the vector
    dw = (1/n) * np.dot(x, (h-y).T)
    db = (1/n) * np.sum(h-y)
    cost = class1_cost - class2_cost
    #assert(dw.shape == w.shape)
    #assert(db.dtype == float)
    cost = cost.sum() / n
    cost = np.squeeze(cost)
    return dw,db,cost
    # return:
    # dw -- gradient of the objective wrt w 
    # db -- gradient of the objective wrt b 
    # loss -- negative log-likelihood loss for logistic regression



# PA-II-3 write optimize function for parameter update
def optimize(w, b, x, y, num_iter, learning_rate, print_loss = True):
    # input: w, b -- parameters, x -- data, y -- label of data, num_iter: number of total iterations
    # output: w, b -- optimized parameters, loss_all -- recored loss at different epoches
    loss_all = []
    for i in range(num_iter):
        dw,db, loss = forward(w,b,x,y)
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            loss_all.append(loss) #tracking loss aftr every 100th iteration
        if print_loss and i % 1000== 0:
            print ("Loss after iteration %i: %f" %(i, loss))
        parameters = {"w": w, "b": b}
    return parameters,loss_all



# PA-II-4 write prediction function
def predict(w, b, x):
    # input: x -- input data, w, b -- learned pars
    # return: y_predict -- prediction values (labels) of data
    y_predict = sigmoid(np.dot(w.T,x) +b)
    #print(y_predict.shape)
    for i in range(len(y_predict)):
        y_predict[0,i] = 1 if y_predict[0,i] >= .5 else 0
    return y_predict


#x,y = initialize_zero(3)
#print(x,y)

