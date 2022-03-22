import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle as pkl
import dnn_utils as U


# PB-I-6 write model function 
def model(x_train, y_train, x_test, y_test, layers_dims, learning_rate, num_iter = 3000, print_loss = True):
        np.random.seed(1)
        loss_all = []
        # initialize pars (1 line of code)
        pars =  U.initialize_pars_deep(layers_dims)
        for i in range(num_iter):
                # forward propagation (1 line of code)
                xL, caches = U.L_model_forward(x_train, pars)
                # compute loss (1 line of code)
                loss = U.compute_loss(xL, y_train)
                # backward propagation (1 line of code)
                grads = U.L_model_backward(xL, y_train, caches)
                # pars update (1 line of code)
                pars = U.update_pars(pars, grads, learning_rate)
                # record loss at different epoches (use loss_all)
                if print_loss and i % 100 == 0:
                        print ("Loss after iteration %i: %f" %(i, loss))
                if print_loss and i % 100 == 0:
                        loss_all.append(loss)
        # predict test/train data samples (2 lines of code)
        y_predict_test = U.predict(pars,x_test)
        y_predict_train = U.predict(pars,x_train)
        # compute train/test accuracy
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100))
        return loss_all, y_predict_test


if __name__ == '__main__':
	# load datasets (1 line of code)
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# PA-I-4 reshape train_data_x/test_data_x as 2D arrays (2 lines of code)
	train_data_x = train_data_x.reshape(train_data_x.shape[0], -1).T
	test_data_x =  test_data_x.reshape(test_data_x.shape[0], -1).T

	# normalize data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0
 	
	# run model (1 line of code), layers_dims = [12288, 32, 16, 1]
	layers_dims = [12288, 32, 16, 1]
	loss1,result1 = model(train_data_x, train_data_y, test_data_x, test_data_y, layers_dims, 0.05, 5100, True)
	loss2,result1 = model(train_data_x, train_data_y, test_data_x, test_data_y, layers_dims, 0.01, 5100, True)
	loss3,result1 = model(train_data_x, train_data_y, test_data_x, test_data_y, layers_dims, 0.005, 5100, True)
	print(loss1)
	numit = np.arange(0,5100,100)
	plt.plot(numit,loss1, label = "learning rate = 0.05")
	plt.plot(numit,loss2, label = "learning rate = 0.01")
	plt.plot(numit,loss3, label = "learning rate = 0.005")
	plt.xlabel("# iter")
	plt.ylabel("loss")
	plt.legend()
	plt.show()





