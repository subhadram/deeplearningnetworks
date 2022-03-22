import numpy as np
import matplotlib.pyplot as plt
import scipy
import pickle as pkl
import lr_utils as U


# PA-II-5 write model function 
def model(x_train, y_train, x_test, y_test, num_iter, learning_rate, print_loss):
        # initialize parameters with gaussian_normal (1 line of code)
        w, b = U.initialize_zero(len(x_train)) #initializing to zero and not gaussian normal
        #w, b = U.initialize_norm(len(x_train)) #intialize to gaussian
        # model optimization (1 line of code)
        parameters, loss_all = U.optimize(w,b,x_train,y_train,num_iter,learning_rate, print_loss)
        # load trained parameters w and b 
        w = parameters["w"]
        b = parameters["b"]
        # predict test/train data samples (2 lines of code)
        y_predict_test = U.predict(w,b,x_test)
        y_predict_train = U.predict(w,b,x_train)
        # compute train/test accuracy 
        print("train accuracy: {} %".format(100 - np.mean(np.abs(y_predict_train - y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100))
	# save result
        result = {"loss_all": loss_all,"y_prediction_test": y_predict_test, "y_predict_train" : y_predict_train, "w" : w, "b" : b}
        return result



if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# QI-4 reshape train_data_x/test_data_x as 2D arrays (2 lines of code)
	train_data_x = train_data_x.reshape(train_data_x.shape[0], -1).T
	test_data_x =  test_data_x.reshape(test_data_x.shape[0], -1).T

	# normalize data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0
 
	# run model 
	result1 = model(train_data_x, train_data_y, test_data_x, test_data_y, 5100, 0.01, True)
	result2 = model(train_data_x, train_data_y, test_data_x, test_data_y, 5100, 0.005, True)
	result3 = model(train_data_x, train_data_y, test_data_x, test_data_y,5100, 0.001, True)
	loss1 = result1["loss_all"]
	loss2 = result2["loss_all"]
	loss3 = result3["loss_all"]
	print(loss1)
	numit = np.arange(0,5100,100)
	plt.plot(numit,loss1, label = "learning rate = 0.01")
	plt.plot(numit,loss2, label = "learning rate = 0.005")
	plt.plot(numit,loss3, label = "learning rate = 0.001")
	plt.xlabel("# iter")
	plt.ylabel("loss")
	plt.legend()
	plt.show()



