import numpy as np
import matplotlib.pyplot as plt
import scipy
import cnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch import Tensor
import os
torch.manual_seed(0)
transform=transforms.Compose([transforms.ToTensor()])

def create_dataset(train_data_x,train_data_y):
        train_data_x= train_data_x.transpose((0,3, 1, 2))
        #print(Tensor(train_data_y.T[:,0]))
        dataset = torch.utils.data.TensorDataset( Tensor(train_data_x), Tensor(train_data_y.T[:,0]) )
        return dataset

# model train
def model_train(train_data_x, train_data_y, test_data_x, test_data_y,opti):
        net = U.Net()
        epoch_num = 100
        trainset = create_dataset(train_data_x, train_data_y)
        loss_e = np.zeros((epoch_num,1))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=5,shuffle=True)
        accuracytr_e = np.zeros((epoch_num,1))
        accuracyts_e = np.zeros((epoch_num,1))
        criterion = nn.CrossEntropyLoss()
        if opti == "adam":
                optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        elif ( opti == "sgd"):
                optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.0, dampening=0, weight_decay=0, nesterov=False)
        else:
                optimizer = optim.Adagrad(net.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
                
        net.train()
        for epoch in range(epoch_num):  # loop over the dataset multiple times
                running_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                        # get the inputs; data is a list of [inputs, labels]
                        inputs, labels = data
                        inputs = inputs.float()
                        labels = labels.float()
                        labels = labels.type(torch.LongTensor)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward + backward + optimize
                        outputs = net(inputs)
                        #print(labels,outputs)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        if i % 45 == 2:    # print every  epoch
                                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 45))
                                loss_e[epoch] = running_loss/45
                                running_loss = 0.0
                accuracystr = model_test(train_data_x, train_data_y, net, epoch_num)
                accuracysts = model_test(test_data_x, test_data_y, net, epoch_num)
                accuracytr_e[epoch] = accuracystr
                accuracyts_e[epoch] = accuracysts
        print('Finished Training')
        return loss_e, accuracytr_e, accuracyts_e


# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, epoch_num):
        correct = 0
        total = 0
        ep = epoch_num
        testset = create_dataset(test_data_x, test_data_y)
        testloader = torch.utils.data.DataLoader(testset, batch_size=5,shuffle=False)
        with torch.no_grad():
                for data in testloader:
                        images, labels = data
                        outputs = net(images)
                        outputs = torch.sigmoid(outputs)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
        accu = 100 * correct / total
        return accu
                
        


if __name__ == '__main__':
        # load datasets
        train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()
        print(train_data_y.shape)
        #rescale data
        train_data_x = train_data_x / 255.0
        test_data_x = test_data_x / 255.0
        # model train (model test function can be called directly in model_train)
        lossa, traina, testa = model_train(train_data_x, train_data_y, test_data_x, test_data_y,"adam")
        lossg, traing, testg = model_train(train_data_x, train_data_y, test_data_x, test_data_y,"sgd")
        lossd, traind, testd = model_train(train_data_x, train_data_y, test_data_x, test_data_y,"ada")

        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        plt.plot(traina,label="Adam")
        plt.plot(traing, label = "SGD")
        plt.plot(traind, label = "Adagrad")
        plt.xlabel("epoch #")
        plt.ylabel("train accuracy")
        plt.legend()
        plt.show()
        
        plt.plot(testa,label="Adam")
        plt.plot(testg, label = "SGD")
        plt.plot(testd, label = "Adagrad")
        plt.xlabel("epoch #")
        plt.ylabel("test accuracy")
        plt.legend()
        plt.show()

        plt.plot(lossa,label="Adam")
        plt.plot(lossg, label = "SGD")
        plt.plot(lossd, label = "Adagrad")
        plt.xlabel("epoch #")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        
        










