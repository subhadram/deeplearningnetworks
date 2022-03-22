import numpy as np
import matplotlib.pyplot as plt
# import scipy
import gnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

torch.manual_seed(0)
def model_test(model, adj, features, labels, idx_test):
    model.eval()
    pred = model(features,adj).max(dim=1)[1]
    accu = U.accuracy(pred,labels)
    return 100*accu


def model_train(adj, features, labels, idx_train, idx_test,opti):
    nfeat = features.size()[1]
    #print(nfeat)
    model = U.GCN(nfeat,32,7,0.5)
    if opti == "sgd":
            optimizer = optim.SGD(model.parameters(),lr=0.3)
                
    elif ( opti == "adam"):
                
                optimizer = optim.Adam(model.parameters(), lr=0.1, weight_decay=5e-4)
    else:
                optimizer = optim.Adagrad(model.parameters(), lr=0.1, lr_decay=0, weight_decay=5e-4, initial_accumulator_value=0, eps=1e-10)
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    epoch_num = 5000
    accuracytr_e = np.zeros((epoch_num,1))
    accuracyts_e = np.zeros((epoch_num,1))
    loss_e = []
    for epoch in range(epoch_num):
            model.train()
            out = model(features,adj)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss1 = loss.detach().numpy()
            loss_e.append(loss1)
            train_acc = model_test(model, adj, features, labels, idx_train)
            test_acc = model_test(model, adj, features, labels, idx_test)
            accuracytr_e[epoch] = train_acc
            accuracyts_e[epoch] = test_acc
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}%, Test Acc: {:.5f}%'.
                  format(epoch, loss, train_acc, test_acc))
    print('Finished Training')
    return loss_e, accuracytr_e, accuracyts_e






if __name__ == '__main__':
    # load datasets
    adj, features, labels, idx_train, idx_test = U.load_data()
    print(features.size())
    # model train (model test function can be called directly in model_train)
    lossg,traing,testg = model_train(adj, features, labels, idx_train, idx_test,"sgd")
    lossa,traina,testa = model_train(adj, features, labels, idx_train, idx_test,"adam")
    
    lossd,traind,testd = model_train(adj, features, labels, idx_train, idx_test,"ada")
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
	






