import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

#DATA LOADING FUNCTIONS
def loadData (which):
    print("loading data...")
    faces = np.load("mnist_{}_images.npy".format(which))
    faces = faces.transpose()
    labels = np.load("mnist_{}_labels.npy".format(which))
    labels = labels.transpose()
    print("loaded ",which," images and labels")
    return faces, labels

#HELPER FUNCTIONS FOR G(X)
def calculate_z(weights,inp,bias):
    assert(bias.shape[0] == weights.shape[0])
    z = np.dot(weights,inp) + bias
    return z

def relu(z):
    r = z * (1*(z > 0))
    return r

def relu_prime(z):
    return 1*(z > 0)

def softmax(z2):
    assert(z2.shape[0] == 10)
    e_x = np.exp(z2)
    s = np.sum(e_x,axis = 0)
    soft = e_x/s # s.reshape(len(s),1)
    return soft


#PREDICT FUNCTION
def predict(w1,b1,w2,b2,x):
    z1 = calculate_z(w1,x,b1)
    h1 = relu(z1)
    z2 = calculate_z(w2,h1,b2)
    g = softmax(z2)
    return g

#GRADIENT FUNCTIONS
def gradient_w2(yhat,y,w1,x,b1,w2,a2_re,b2_re):
    z1 = calculate_z(w1,x,b1)
    h1 = relu(z1)
    pr = np.dot(yhat-y,h1.transpose())
    s = pr + a2_re*w2 + b2_re*np.sign(w2)
    return pr

def gradient_b2(yhat,y):
    return yhat - y

def gradient_w1(yhat,y,w2,w1,x,b1,a1_re,b1_re):
    g = g_tr(yhat,y,w2,w1,x,b1).transpose()
    pr = np.dot(g,x.transpose())
    s = pr + a1_re*w1 + b1_re*np.sign(w1)
    return pr

def gradient_b1(yhat,y,w2,w1,x,b1):
    g = g_tr(yhat,y,w2,w1,x,b1).transpose()
    return g

def g_tr(yhat,y,w2,w1,x,b1):
    z1 = calculate_z(w1,x,b1)
    f = np.dot((yhat - y).transpose(),w2)
    s = relu_prime(z1.transpose())
    return f*s


#CROSS-ENTROPY LOSS FUNCTION
def loss_fn(w1,b1,w2,b2,x_in,y):
    assert(x_in.shape[1] == y.shape[1])
    assert(y.shape[0] == 10)
    assert(x_in.shape[0] == 784)
    yhat = predict(w1,b1,w2,b2,x_in)
    pr = y*np.log(yhat)
    s_classes = np.sum(pr,axis = 0)
    n = len(s_classes)
    s_obs = np.sum(s_classes)
    unreg_cost = (-1/n) * s_obs
    frob1 = (1/2)*np.sum(np.square(w1))
    frob2 = (1/2)*np.sum(np.square(w2))
    reg_cost = unreg_cost + frob1 + frob2
    #print("unregulairzed, regularized cost: ",unreg_cost,reg_cost)
    #print("unregularized cost: ",unreg_cost)
    return unreg_cost,reg_cost

#TUNING FUNCTION
def findBestHyperparameters(vSet,vLabels,trainSet,trainLabels):
    c = 0
    i = 0
    min_val = 0
    best_params = None
    w_bs = None
    for epochs in [5,25]:
        for hidden_units in [30,50]:
            for learning_rate in [0.001,0.05]:
                for mbatch_size in [16,128]:
                    for reg in [(0.5,0.5,0.5,0.5),(0.8,0.2,0.8,0.2)]:
                        c+=1
                        print("\ntunning attempt#: ",c,"\n")
                        a1,b1,a2,b2 = reg[0],reg[1],reg[2],reg[3]
                        w1,b1,w2,b2 = sgd(trainSet,trainLabels,hidden_units,learning_rate,mbatch_size,epochs,a1,b1,a2,b2,0)
                        yhat = predict(w1,b1,w2,b2,vSet)
                        #print("")
                        print("settings (epochs,hidden units,learning rate, mbatch size, regularization params) : ",epochs,hidden_units,learning_rate,mbatch_size,reg)
                        val_acc = measure_acc(vLabels,yhat)
                        print("validation accuracy: ",val_acc)
                        #print("")
                        if val_acc > min_val:
                            i+=1
                            print("Found better params for the # ",i," time")
                            min_val = val_acc
                            best_params = (hidden_units,learning_rate,mbatch_size,epochs,reg)
                            w_bs = (w1,b1,w2,b2)
                        print("")
    print("")
    print("best params: (epochs,hidden units,learning rate, mbatch size, regularization params)")
    print("best params: ",best_params)
    print("best params validation accuracy: ",min_val)
    print("")
    return best_params,w_bs
                        
    
    

  
#MEASURE ACCURACY FUNCTION
def measure_acc(y,y_hat):
    return np.sum (1* (np.argmax(y, axis = 0) == np.argmax(y_hat, axis = 0)) ) / y.shape[1]
    

#STOCHASTIC GRADIENT DESCENT FUNCTION
def sgd(training_set,training_labels,number_of_neurons,learning_rate,mini_batch_size,number_of_epochs,a1,b1,a2,b2,to_print = 1):
    #initialize hyperparameters

    number_of_neurons = number_of_neurons #50
    ne = number_of_neurons
    e = learning_rate #0.001
    mini_batch_size = mini_batch_size #= 16
    n_ = mini_batch_size 
    number_of_epochs = number_of_epochs #25
    a1 = a1 #0.5
    b1 = b1 #0.5
    a2 = a2 #0.5
    b2 = b2 #0.5

    n = training_set.shape[1]
    assert(n == training_labels.shape[1])

    mini_batch = None

    #initialize weights,bias terms

    assert(training_set.shape[0] == 784)
    assert(training_labels.shape[0] == 10)
    w1 = ((1/math.sqrt(784))*np.random.randn(ne,784)) 
    w2 = ((1/math.sqrt(ne))*np.random.randn(10,ne)) 
    b1 = np.array([0.01]*w1.shape[0]).reshape(w1.shape[0],1)
    b2 = np.array([0.01]*w2.shape[0]).reshape(w2.shape[0],1)
    
    for epoch in range(0,number_of_epochs):
        if to_print == 1:
            print("epoch: ",epoch + 1)
        for round_n in range(0, (int(n/n_) + 1) - 1):
            #SELECT MINI BATCH SETS / STATS
            mini_batch = training_set[:,round_n * n_ : ((round_n + 1) * n_)]
            mini_batch_labels = training_labels[:,round_n * n_ : ((round_n + 1) * n_)]
            mini_batch_y_hat = predict(w1,b1,w2,b2,mini_batch)
            #gradients
            w2_gr = gradient_w2(mini_batch_y_hat,mini_batch_labels,w1,mini_batch,b1,w2,a2,b2)
            b2_gr = gradient_b2(mini_batch_y_hat,mini_batch_labels)
            w1_gr = gradient_w1(mini_batch_y_hat,mini_batch_labels,w2,w1,mini_batch,b1,a1,b1)
            b1_gr = gradient_b1(mini_batch_y_hat,mini_batch_labels,w2,w1,mini_batch,b1)

            #update based on gradients
            w2 = w2 - e*w2_gr
            b2 = np.mean(b2 -e*b2_gr, axis =1).reshape(b2.shape[0],1)
            w1 = w1 - e*w1_gr
            b1 = np.mean(b1 - e*b1_gr, axis = 1).reshape(b1.shape[0],1)
        yhat = predict(w1,b1,w2,b2,training_set)
        if to_print == 1:
            print("training acc: ",measure_acc(training_labels,yhat))
        if epoch > number_of_epochs - 20 and to_print == 2:
            print("epoch :",epoch + 1)
            u,r = loss_fn(w1,b1,w2,b2,training_set,training_labels) #print cost
            print("unregularized cost: ",u)
            test_acc = measure_acc(training_labels,yhat)
            print("accuracy: ",test_acc)
            
    return w1,b1,w2,b2
        
        
    
    


    

#MAIN FUNCTION
if __name__ == "__main__":
    testingDigits, testingLabels = loadData("test")
    trainingDigits, trainingLabels = loadData("train")
    validationDigits, validationLabels = loadData("validation")
    print("\n TUNING \n")
    #params,w_b = findBestHyperparameters(validationDigits,validationLabels,trainingDigits, trainingLabels) #UNCOMMENT IF YOU RUN TUNE, ABOUT 30 MINUTES
    print("\n DONE TUNING \n")
    
    '''
    BEST HYPERPARAMETERS
    epochs,hidden units,learning rate, mbatch size, regularization params
    best params: 50,0.001,16,25,(0.5,0.5,0.5,0.5)
    '''
    
    print("\nTESTING")
    print("\n Train on training set, using best hyperparams\n")
    print("best params from tuning {epochs,hidden units,learning rate, mbatch size, regularization params}: \n",25,50,0.001,16,(0.5,0.5,0.5,0.5),"\n\n")
    w1,b1,w2,b2 = sgd(trainingDigits,trainingLabels,50,0.001,16,25,0.5,0.5,0.5,0.5,2) #STOCHASTIC GRADIENT ON TRAINING, USING BEST PARAMS FROM TUNING
    yht = predict(w1,b1,w2,b2,testingDigits) #TESTING PREDICTION
    t_acc = measure_acc(testingLabels,yht) #TESTING ACCURACY
    print("")
    print("\n\nAccuracy on the testing set: ",t_acc)
    u,r = loss_fn(w1,b1,w2,b2,testingDigits,testingLabels) #TESTING LOSS
    print("\nUnregularized cost: ",u)
    print("\n")
    
    







