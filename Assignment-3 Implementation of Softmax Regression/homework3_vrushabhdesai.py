import numpy as np
import math

############################################ Part 3 #########################################################################
def load_data():
    
    Xtr = np.load("H:\Masters Study\Deep Learning\Home Work\HW3\dataset\mnist_train_images.npy")
    ytr = np.load("H:\Masters Study\Deep Learning\Home Work\HW3\dataset\mnist_train_labels.npy")
    Xval = np.load("H:\Masters Study\Deep Learning\Home Work\HW3\dataset\mnist_validation_images.npy")
    yval = np.load("H:\Masters Study\Deep Learning\Home Work\HW3\dataset\mnist_validation_labels.npy") 
    Xte = np.load("H:\Masters Study\Deep Learning\Home Work\HW3\dataset\mnist_test_images.npy") 
    yte = np.load("H:\Masters Study\Deep Learning\Home Work\HW3\dataset\mnist_test_labels.npy")
    
    ''' Add 1's coloumn and define weight vector '''
    X_tr = np.c_[np.ones(np.shape(Xtr[:,0])),Xtr]
    X_val = np.c_[np.ones(np.shape(Xval[:,0])),Xval]
    X_te = np.c_[np.ones(np.shape(Xte[:,0])),Xte]
    
    return X_tr,ytr,X_val,yval,X_te,yte

################################################################################################################################
    
def normalize_Z(X_tr,W):
    
    ''' This function computes Z and the apply softmax on the output layer '''
    Z = np.dot(X_tr,W)
    #print("Shape of Z: {z}\n".format(z=Z.shape))
    exp_z = np.exp(Z)
    for i in range(len(Z[:,-1])):
        sum_z = np.sum(exp_z[i,:])
        exp_z[i,:] = exp_z[i,:]/sum_z
        
    return exp_z
    
###################################################################################################################################
    
def log_loss(Yhat,y,w,alpha):
    
    W_reg = np.zeros((Yhat[-1,:].shape))         # create a vector to store the product 
    
    for i in range(len(Yhat[-1,:])):                # calculate W1.T*W1, W2.T*W2, ........ ,Wc.T*Wc  
        W_reg[i] = np.dot(w[:-1,i].T,w[:-1,i])
        #print(w[:-1,i].shape)
    
    Loss = -1/len(y[:,0])*np.sum(y*np.log(Yhat))
    Loss_reg = Loss + alpha/2*(np.sum(W_reg))
    
    return Loss_reg 
  
####################################################################################################################################
def accuracy(X,y,W):
    
    Yhat = normalize_Z(X,W)               # apply softmax and find Yhat 
    
    #Yhat,Z = forward_prop(X,no_of_layers,W,b)            
    yhat_boolean=(Yhat.argmax(axis=1)==y.argmax(axis=1))
    return (np.count_nonzero(yhat_boolean == True)/float(len(y)))*100 

#######################################################################################################################################
    
def tune(W_int,Xtr,ytrain,X_val,y_val ):
    
    ''' This fuction take the Xtrain and Ytrain and varry the hyper-parameter in the specified range and train the model '''
    
    L_reg_val_old = 10000   # Initialize the regularized Loss as 10000 at start 

    for e in range(1,5,1):        # varry epoch from [50,150,200,250] 
        epoch = e*50
        for b in range(1,5,1):      # varry mini_batch_size from [1000,2000,3000,4000]
            mini_batch_size = b*1000
            for a in range(1,5,1):      # varry alpha from [0.1,0.01,0.001,0.0001] 
                alpha = math.pow(10,-a)    
                for l in range(1,5,1):      # varry learning_rate from [0.1,0.01,0.001,0.0001]
                    
                    learning_rate = math.pow(10,-l)    
                    print("epoch: {e}, batch: {b}, learning: {l}, alpha: {a}".format(e=epoch,b=mini_batch_size,l= learning_rate,a=alpha))
                    W = train(W_int,Xtr,ytrain,epoch,mini_batch_size,alpha,learning_rate)
                    
                    # This part calculate the value of the Validation loss using updated W
                    Yhat_val = normalize_Z(X_val,W)
                    L_reg_val = log_loss(Yhat_val,y_val,W,alpha)
                        
                    ''' This part will store the parameter in the variable and update the loss if it is less than previous loss '''
                    if(L_reg_val < L_reg_val_old):
                        L_reg_val_old = L_reg_val
                        
                        epoch_opti = epoch 
                        mini_batch_size_opti = mini_batch_size 
                        alpha_opti = alpha
                        learning_rate_opti = learning_rate
                        W_opti = W
                
                    print("Validation set loss: {L}".format(L=L_reg_val))
 
    return W_opti, epoch_opti,mini_batch_size_opti,alpha_opti,learning_rate_opti

#######################################################################################################################################
    
def train(W_int,Xtr,ytrain,epoch,mini_batch_size,alpha,learning_rate):
    
    ''' This function will take the hyper-parameter and W and train the model and return the updated W ''' 
    W = W_int

    for _ in range(epoch):
        for i in range(math.floor(len(Xtr[:,-1])/mini_batch_size)):
            
            start = i*mini_batch_size
            stop = start + mini_batch_size
            #print(self.Xtr[start:stop,:].shape)
            
            Yhat = normalize_Z(Xtr,W)
            Loss = log_loss(Yhat[start:stop,:],ytrain[start:stop,:],W,alpha)
            dL_dw = (1/len(ytrain[start:stop,0]))* np.dot(Xtr[start:stop,:].T,(Yhat[start:stop,:]-ytrain[start:stop,:]))
            
            W = W - learning_rate*(dL_dw) - learning_rate*alpha*W
    
            acc = accuracy(Xtr[start:stop,:],ytrain[start:stop,:],W)
            print(" Training, Loss: {L} Epoch Number: {_} Batch Number: {b}".format(L=Loss,_=_,b=i))
            print(" Accuracy: {a}".format(a = acc))
    
    Loss = 0     # This is to avoid the overflow of the variable Loss
    return W

#########################################################################################################################################
    
def main():
    
    ''' Load the data set '''
    X_tr,ytr,X_val,yval,X_te,yte = load_data()
    

    W = np.random.rand(len(X_tr[-1]),len(ytr[-1]))
    print("Shape of X_tr:{t} \nShape of X_val:{v} \nShape of X_te:{te} \nShape of W:{w} \n".format(t=X_tr.shape,v=X_val.shape,te=X_te.shape,w=W.shape))
    
    
    ''' Tune the hyper parameter and training the model'''
    #w,e,m,a,l = tune(W,X_tr,ytr,X_val,yval) 
    w = train(W,X_tr,ytr,epoch = 100,mini_batch_size = 1000,alpha = 0.001,learning_rate= 0.1) 
    
    
    ''' Calculate the Cross Entropy Loss on Validation and Test set '''
    Yhat_val = normalize_Z(X_val,w)
    Loss_val = log_loss(Yhat_val,yval,w,0.01)
    print(" Validation Loss: {l}".format(l=Loss_val))
    
    Yhat_test = normalize_Z(X_te,w)
    Loss_test = log_loss(Yhat_test,yte,w,0)
    print(" Test Loss: {l}".format(l=Loss_test))
    
    
    ''' Calculate the Accuracy on data set'''
    Acc_train = accuracy(X_tr,ytr,w)
    Acc_val = accuracy(X_val,yval,w)
    Acc_test = accuracy(X_te,yte,w)
    print("Validation Accuracy:{v} \nTest Accuracy: {l} \nTrain Accuracy: {w} \n".format(v=Acc_val,l=Acc_test,w=Acc_train))

if __name__=="__main__":
    main()

###########################################################################################################################################
''' 
    Output:
    Training Loss: 0.6847250762964152 
    Validation Loss: 4.86320517816365
    Test Loss: 0.29543302401205157
    
    Train Accuracy: 91.59272727272727
    Validation Accuracy: 92.2 
    Test Accuracy: 91.81 
     
'''
#############################################################################################################################################