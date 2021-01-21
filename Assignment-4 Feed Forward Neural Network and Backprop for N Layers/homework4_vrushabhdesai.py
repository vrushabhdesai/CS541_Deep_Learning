import numpy as np
import math

def load_data():
    ''' This function load the data set '''
    
    X_tr = np.load("mnist_train_images.npy")
    ytr = np.load("mnist_train_labels.npy")
    X_val = np.load("mnist_validation_images.npy")
    yval = np.load("mnist_validation_labels.npy") 
    X_te = np.load("mnist_test_images.npy") 
    yte = np.load("mnist_test_labels.npy")
        
    return X_tr,ytr,X_val,yval,X_te,yte

#########################################################################################################################

def int_W(no_of_layers,no_of_unit):
    ''' This funtion initialize the W and b matrix '''
    
    W = {}      #created a dictionary to store the W of each layeres
    b = {}      #created a dictionary to store the b of each layeres
    for i in range(1,no_of_layers+2):           # for loop creates a dictionaries of W and b 
        var=-1*(no_of_unit[i-1]**(-0.5))/2
        if i == 1:
            W[i] = var*np.random.randn(784,no_of_unit[i-1])
            b[i] = 0.01*np.ones((1,no_of_unit[i-1]))
        else:
            W[i] =  var*np.random.randn(no_of_unit[i-2],no_of_unit[i-1])
            b[i] = 0.01*np.ones((1,no_of_unit[i-1]))
        print("Shape of W{i} is: {z}\nShape of b{i} is: {b}".format(i=i,z=W[i].shape,b=b[i].shape))
    return W,b  

################################################################################################################################

def comp_Z(X_tr,W,b):
    ''' This function computes Z '''
    Z = np.add(np.dot(X_tr,W),b)
    return Z

####################################################################################################################################

def normalize_Z(Z):
    ''' This function applys softmax on the output layer '''
    exp_z = np.exp(Z)
    for i in range(len(Z[:,-1])):
        sum_z = np.sum(exp_z[i,:])
        exp_z[i,:] = exp_z[i,:]/sum_z

    return exp_z

#################################################################################################################################

def relu(Z):
    Z = np.array(Z)
    Z[Z<=0]=0
    return Z
    
###################################################################################################################################
    
def log_loss(Yhat,y,w,alpha):
    ''' This function calculated Cross entropy loss '''
    sum_w = 0
    for j in range(1,len(w)+1):
        for i in range(len(Yhat[-1,:])):                # calculate W1.T*W1, W2.T*W2, ........ ,Wc.T*Wc  
            W_reg = np.dot(w[j][:-1,i].T,w[j][:-1,i])
            sum_w = sum_w + W_reg
            #print(w[:-1,i].shape)
    Loss = -1/len(y[:,0])*np.sum(y*np.log(Yhat))
    Loss_reg = Loss + alpha/2*sum_w
    
    return Loss_reg 
  
####################################################################################################################################
def accuracy(X,no_of_layers,y,W,b):
    
    Yhat,Z = forward_prop(X,no_of_layers,W,b)            
    yhat_boolean=(Yhat.argmax(axis=1)==y.argmax(axis=1))
    return (np.count_nonzero(yhat_boolean == True)/float(len(y)))*100    

####################################################################################################################################

def forward_prop(X_tr,no_of_layers,W,b):
    Z = {}
    Z[0] = X_tr 
    
    for i in range(1,no_of_layers+2):
        if i == 1:                          
            Z[i] = comp_Z(X_tr,W[i],b[i])   # for the first layer X.W + b and no relu 
        elif(i<=no_of_layers):
            Z[i] = relu(comp_Z(Z[i-1],W[i],b[i]))   # for the hidden layers Z.W + b and apply relu  
        else:
            Z[i] = comp_Z(Z[i-1],W[i],b[i])     # for the last layer Z.W + b and no relu as we apply softmax directly 
        #print("Shape of Z{i} is: {z}".format(i=i,z=Z[i].shape))
    Yhat = normalize_Z(Z[no_of_layers+1])   #apply softmax to the last layers
    return Yhat,Z

####################################################################################################################################

def Relu_derv(z):
    z[z>0] = 1
    z[z<=0] = 0
    return z

#######################################################################################################################################
    
def findBestHyperparameters(W,b,Xtr,ytrain,X_val,y_val,no_of_layers):
    
    ''' This fuction take the Xtrain and Ytrain and varry the hyper-parameter in the specified range and train the model '''
    
    L_reg_val_old = 10000   # Initialize the regularized Loss as 10000 at start 

    for e in range(1,5,1):        # varry epoch from [30,60,90,120] 
        epoch = e*30
        for b in range(1,5,1):      # varry mini_batch_size from [50,100,150,200]
            mini_batch_size = b*50
            for a in range(1,5,1):      # varry alpha from [0.1,0.01,0.001,0.0001] 
                alpha = math.pow(10,-a)    
                for l in range(1,5,1):      # varry learning_rate from [0.1,0.01,0.001,0.0001]
                    learning_rate = math.pow(10,-l)    
                    print("epoch: {e}, batch: {b}, learning: {l}, alpha: {a}".format(e=epoch,b=mini_batch_size,l= learning_rate,a=alpha))
                    W = train(W,b,Xtr,ytrain,no_of_layers,epoch,mini_batch_size,alpha,learning_rate)
                    
                    # This part calculate the value of the Validation loss using updated W
                    Yhat_val,Z_val = forward_prop(X_val,no_of_layers,W,b)   
                    L_reg_val = log_loss(Yhat_val,y_val,W,0.01)
    
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

    
def train(W,b,Xtr,ytr,X_val,yval,no_of_layers,epoch,mini_batch_size,alpha,lr):
    
    ''' This function will take the hyper-parameter and W and train the model and return the updated W ''' 
    dL_db = {}
    dL_dw = {}
    for _ in range(epoch):
        for j in range(int(math.floor(len(Xtr[:,1])/mini_batch_size))):
            
            start = j*mini_batch_size
            stop = start + mini_batch_size

            
            Yhat,Z =forward_prop(Xtr[start:stop,:],no_of_layers,W,b)

            '''  Back_prop calcualtion are done below '''
            
            n=mini_batch_size            
            g_T = (Yhat-ytr[start:stop,:])/n
            g_T_c = np.sum(g_T,axis = 0)
            g_T_c = g_T_c.reshape(1,len(g_T_c))

            dL_db[no_of_layers+1] = g_T_c
            dL_dw[no_of_layers+1] = np.dot(Z[(no_of_layers+1)-1].T,g_T)

            ''' calculates dL_dW and dL_db for hidden layers '''
            
            for l in range(no_of_layers,0,-1):

                g_T = np.dot(g_T,W[l+1].T)*Relu_derv(Z[l])
                g_T_c = np.sum(g_T,axis=0)
                g_T_c = g_T_c.reshape(1,len(g_T_c))
                dL_db[l] = g_T_c
                #print("dL_db  =" ,g_T_c.shape)
                dL_dw[l] = np.dot(Z[l-1].T,g_T)
                #print("dL_dW  =" ,dL_dw[l].shape)
            
            ''' Updates the value of W and b after back prop'''
            for i in range(1,no_of_layers+2):
                b[i] -= lr*dL_db[i]
                W[i] -= lr*(dL_dw[i]+alpha*W[i])
            
            
            # This part calculate the value of the Validation loss using updated W
            Yhat_val,Z_val = forward_prop(X_val,no_of_layers,W,b)
            l = log_loss(Yhat_val,yval,W,0.01)
        
            # Calculate Training accuracy 
            acc = accuracy(Xtr[start:stop,:],no_of_layers,ytr[start:stop,:],W,b)
            print("Validation Loss: {l}  Epoch Number: {_}  Batch Number: {b}  Training Accuracy: {a}".format(l=l,_=_,b=j,a=acc))
    
    return W,b

#########################################################################################################################################
    
def main():
    
    ''' Load the data set '''
    X_tr,ytr,X_val,yval,X_te,yte = load_data()
    
    #no_of_layers = [1,2,3,4]
    no_of_layers = 3              # No of hidden layers 
    no_of_unit = [20,30,40,10]    #last layer has 10 classes thats y it has an extra 10 units
    
    W,b = int_W(no_of_layers,no_of_unit)
    
    ''' Tune the hyper parameter and training the model'''
    #w,e,m,a,l = findBestHyperparameters(W,b,X_tr,ytr,X_val,yval,no_of_layers) 
    
    ''' Run model with tunned parameters '''
    w,b = train(W,b,X_tr,ytr,X_val,yval,no_of_layers,epoch = 15,mini_batch_size = 50,alpha = 0.01,lr= 0.1) 
    
    ''' Calculate the Cross Entropy Loss on Validation and Test set '''
    Yhat_val,Z_val = forward_prop(X_val,no_of_layers,W,b)
    Loss_val = log_loss(Yhat_val,yval,w,0.01)
    print("Validation Loss: {l}".format(l=Loss_val))
    
    Yhat_test,Z_te = forward_prop(X_te,no_of_layers,W,b)
    Loss_test = log_loss(Yhat_test,yte,w,0)
    print("Test Loss: {l}".format(l=Loss_test))
    
    
    # ''' Calculate the Accuracy on data set'''
    Acc_train = accuracy(X_tr,no_of_layers,ytr,w,b)
    Acc_val = accuracy(X_val,no_of_layers,yval,w,b)
    Acc_test = accuracy(X_te,no_of_layers,yte,w,b)
    print("Validation Accuracy:{v} \nTest Accuracy: {l} \nTrain Accuracy: {w} \n".format(v=Acc_val,l=Acc_test,w=Acc_train))
   
if __name__=="__main__":
    main()
