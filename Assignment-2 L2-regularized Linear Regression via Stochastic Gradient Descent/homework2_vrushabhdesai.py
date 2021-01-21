import numpy as np
import math

############################################ Part 2 ###########################################

class Data_Preprocessing():
    
    def __init__(self,X_tr,ytr,X_te,yte):
        
        X_tr = np.reshape(X_tr,(np.size(X_tr[:,0,0]),np.size(X_tr[0,:,:]))) #used generalize approch which is independent of the input image size. 
        self.X_tr = np.c_[np.ones(np.shape(X_tr[:,0])),X_tr]
        
        self.ytr = ytr
        print("\nShape of X train is: {x_tr} and Shape of Y train is: {y_tr}".format (x_tr=np.shape(self.X_tr),y_tr= np.shape(self.ytr)))
        
        X_te = np.reshape(X_te,(np.size(X_te[:,0,0]),np.size(X_te[0,:,:]))) #used generalize approch which is independent of the input image size.
        self.X_te = np.c_[np.ones(np.shape(X_te[:,0])),X_te]
        
        self.yte = yte
        #print("Shape of X test is: {x_te} and Shape of Y test is: {y_te}".format (x_te=np.shape(self.X_te),y_te= np.shape(self.yte)))
        
        self.w = np.ones(self.X_tr.shape[1])
        
    def combined_data(self):
        
        ''' This function combine the X_te and yte into a single matrix and shuffle the data randomly along the row'''
        
        combined_train = np.c_[self.X_tr,self.ytr]
        np.random.shuffle(combined_train)    #shuffles the data in the combined data set 
        print("Size of the combined data set is {s}".format (s = combined_train.shape))
        return  combined_train 
    
    def split_data(self):
        
        ''' This program splits the training data into training and validation set ''' 
        
        data = self.combined_data()
        row, col = data.shape
        split_per = int(0.2*row)            #split 20 percent of data into validation set
        data_train = data[split_per:,:]
        data_val = data[0:split_per,:]
        print("Shape of training set is {a}".format(a=data_train.shape))
        print("Shape of validation set is {a}".format(a=data_val.shape))
        return data_train, data_val
    
    def X_Y_data(self):
        
        ''' This function sperate combined train data and validation data into X train, Y train, X validation and Y validation '''
        
        data_train, data_val = self.split_data()    
        ytrain = data_train[:,-1]
        Xtr = data_train[:,:-1]
        y_val = data_val[:,-1]
        X_val = data_val[:,:-1]
#        print("Shape of X training set is {a}".format(a=Xtr.shape))
#        print("Shape of Y training set is {a}".format(a=ytrain.shape))
#        print("Shape of X validation is {a}".format(a=X_val.shape))
#        print("Shape of Y validation is {a}".format(a=y_val.shape))   
        return Xtr,ytrain,X_val,y_val 
    
        
def tune(W_int,Xtr,ytrain,X_val,y_val ):
    
    ''' This fuction take the Xtrain and Ytrain and varry the hyper-parameter in the specified range and train the model '''
    
    L_reg_val_old = 10000   # Initialize the regularized Loss as 10000 at start 

    for e in range(1,5,1):        # varry epoch from [100,200,300,400] 
        epoch = e*100
        for b in range(1,5,1):      # varry mini_batch_size from [100,200,300,400]
            mini_batch_size = b*100
        #mini_batch_size = 100
            for a in range(1,5,1):      # varry alpha from [0.1,0.01,0.001,0.0001] 
                alpha = math.pow(10,-a)
        #alpha = 0.0001                 
                for l in range(1,5,1):      # varry learning_rate from [0.1,0.01,0.001,0.0001]
                    learning_rate = math.pow(10,-l)    
        #learning_rate = 0.001
                    print("epoch: {e}, batch: {b}, learning: {l}, alpha: {a}".format(e=epoch,b=mini_batch_size,l= learning_rate,a=alpha))
                    W = train(W_int,Xtr,ytrain,epoch,mini_batch_size,alpha,learning_rate)
                    
                    # This part calculate the value of the Validation loss using updated W
                    yhat_val = np.dot(X_val,W) 
                    Loss_val = (1/(2*len(y_val)))*(np.sum((np.square(yhat_val - y_val))))
                    L_reg_val = Loss_val + (alpha/2)*np.dot(W[:-1].T,W[:-1])
                        
                    # This part will store the parameter in the variable and update the loss if it is less than previous loss 
                    if(L_reg_val < L_reg_val_old):
                        L_reg_val_old = L_reg_val
                        
                        epoch_opti = epoch 
                        mini_batch_size_opti = mini_batch_size 
                        alpha_opti = alpha
                        learning_rate_opti = learning_rate
                        W_opti = W
                
                    print("Validation set loss: {L}".format(L=L_reg_val))
 
    return W_opti, epoch_opti,mini_batch_size_opti,alpha_opti,learning_rate_opti
    
     
          
def train(W_int,Xtr,ytrain,epoch,mini_batch_size,alpha,learning_rate):
    
    ''' This function will take the hyper-parameter and W and train the model and return the updated W ''' 
    W = W_int
            
    for _ in range(epoch):
        for i in range(math.floor(len(Xtr[:,-1])/mini_batch_size)):
            
            start = i*mini_batch_size
            stop = start + mini_batch_size
            #print(self.Xtr[start:stop,:].shape)
            
            yhat = np.dot(Xtr[start:stop,:],W)
            
            Loss = (1/(2*len(ytrain[start:stop])))*(np.sum((np.square(ytrain[start:stop] - yhat))))
            Loss_reg = Loss + (alpha/2)*np.dot(W[:-1].T,W[:-1])
            
            dL_dw = (1/len(ytrain[start:stop]))*np.dot(np.transpose(Xtr[start:stop,:]),(yhat - ytrain[start:stop]))
            
            W = W - learning_rate*(dL_dw) - learning_rate*alpha*W
    
            print(" The loss at is {L} and epoch number {_} batch number {b}".format(L=Loss_reg,_=_,b=i))
   
    Loss = 0  # This is to avoid the overflow of the variable Loss
    return W

def loss (W,X,y,alpha):
        
    yhat = np.dot(X,W) 
    Loss_test = (1/(2*len(y)))*(np.sum((np.square(yhat - y))))
    L_reg_test = Loss_test + (alpha/2)*np.dot(W[:-1].T,W[:-1])
    print("Test set loss: {L}".format(L=L_reg_test))        
    

def main():
    
    # Load the data set 
    X_tr = np.load("age_regression_Xtr.npy")
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.load("age_regression_Xte.npy") 
    yte = np.load("age_regression_yte.npy")
    
    #Data_Preprocessing(X_tr,ytr,X_te,yte)
    d = Data_Preprocessing(X_tr,ytr,X_te,yte) 
#    Xtr,ytrain,X_val,y_val = d.X_Y_data()
#    W_int = d.w
##    w,e,m,a,l = tune(W_int,Xtr,ytrain,X_val,y_val)         # used to the parameters
#    W = train(W_int, Xtr,ytrain,1600,100,0.0001,0.001)   #train the model with optimized hyperparameters
#    loss(W,X_val,y_val,0.0001) # obtain Validation Loss 
#    loss(W,d.X_te,d.yte,0)      # obtain Test Loss 

if __name__=="__main__":
    main()
   