import numpy as np

###################################### Part 1 ################################################

def problem_a (A, B):
    return A + B

def problem_b (A, B, C):
    return np.dot(A,B) - C

def problem_c (A, B, C):
    return A*B - C.T

def problem_d (x, y):
    return  np.dot(x.T,y)

def problem_e (A):
    return np.zeros(np.shape(A))

def problem_f (A, x):
    return np.linalg.solve(A,x)            # here this expresion basically slove A^-1 . x 

def problem_g (A, x):
    return (np.linalg.solve(A.T,x.T)).T   # using this property A.B = (B^T . A^T)^T where A is X and B is A^-1. 

def problem_h (A, alpha):
    return A + alpha*(np.eye(len(A),dtype = int))

def problem_i (A, i, j):
    return A[i][j]

def problem_j (A, i):
    return np.sum(A[i][0:len(A[0]):2])    # A[ith row] [Start (0) : Stop (till the length for the column) : Steps (2 since even)] 

def problem_k (A, c, d):
    A = A[np.nonzero(A)]
    return np.mean(A[(A>=c)&(A<=d)],dtype= np.float64)  

''' 
    A[(A>=c)&(A<=d)]  returns the value which are true in the given passed Array 
        
    eg: A = [1,2,3,4,5,6,7]
    A[(A>=3)&(A<=6)] will compute this condition for each element A[False,False,True,True,True,True,False] 
         
    Then one with the True value is returned as an array A and then we calculate the mean of the non-zero element 
    in that range. 
'''
 
def problem_l (A, k):
    eig_val,eig_vect = np.linalg.eig(A)
    index = eig_val.argsort()[::-1]  # Sorting the eig_val in descending order
    eig_val = eig_val[index]
    eig_vect = eig_vect[:,index]
    return eig_vect[:,:k] 

'''
    To extract the uptill ith column vector from the eig_vector , we use  eig_vect[:,:k]
    the eig_vector for each individual eig_val are stored in column so we need to slice across the column
'''

def problem_m (x, k, m, s):
    z = np.ones(len(x))
    mean = x+m*z
    covariance = s*np.eye(len(mean),dtype = int)
    return np.random.multivariate_normal(mean, covariance, k).T

'''
    N(x + mz (Mean: 1D Array); sI (Covariance : 2D Array))
    here the shape is (k x N) 
'''

def problem_n (A):
    return np.random.permutation(A)


############################################ Part 2 ###########################################

def linear_regression (X_tr, y_tr):
    
    ''' This function takes input as X and y and compute the value of W  '''
    
    w = np.dot(np.linalg.inv(np.dot(X_tr.T,X_tr)),(np.dot(X_tr.T,y_tr)))
    return w

def loss (X,y,w):
    
    ''' I defined a Loss function which takes in the X, Y and W as an 
        input and returns Loss (L) '''
        
    yhat = np.dot(X,w) 
    L = (1/(2*len(y)))*(np.sum((np.square(yhat - y))))
    return L

def train_age_regressor ():
    
    # Load data
    X_tr = np.load("age_regression_Xtr.npy")
    X_tr = np.reshape(X_tr,(np.size(X_tr[:,0,0]),np.size(X_tr[0,:,:]))) #used generalize approch which is independent of the input image size. 
    ytr = np.load("age_regression_ytr.npy")
    print("\nShape of X train is: {x_tr} and Shape of Y train is: {y_tr}".format (x_tr=np.shape(X_tr),y_tr= np.shape(ytr)))
    
    X_te = np.load("age_regression_Xte.npy")   
    X_te = np.reshape(X_te,(np.size(X_te[:,0,0]),np.size(X_te[0,:,:]))) #used generalize approch which is independent of the input image size.
    yte = np.load("age_regression_yte.npy")
    print("Shape of X test is: {x_te} and Shape of Y test is: {y_te}".format (x_te=np.shape(X_te),y_te= np.shape(yte)))
    
    w = np.zeros(X_tr.shape[1]) 
    w = linear_regression(X_tr, ytr)
    print("Shape of W is: {w} \n".format(w= np.shape(w)))
    
    L_tr = loss(X_tr,ytr,w)   # Calling the function Loss for calculating Training Loss     
    L_te = loss(X_te,yte,w)   # Calling the function Loss for calculating Testing Loss
    print("Training loss is {a} and Test loss is {b}".format(a=L_tr,b=L_te))

def main():    
    train_age_regressor()

if __name__=="__main__":
    main()
   
''' 
    Output: 

    Shape of X train is: (5000, 2304) and Shape of Y train is: (5000,)
    Shape of X test is: (2500, 2304) and Shape of Y test is: (2500,)
    Shape of W is: (2304,) 

    Training loss is 50.46755488028351 and Test loss is 269.1481156684566
    
'''
