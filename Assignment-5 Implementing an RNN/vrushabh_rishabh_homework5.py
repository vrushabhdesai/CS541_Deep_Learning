import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime

class RNN:    
  def __init__ (self, numHidden, numInput, numOutput): 
    self.numHidden = numHidden
    self.numInput = numInput        
    self.U = np.random.randn(numHidden, numHidden) * 1e-1        
    self.V = np.random.randn(numHidden, numInput) * 1e-1        
    self.w = np.random.randn(1,numHidden) * 1e-1        # TODO: IMPLEMENT ME   
    self.steps = 0

  def tanh_prime(self,z):
    return 1 - np.square(np.tanh(z))
    
  def forward_propogation (self, x):        
    # TODO: IMPLEMENT ME  
    numTimesteps = len(x)  
    y_pred = np.zeros(x.shape)
    y_obt=0.0
    self.activations = []
    self.outputs = []
    h_t_1 = np.zeros((self.numHidden,1))
    for time in range(numTimesteps):
      out1 = np.dot(self.V,x[time])
      out2 = np.dot(self.U,h_t_1)
      output = out1 + out2
      h_t = np.tanh(output)
      y_hat = np.dot(self.w,h_t)
      y_obt += y_hat
      y_pred[time] = y_hat
      h_t_1 = h_t
      self.activations.append(h_t)
      self.outputs.append(output)
    return y_pred

  def backward_prop(self,x,y,y_pred):
    dU = np.zeros(self.U.shape)
    dV = np.zeros(self.V.shape)
    dw = np.zeros(self.w.shape)

    dU_t = np.zeros(self.U.shape)
    dV_t = np.zeros(self.V.shape)
    dw_t = np.zeros(self.w.shape)
    #print("dV_t: ",dV_t.shape)

    dU_i = np.zeros(self.U.shape)
    dV_i = np.zeros(self.V.shape)

    numTimesteps = len(x)

    y_delta = y_pred - y

    for t in range(numTimesteps-1,0,-1):
      dw_t = np.dot(y_delta[t],np.transpose(self.activations[t]))
      dh_prime = self.tanh_prime(self.outputs[t])
      dz_t = np.dot(self.w.T, y_delta[t])
      dh_t = dz_t * dh_prime
      prev_dh = np.dot(self.U.T, dh_t)
      
      for i in range(t-1, max(-1, t-self.steps-1), -1):

        ds = dz_t + prev_dh  
        ds_t = ds * self.tanh_prime(self.outputs[t])
        dU_i = np.dot(self.U,self.activations[t-1])
        dV_i = np.dot(self.V,x[t]) 
        prev_dh = np.dot(self.U.T, ds_t)
    
        dU_t += dU_i
        dV_t += dV_i

      dV += dV_t
      dU += dU_t
      dw += dw_t
    
    return dV, dU, dw

  def loss(self,y,y_pred):
    y = np.array(y)
    loss = 0.5 * sum(np.square(y_pred - y))
    return loss
  

  def train_RNN(self,x,y,learning_rate):
    y_pred = self.forward_propogation(x)
    dV, dU, dw = self.backward_prop(x,y,y_pred)
    self.U -= self.learning_rate * dU
    self.V -= self.learning_rate * dV
    self.w -= self.learning_rate * dw


  def fit(self, x, y,learning_rate,epochs):
    self.learning_rate = learning_rate
    for i in range(epochs):
      x_ = self.train_RNN(x,y,learning_rate)
      if i%10 == 0:
        y_hat = self.forward_propogation(x)
        trloss = self.loss(y,y_hat)
        print("[{}]".format(i+1),'Train Loss:', trloss)
        print("#" * 40)
              
def generateData ():
    total_series_length = 50
    echo_step = 2
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    print(xs)
    print(ys)
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    rnn.fit(xs,ys,0.02,1000000000)
    
