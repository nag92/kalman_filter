'''
Created on Jan 2, 2016

@author: nathaniel Goldfarb
This is a EKF, it is dependent on the numdifftools modul
It will evaluate any non-linear (or linear) set of functions

'''
import numpy as np
import numdifftools as nd

class EKF(object):
    '''
    classdocs
    '''


    def __init__(self, g, h,sigma,Q,R,state):
        self.g = g #model function, a list if lamnda fns
        self.h = h #sensor function,a list if lamnda fns
        self.sigma = sigma #prediction
        self.Q = Q #Process noise covarence
        self.R = R #measurement noise 
        self.state = state #state
        self.G = [nd.Jacobian(f) for f in self.g  ] #jacobian of the model->a list if lamnda fns
        self.H = [nd.Jacobian(f) for f in self.h  ] #jacobian of the sensor->a list if lamnda fns
    
    #evaluate the model and return a matrix
    def eval_model(self,u,x,fn):
        solved = []
        for ii in xrange(len(fn)):
            solved.append(fn[ii](x[ii],u[ii]) )
        #return  np.matrix( np.vstack([f(x,u) for f in fn]))
        return np.vstack(solved)
    
    #evaluate the sensor and return a matrix
    def eval_sensor(self,x,fn):
        solved = []
        #for ii in xrange(len(fn)):
            #solved.append(fn[ii](x[ii]) )
        #return np.matrix(np.vstack(solved))
        return np.matrix( np.vstack([f(x) for f in fn]))
    
    #update the model 
    # u = control input
    # z = sensor input
    def update(self,u,z):
        
        #evaluate the model and the jacobian of the model
        mu = self.eval_model(u, self.state, self.g)
        G = np.matrix( self.eval_model(u, self.state, self.G))
        #make the state into a list
        temp = np.array(mu.T)[0].tolist()
       
        #evaluate the sensor and the jacobian of the sensor
        h = self.eval_sensor( temp, self.h)
        H = self.eval_sensor(temp, self.H)
        
        #do the EKF 
        cov  = G*(self.sigma)*G.T + self.R
        
        temp = H*cov*(H.T) + self.Q
        
        if np.linalg.det(temp) == 0:
            K = cov* H.T * np.linalg.pinv(temp)
        else:
            K = cov* H.T * np.linalg.inv(temp)
            
        #get the new state
        self.state = mu + K*( z - h )
        #update the error
        size = self.state.shape[0]
        self.sigma = (np.eye(size) - K*H )*cov
        #return the new state
        return self.state
    
        
    
        
        
        
        
        