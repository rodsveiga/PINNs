import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
#import torch.nn.functional as F

class PINN(nn.Module):
    '''
    Neural Network Class
    net_layer: list with the number of neurons for each network layer, [n_imput, ..., n_output]
    '''
    def __init__(self, 
                 x, 
                 y, 
                 t, 
                 u,
                 v,
                 layers_size= [10000, 512, 256, 64], 
                 out_size = 29,
                 params_list= None):
        
        super(PINN, self).__init__()

        #### Data
        self.x = x
        self.y = y
        self.t = t

        self.u = u
        self.v = v


        #### Initialize parameters
        self.lambda_1 = torch.zeros(1, requires_grad= True)
        self.lambda_2 = torch.zeros(1, requires_grad= True)

        #### Initializing training sets

        # params (iterable) – iterable of parameters to optimize or dicts defining parameter groups
        # Parameters need to be specified as collections that have a deterministic ordering that is consistent between runs.
        # Examples of objects that don’t satisfy those properties are sets and iterators over values of dictionaries.

        self.optimizer = optim.SGD(params= TODO, lr= 0.001, weight_decay= 0.0)



        #### Initializing neural network 

        self.layers = nn.ModuleList()
        
        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k+1]))
            
        # Output layer
        self.out = nn.Linear(layers_size[-1], out_size)
        
        m_aux = 0
        
        for m in self.layers:
            
            if params_list is None:
                nn.init.normal_(m.weight, mean= 0, std= 1/np.sqrt(100*len(layers_size)))
                nn.init.constant_(m.bias, 0.0)
            else:
                m.weight = params_list[m_aux]
                m.bias = params_list[m_aux + 1]
                m_aux += 1
                
        if params_list is None:
            nn.init.normal_(self.out.weight, mean= 0, std= 1/np.sqrt(100*len(layers_size)))
            nn.init.constant_(self.out.bias, 0.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]


    #### Forward pass
       
    def forward(self, x):

        for layer in self.layers:

            # Activation function
            x = torch.tanh(layer(x))
        
        # Last layer: we could choose a different functionsoftmax
        #output= F.softmax(self.out(x), dim=1)
        output= self.out(x)
        
        return output


    #### Net NS

    def net(self, x, y, t):

        lambda1 = self.lambda_1
        lambda2 = self.lambda_2
        
        X = torch.cat([x, y, t], dim=1)

        psi_p = self.forward(X)

        ### Check this on a notebook
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]

        ### Gradients

        u = torch.autograd.grad(psi, y)
        v = -torch.autograd.grad(psi, x)

        u_t = torch.autograd.grad(u, t)
        u_x = torch.autograd.grad(u, x)
        u_y = torch.autograd.grad(u, y)
        u_xx = torch.autograd.grad(u_x, x)
        u_yy = torch.autograd.grad(u_y, y)

        v_t = torch.autograd.grad(v, t)
        v_x = torch.autograd.grad(v, x)
        v_y = torch.autograd.grad(v, y)
        v_xx = torch.autograd.grad(v_x, x)
        v_yy = torch.autograd.grad(v_y, y)

        p_x = torch.autograd.grad(p, x)
        p_y = torch.autograd.grad(p, y)

        ### PINN

        f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
        f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

        return u, v, p, f_u, f_v

    def loss(self, u, v, u_hat, v_hat, f_u, f_v):

        error = torch.mean(torch.square(u - u_hat)) + torch.mean(torch.square(v - v_hat))
        regul = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))

        return error + regul

    def train(self, epochs):

        t0 = time.time()
        
        for epoch in range(epochs):

            u_hat, v_hat, p_hat, f_u, f_v = self.net(self.x, self.y, self.t)
            loss_ = self.loss(self.u, self.v, u_hat, v_hat, f_u, f_v)
            loss_print  = loss_

            self.optimizer.zero_grad()   # Clear gradients for the next mini-batches
            loss_.backward()         # Backpropagation, compute gradients
            self.optimizer.step()

            t1 = time.time()


            ### Training status
            print('Epoch %d, Loss= %.10f, Time= %.4f' % (epoch,
                                                         loss_print,
                                                         t1-t0)



    

        


