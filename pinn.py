import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import time
from copy import deepcopy
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
                 layers_size= [3, 20, 20, 20, 20, 20, 20, 20, 20], 
                 out_size = 2,
                 params_list= None):
        
        super(PINN, self).__init__()

        #### Data
        self.x = x
        self.y = y
        self.t = t

        self.u = u
        self.v = v

        
        #### Initializing neural network 

        self.layers = nn.ModuleList()
        #### Initialize parameters (we are going to learn lambda)
        self.lambda_1 = nn.Parameter(torch.randn(1, requires_grad= True))
        self.lambda_2 = nn.Parameter(torch.randn(1, requires_grad= True))
        
        for k in range(len(layers_size) - 1):
            self.layers.append(nn.Linear(layers_size[k], layers_size[k+1]))
            
        # Output layer
        self.out = nn.Linear(layers_size[-1], out_size)
        
        m_aux = 0
        
        for m in self.layers:
            
            if params_list is None:
                nn.init.normal_(m.weight, mean= 0, std= 1/np.sqrt(len(layers_size)))
                nn.init.constant_(m.bias, 0.0)
            else:
                m.weight = params_list[m_aux]
                m.bias = params_list[m_aux + 1]
                m_aux += 1
                
        if params_list is None:
            nn.init.normal_(self.out.weight, mean= 0, std= 1/np.sqrt(len(layers_size)))
            nn.init.constant_(self.out.bias, 0.0)
        else:
            self.out.weight = params_list[-2]
            self.out.bias = params_list[-1]

        #self.model = nn.ModuleDict()

        #self.optimizer = optim.SGD(params= self.model.parameters(), 
        #                           lr= 0.001, 
        #                           weight_decay= 0.0)


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
        
        #X = torch.cat([x, y, t], dim=1)   #  What about velocity field data?
        X = torch.stack((x, y, t), 1)

        # Parametrize [psi, p] as neural network
        psi_p = self.forward(X)
        #print(psi_p)

        ### Check this on a notebook
        psi = psi_p[:, 0:1]
        p = psi_p[:, 1:2]

        ### Gradients

        # Assert requires_grad = True
        x.requires_grad_(True)
        y.requires_grad_(True)
        t.requires_grad_(True) 
        
        u = torch.sum(self.get_gradient(psi, y), dim= 1)  
        u.requires_grad_(True)
        u_t = torch.sum(self.get_gradient(u, t), dim= 1)
        u_x = torch.sum(self.get_gradient(u, x), dim= 1)
        u_x.requires_grad_(True)
        u_y = torch.sum(self.get_gradient(u, y), dim= 1)
        u_y.requires_grad_(True)      
        u_xx = torch.sum(self.get_gradient(u_x, x), dim= 1)
        u_yy = torch.sum(self.get_gradient(u_y, y), dim= 1)

        v = - torch.sum(self.get_gradient(psi, x), dim= 1)
        v.requires_grad_(True)
        v_t = torch.sum(self.get_gradient(v, t), dim= 1)
        v_x = torch.sum(self.get_gradient(v, x), dim= 1)
        v_x.requires_grad_(True)
        v_y = torch.sum(self.get_gradient(v, y), dim= 1)
        v_y.requires_grad_(True)
        v_xx = torch.sum(self.get_gradient(v_x, x), dim= 1)
        v_yy = torch.sum(self.get_gradient(v_y, y), dim= 1)

        p.requires_grad_(True)
        p_x = torch.sum(self.get_gradient(p, x), dim= 1)
        p_y = torch.sum(self.get_gradient(p, y), dim= 1)

        ### PINN

        f_u = u_t + lambda1*(u*u_x + v*u_y) + p_x - lambda2*(u_xx + u_yy)
        f_v = v_t + lambda1*(u*v_x + v*v_y) + p_y - lambda2*(v_xx + v_yy)

        return u, v, p, f_u, f_v


    def loss(self, u, v, u_hat, v_hat, f_u, f_v):

        error = torch.mean(torch.square(u - u_hat)) + torch.mean(torch.square(v - v_hat))
        regul = torch.mean(torch.square(f_u)) + torch.mean(torch.square(f_v))

        return error + regul


    def get_gradient(self, f, x):
        """ computes gradient of tensor f with respect to tensor x """
        assert x.requires_grad

        x_shape = x.shape
        f_shape = f.shape
        f = f.view(-1)

        x_grads = []
        for f_val in f:
            if x.grad is not None:
                x.grad.data.zero_()
            f_val.backward(retain_graph=True)
            if x.grad is not None:
                x_grads.append(deepcopy(x.grad.data))
            else:
                # in case f isn't a function of x
                x_grads.append(torch.zeros(x.shape).to(x))
        output_shape = list(f_shape) + list(x_shape)
        return torch.cat((x_grads)).view(output_shape)


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
                                                        t1-t0))