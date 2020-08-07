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
                 s, 
                 i, 
                 t, 
                 layers_size= [3, 20, 20], 
                 out_size = 2,
                 params_list= None):
        
        super(PINN, self).__init__()

        #### Data
        self.s = s
        self.i = i
        self.t = t
        
        #### Initializing neural network 

        self.layers = nn.ModuleList()
        #### Initialize parameters (we are going to learn lambda)
        self.beta = nn.Parameter(torch.zeros(1, requires_grad= True))
        self.gamma = nn.Parameter(torch.zeros(1, requires_grad= True))
        
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

    #### Derivative

    def delta(self, x):
        list_ = []
        list_.append(0)
        for j in range(len(x) - 1):
            list_.append(x[j+1].item() - x[j].item())
        return torch.Tensor(list_)  


    #### Net NS

    def net(self, x, y, t):

        beta = self.beta
        gamma = self.gamma
        
        X = torch.stack((x, y, t), 1)

        # Parametrize [psi, p] as neural network
        s_i_model = self.forward(X)
        #print(psi_p)

        ### Check this on a notebook
        s_model = s_i_model[:, 0:1]
        i_model = s_i_model[:, 1:2]

        dsdt = self.delta(x)
        didt = self.delta(y)


        ### PINN

        f_s = dsdt + beta*s_model*i_model
        f_i = didt - beta*s_model*i_model + gamma*i_model

        return s_model, i_model, f_s, f_i


    def loss(self, s, i, s_hat, i_hat, f_s, f_i):

        error = torch.mean(torch.square(s - s_hat)) + torch.mean(torch.square(i - i_hat))
        regul = torch.mean(torch.square(f_s)) + torch.mean(torch.square(f_i))

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

            s_model, i_model, f_s, f_i = self.net(self.s, self.i, self.t)
            loss_ = self.loss(self.s, self.i, s_model, i_model, f_s, f_i)
            loss_print  = loss_

            self.optimizer.zero_grad()   # Clear gradients for the next mini-batches
            loss_.backward()         # Backpropagation, compute gradients
            self.optimizer.step()

            t1 = time.time()


            ### Training status
            print('Epoch %d, Loss= %.10f, Time= %.4f' % (epoch, 
                                                        loss_print,
                                                        t1-t0))