import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

    def net_jointly(self, x, y, t):

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
        


