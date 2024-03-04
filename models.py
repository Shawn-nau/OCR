import torch
import torch.nn as nn
import torch.optim as optim

from torch.distributions.utils import logits_to_probs

class NegativeBinomial(nn.Module):
    
    def __init__(self, input_size, output_size, logit =True):
        '''
        Negative Binomial Supports Positive Count Data
        Args:
        input_size (int): hidden h_{i,t} column size
        output_size (int): embedding size
        '''
        super(NegativeBinomial, self).__init__()
        self.logit = logit
        self.mu_layer = nn.Linear(input_size, output_size)
        
        self.logit_layer = nn.Linear(input_size, output_size)
 
        self.Softplus = nn.Softplus()
        self.relu = nn.ReLU()
    
    def forward(self, h):
        
        if self.logit:
            p_t = self.logit_layer(h)
        else:
            p_t = torch.sigmoid(self.logit_layer(h))
        #r_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        r_t = self.Softplus(self.mu_layer(h)) #+ 1e-8

        return r_t, p_t
    
   
class MLPRF(nn.Module):
    def __init__(self, config):
        super(MLPRF, self).__init__()
        self.name = 'mlp_like'        
        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.relu2 = nn.ReLU()
        self.out = NegativeBinomial(config.hidden_size2, 2)
        self.embedding_layers = nn.ModuleList([nn.Embedding(i+1, config.dim_embedding) for i in config.max_embedding])

    def forward(self, x_num, x_cat):
        cat_feature = [self.embedding_layers[i](x_cat[:,i]) for i in range(len(self.embedding_layers))]       
        cat_feature = torch.cat(cat_feature, axis=-1)
        x= torch.cat([x_num,cat_feature],axis=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))        
        r,p = self.out(x)
        return r,p   
            

class MLPQR(nn.Module):
    def __init__(self, config):
        super(MLPQR, self).__init__()
        self.name = 'mlp_qr'

        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(config.hidden_size2, 2)
        self.relu3 = nn.Softplus()
        self.embedding_layers = nn.ModuleList([nn.Embedding(i+1, config.dim_embedding) for i in config.max_embedding])

    def forward(self, x_num, x_cat):
        cat_feature = [self.embedding_layers[i](x_cat[:,i]) for i in range(len(self.embedding_layers))]       
        cat_feature = torch.cat(cat_feature, axis=-1)
        x= torch.cat([x_num,cat_feature],axis=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))        
        x = self.relu3(self.fc3(x)) 
        return x[:,0],x[:,1]
    

class MLPSQR(nn.Module):
    def __init__(self, config):
        super(MLPSQR, self).__init__()
        self.name = 'mlp_Sqr'

        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(config.hidden_size2, 1)
        self.relu3 = nn.Softplus()
        self.embedding_layers = nn.ModuleList([nn.Embedding(i+1, config.dim_embedding) for i in config.max_embedding])

    def forward(self, x_num, x_cat):
        cat_feature = [self.embedding_layers[i](x_cat[:,i]) for i in range(len(self.embedding_layers))]       
        cat_feature = torch.cat(cat_feature, axis=-1)
        x= torch.cat([x_num,cat_feature],axis=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))        
        x = self.relu3(self.fc3(x)) 
        return x,x
    
