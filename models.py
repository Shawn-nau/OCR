import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.utils import logits_to_probs


class DifferentiableDiscretization(nn.Module):
    """Differentiable approximation to integer constraints"""
    def __init__(self, temperature=1.0):
        super(DifferentiableDiscretization, self).__init__()
        self.temperature = temperature
        
    def forward(self, x):
        # Create a range of integers around each value
        base_values = torch.floor(x)
        offsets = torch.tensor([0, 1], device=x.device).unsqueeze(0)
        candidates = base_values.unsqueeze(-1) + offsets
        
        # Calculate probability distribution over integers
        logits = -self.temperature * torch.abs(x.unsqueeze(-1) - candidates)
        probs = torch.softmax(logits, dim=-1)
        
        # Expected value (soft discretization)
        discretized = torch.sum(probs * candidates, dim=-1)
        
        # During training, return soft discretized values
        # During inference, return hard integer values
        if self.training:
            return discretized
        else:
            return torch.round(x)

class MIPLayer(nn.Module):
    """Differentiable MIP layer for integer constraints"""
    def __init__(self, lambda_int=10.0):
        super(MIPLayer, self).__init__()
        self.lambda_int = lambda_int
        
    def forward(self, x):
        # During training: add penalty for non-integer values
        if self.training:
            fractional_part = torch.abs(x - torch.round(x))
            # This penalty will push values toward integers through gradient descent
            self.integer_penalty = self.lambda_int * torch.mean(fractional_part)
            return x
        else:
            return torch.round(x)
            
    def get_integer_penalty(self):
        return self.integer_penalty if hasattr(self, 'integer_penalty') else 0.0



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
 
        #self.Softplus = nn.functional.softplus()
        self.relu = nn.ReLU()
    
    def forward(self, h):
        
        if self.logit:
            p_t = self.logit_layer(h)
        else:
            p_t = torch.sigmoid(self.logit_layer(h))
        #r_t = torch.log(1 + torch.exp(self.mu_layer(h)))
        r_t = F.softplus(self.mu_layer(h))

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

class MLPJoint(nn.Module):
    def __init__(self, config):
        super(MLPJoint, self).__init__()
        self.name = 'mlp_like'        
        self.fc1 = nn.Linear(config.input_size, config.hidden_size1)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(config.hidden_size1, config.hidden_size2)
        self.relu2 = nn.ReLU()
        self.out = NegativeBinomial(config.hidden_size2, 2)
        self.fc3 = nn.Sequential(nn.Linear(config.input_size-14, config.dim_embedding*3*2),
                                 nn.ReLU(),
                                 nn.Linear(config.dim_embedding*3*2, 1))
                                 
        self.embedding_layers = nn.ModuleList([nn.Embedding(i+1, config.dim_embedding) for i in config.max_embedding])
        self.embedding_sku = nn.Embedding(735, 10) 
    

    def forward(self, x_num, x_cat):
        cat_feature = [self.embedding_layers[i](x_cat[:,i]) for i in range(len(self.embedding_layers))]       
        cat_feature = torch.cat(cat_feature, axis=-1)
        x= torch.cat([x_num,cat_feature],axis=-1)
        #sku_embedding = self.embedding_sku(x_cat[:,1])
        rho = self.fc3(cat_feature)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))        
        r,p = self.out(x)
        p = torch.cat([p[:,0:1],rho,p[:,1:2]], axis=-1)
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
        self.mip_layer = MIPLayer(lambda_int= config.lambda_int)

    def forward(self, x_num, x_cat):
        cat_feature = [self.embedding_layers[i](x_cat[:,i]) for i in range(len(self.embedding_layers))]       
        cat_feature = torch.cat(cat_feature, axis=-1)
        x= torch.cat([x_num,cat_feature],axis=-1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))        
        x = self.relu3(self.fc3(x)) 
        #x = self.mip_layer(x)
        return x[:,0],x[:,1]
    
    def get_integer_penalty(self):
        return self.mip_layer.get_integer_penalty()
    

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
    
