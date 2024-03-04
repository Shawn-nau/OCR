
import os
import sys
import time
import torch
import numpy as np
from torch import optim
from torch.optim import lr_scheduler 

from .models import MLPRF,MLPQR,MLPSQR
from .data import data_provider
from .loss_fun import QuantileLoss,QuantileLoss_omni,Omni_cost_loss,Likelyhood
from DEEPTS.utils.tools import EarlyStopping

def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))



class Exp_Main(object):
    def __init__(self, args):
        super(Exp_Main, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda')
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device    

    def _build_model(self):
        model_dict = {
            'MLPQR': MLPQR,
            'MLPRF': MLPRF,
            'MLPSQR': MLPSQR,
        }
        model = model_dict[self.args.model](self.args)#.float()
        return model    
    
    def _get_data(self, flag):
        data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        loss_dict = {
            'QuantileLoss': QuantileLoss,
            'QuantileLoss_omni': QuantileLoss_omni,
            'Omni_cost_loss':Omni_cost_loss,
            'Likelyhood_loss':Likelyhood,
        }
        criterion = loss_dict[self.args.loss](self.args,self.device)
        return criterion   
    
    def vali(self,vali_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_num,x_cat, val_targets in vali_loader:

                r,p = self.model(x_num.to(self.device),x_cat.to(self.device))
            
                cost =criterion( r,p, val_targets.to(self.device))
            
                val_loss += cost.item()
            average_val_loss = val_loss / len(vali_loader)
        self.model.train()
        return average_val_loss     

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
       
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=12)
        #scheduler = lr_scheduler.OneCycleLR(optimizer = optimizer,
        #                            steps_per_epoch = train_steps,
        #                            pct_start = self.args.pct_start,
        #                            epochs = self.args.train_epochs,
        #                            max_lr = 0.05)
        #scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= train_steps,eta_min=0.00001)
        epochs = self.args.train_epochs
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0

            for x_num,x_cat, targets in train_loader:

                optimizer.zero_grad()
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device))
                loss = criterion( r,p, targets.to(self.device))                
                loss.backward()
                optimizer.step()
                #scheduler.step()
                total_loss += loss.item()

            average_loss = total_loss / train_steps
            print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss:.4f}")
            val_loss = self.vali(vali_loader, criterion)
            #test_loss = self.vali(test_loader, criterion)

            # Validation
            
            scheduler.step(val_loss)
            early_stopping(val_loss, self.model, path)
            #adjust_learning_rate(optimizer, scheduler, epoch + 1, self.args)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path))
            
            
        return self.model
    
    def predict(self,setting, load=False):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        path = os.path.join(self.args.checkpoints, setting)
        
        if load:
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)
                
                prediction = torch.stack([r,p])
                # Append predictions and true labels to the lists
                predictions.extend(prediction.detach().cpu().numpy().T)
                true_labels.extend(targets.detach().cpu().numpy())
            
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return

    
    def evaluate(self,setting,metric):
        path = os.path.join(self.args.checkpoints, setting)
        folder_path = path+ '/results/' 
        pred = np.load(folder_path  + 'test_pred.npy')
        true = np.load(folder_path  + 'test_true.npy')
        costs = metric.np_loss(pred,true)
        np.save(folder_path + 'test_cost.npy', costs)
        print(setting, costs.sum(1).mean())
        return(costs.mean(0))
  
    
class Exp_likelyhood(Exp_Main):
    
    

    def predict(self,setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        print('loading model')
        path = os.path.join(self.args.checkpoints, setting)
        if load:                
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        nsample = self.args.nsample    
        with torch.no_grad():
            for x_num,x_cat, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_num.to(self.device),x_cat.to(self.device),)               
                dist_k = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,0:1],logits = p[:,0:1])   
                dist_m = torch.distributions.negative_binomial.NegativeBinomial(total_count=r[:,1:2],logits = p[:,1:2]) 
                prediction_k = [dist_k.sample() for i in range(nsample)]  
                prediction_m = [dist_m.sample() for i in range(nsample)] 
                quantile_k = self.args.cs/(self.args.cs+self.args.ch)
                quantile_m = quantile_k
                prediction_k = torch.stack(prediction_k).quantile(quantile_k,dim=0)
                prediction_m = torch.stack(prediction_m).quantile(quantile_m,dim=0)
                prediction = torch.concat([prediction_k,prediction_m],-1)
                # Append predictions and true labels to the lists
                predictions.extend(prediction.cpu().numpy())
                true_labels.extend(targets.cpu().numpy())            
           
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return
    

class Exp_single(Exp_Main):
       
    def predict(self,setting, load=True):
                
        self.model.eval()  # Set the model to evaluation mode
        predictions = []  # To store the model predictions
        true_labels = []  # To store the true labels
        path = os.path.join(self.args.checkpoints, setting)
        
        if load:
            
            best_model_path = os.path.join(path, 'checkpoint.pth')
            print('loading model')
            self.model.load_state_dict(torch.load(best_model_path))
            
        pred_loader = self._get_data(flag='test')  
        
        with torch.no_grad():
            for x_k,x_m, targets in pred_loader:

                # Forward pass to get predictions
                r,p = self.model(x_k.to(self.device),x_m.to(self.device),)
                
                prediction = torch.stack([r[:,0]*targets[:,3].to(self.device),r[:,0]*(1-targets[:,3].to(self.device))])
                # Append predictions and true labels to the lists
                predictions.extend(prediction.detach().cpu().numpy().T)
                true_labels.extend(targets.detach().cpu().numpy())
            
            folder_path = path+ '/results/' 
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path  + 'test_pred.npy', predictions)
            np.save(folder_path  + 'test_true.npy', true_labels)
        return


