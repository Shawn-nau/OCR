import argparse
import sys
import pandas as pd
import torch
#from exp.exp_main import Exp_Main

from experiment import Exp_Main,Exp_likelyhood,Exp_Jointlikelyhood
from loss_fun import BiC_cost_detail
import random
import numpy as np

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Omni retail replenishment')

    # random seed
    parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

    # basic config
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model', type=str,  default='MLPRQ',
                        help='model name, options: [MLPRQ, MLPRF]')

    # data loader
    #parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./Data/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='processed_df.csv', help='data file')

    parser.add_argument('--checkpoints', type=str, default='./Exp_1', help='location of model checkpoints')  
   
    # MLP
    parser.add_argument('--dim_embedding', type=int, default=3, help='input embedding')
    parser.add_argument('--input_size', type=int, default=44, help='input_size')    
    parser.add_argument('--hidden_size1', type=int, default=88, help='hidden_size1')
    parser.add_argument('--hidden_size2', type=int, default=44, help='hidden_size2')    
    parser.add_argument('--max_embedding', type=list, default=[5, 735, 303, 2, 7, 44, 97, 52, 12, 6], help='max_embedding')

    # optimization
    
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=5120, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=20, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='Omni_cost_loss', help='loss function')
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')

    parser.add_argument('--nsample', type=int, default=200, help='samples in distribution')
    

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    
    # exp_id
    parser.add_argument('--s', type=int, default=2, help='no. of exp setting')

    args = parser.parse_args()
    
    #print(args)
    
    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    print('Args in experiment:')
    print(args)   
    
    exp_settings = pd.concat([pd.Series(np.arange(1,6)/10,index = np.arange(5),name='cs'),
    pd.Series(np.arange(1,6)/10,index = np.arange(5,10),name='ch'),
    pd.Series(np.arange(5,20,3)/10,index = np.arange(10,15),name='cm2k'),
    pd.Series(np.arange(10,25,3)/10,index = np.arange(15,20),name='ck2m'),
    ],axis = 1)

    exp_settings['cs'] = exp_settings['cs'].fillna(0.3)
    exp_settings['ch'] = exp_settings['ch'].fillna(0.2)
    exp_settings['cm2k'] = exp_settings['cm2k'].fillna(1.1)
    exp_settings['ck2m'] = exp_settings['ck2m'].fillna(1.6)
    
    
    args.cs,args.ch,args.cm2k,args.ck2m = exp_settings.iloc[args.s,:] 
    cs,ch,cm2k,ck2m = exp_settings.iloc[2,:] 
    metrics = BiC_cost_detail(args)
    
    Exp = Exp_Jointlikelyhood
    
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_cs{}_ch{}_cm2k{}_ck2m{}_embedding{}_h1{}_lr{}_itr{}'.format(
            args.model,
            args.loss,
            args.cs,
            args.ch,
            args.cm2k,
            args.ck2m,
            args.dim_embedding,
            args.hidden_size1,
            args.learning_rate,                
            ii)
        
        model_setting = '{}_{}_cs{}_ch{}_cm2k{}_ck2m{}_embedding{}_h1{}_lr{}_itr{}'.format(
            args.model,
            args.loss,
            cs,
            ch,
            cm2k,
            ck2m,
            args.dim_embedding,
            args.hidden_size1,
            args.learning_rate,                
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))        
        exp.optimize(setting,model_setting,True)
        #exp.optimize_quantile(setting,model_setting,True)        
        torch.cuda.empty_cache()


    

