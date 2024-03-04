
import pandas as pd
import numpy as np
import  warnings

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore')

class Dataset_Omni_mlp(Dataset):
    def __init__(self, root_path, data_path='processed_df.csv',  flag='train',scale=True):
        

        # init
        assert flag in ['train', 'test', 'val']

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.df = self.__read_data__()
        
        self.num_var = ['price',
            'y_k_rolling_mean_1_3',
            'y_k_rolling_mean_1_7',
            'y_k_rolling_mean_1_14',
            'y_k_rolling_mean_1_28',
            'y_m_rolling_mean_1_3',
            'y_m_rolling_mean_1_7',
            'y_m_rolling_mean_1_14',
            'y_m_rolling_mean_1_28',
            'min_temperature',
            'max_temperature',
            'discount_off_x',
            'discount_off_y',
            'salable_status', ]

        self.cat_var = ['store_id', 'sku_id',  'weather_type', 'item_first_cate_cd', 'item_second_cate_cd',
            'item_third_cate_cd', 'brand_code', 'tm_w', 'tm_m', 'tm_dw']
        
        self.y_label = ['y_k','y_m','original_price','ratio']
        type_map = {'train': self.tr_mask, 'val': self.vl_mask, 'test': self.ts_mask}
        mask = type_map[flag]
        
        self.x_num = torch.FloatTensor(self.df.loc[mask,self.num_var].reset_index(drop=True).values)
        self.x_cat = torch.LongTensor(self.df.loc[mask,self.cat_var].reset_index(drop=True).values)
        self.labels = torch.FloatTensor(self.df.loc[mask,self.y_label].reset_index(drop=True).values)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.root_path + self.data_path,parse_dates=['date'])
        df = df.loc[df['store_id']<=5]
        
        validation = {
            'validation' :  pd.to_datetime(['2023-07-15', '2023-07-31']),
            'test' :  pd.to_datetime(['2023-08-01', '2023-08-31']),
        }
        cv = 'validation'
        
        
        df['price'] = df['original_price']
        df['discount'] = (df['discount_off_y'] + df['discount_off_x'])/2
        
        self.tr_mask = df['date'] < validation[cv][0]
        self.vl_mask = (df['date'] >= validation[cv][0]) & (df['date'] <= validation[cv][1])
        self.ts_mask = (df['date'] > validation[cv][1]) 
        
        df['y_k'] = np.round(df['y_k'])
        df['y_m'] = np.round(df['y_m'])
        df['ratio'] = df['y_k_rolling_mean_1_28']/(df['y_k_rolling_mean_1_28']+df['y_m_rolling_mean_1_28']+1e-9)
        
        num_var = ['price','y_k_rolling_mean_1_3', 'y_k_rolling_mean_1_7',
       'y_k_rolling_mean_1_14', 'y_k_rolling_mean_1_28',
       'y_m_rolling_mean_1_3', 'y_m_rolling_mean_1_7', 'y_m_rolling_mean_1_14',
       'y_m_rolling_mean_1_28','min_temperature',
       'max_temperature','discount_off_x', 'discount_off_y']
        
        if self.scale:
            self.scaler.fit(df.loc[self.tr_mask,num_var])
            df[num_var] = self.scaler.transform(df[num_var])    
        return df  
    
    def __len__(self):
            return len(self.labels)

    def __getitem__(self, index):
        # Directly return preprocessed tensors
        return self.x_num[index], self.x_cat[index], self.labels[index] 


def data_provider(args, flag):
    if flag == 'train':
        shuffle_flag = True
        batch_size = args.batch_size
    else:
        shuffle_flag = False
        batch_size = args.batch_size

    data_set = Dataset_Omni_mlp(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,)

    return data_loader