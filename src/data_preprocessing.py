import pandas as pd
import numpy as np
import torch
import random
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

class DataHandler:
    def __init__(self, args, mode='rec'):
        self.args = args
        self.data_name = args.dataset
        self.data = pd.read_csv(str(Path.cwd()) + f'/processed_data/{args.dataset}/interaction.csv')
        self.prepare_train_test(args, mode)

    def prepare_train_test(self, args, mode):
        train, test = train_test_split(self.data.values, test_size=0.2, random_state=16)
        train = pd.DataFrame(train, columns=self.data.columns)
        test = pd.DataFrame(test, columns=self.data.columns)

        train_user_ids = train['user_id'].unique()
        train_item_ids = train['item_id'].unique()
        test = test[(test['user_id'].isin(train_user_ids)) & (test['item_id'].isin(train_item_ids))]

        le_user = LabelEncoder()
        le_item = LabelEncoder()
        train['user_id'] = le_user.fit_transform(train['user_id'].values)
        train['item_id'] = le_item.fit_transform(train['item_id'].values)
        test['user_id'] = le_user.transform(test['user_id'].values)
        test['item_id'] = le_item.transform(test['item_id'].values)

        self.n_users = train['user_id'].nunique()
        self.n_items = train['item_id'].nunique()

        self.train_df = train
        self.test_df = test

        self.train_group_user = train.groupby('user_id')['item_id'].agg(list).reset_index(name='item_ids')
        self.train_group_item = train.groupby('item_id')['user_id'].agg(list).reset_index(name='user_ids')
        self.test_group = test.groupby('user_id')['item_id'].agg(list).reset_index(name='item_ids')

        if mode == 'rec':
            batch_train = args.batch_train if args.batch_train != -1 else len(train)
            model_name = getattr(args, 'model', getattr(args, 'rec_model', None))
            user_only = model_name in ['VAE', 'DiffRec']
            self.train_dataset = TrainRecDataset(train, user_only=user_only)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_train, shuffle=True, num_workers=0)

            batch_test = args.batch_test if args.batch_test != -1 else len(test)
            self.test_dataset = TestRecDataset(self.test_group)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_test, shuffle=False, num_workers=0)
        
        elif mode == 'exp':
            random.seed(16)
            user_exp = list(self.train_group_user['user_id'])
            random.shuffle(user_exp)
            num_test = int(args.test_users) if hasattr(args, 'test_users') and args.test_users is not None else 500
            num_val = int(0.1*(len(user_exp) - num_test))
            user_test_exp = user_exp[:num_test]
            user_val_exp = user_exp[num_test:num_test+num_val]
            user_train_exp = user_exp[num_test+num_val:]

            batch_train = args.batch_train if args.batch_train != -1 else len(train)
            self.train_dataset = TrainExpDataset(user_train_exp)
            self.train_loader = DataLoader(self.train_dataset, batch_size=batch_train, shuffle=True, num_workers=0)

            batch_test = args.batch_test if args.batch_test != -1 else len(test)
            self.val_dataset = ValExpDataset(user_val_exp)
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_test, shuffle=False, num_workers=0)
            self.test_dataset = TestExpDataset(user_test_exp)
            self.test_loader = DataLoader(self.test_dataset, batch_size=batch_test, shuffle=False, num_workers=0)
        
        else:
            raise ValueError(f"Mode {mode} not supported. Choose 'rec' or 'exp'.")


class TrainRecDataset(Dataset):
    def __init__(self, data, user_only=False):
        super(TrainRecDataset, self).__init__()
        self.user_only = user_only
        if user_only:
            self.rows = data['user_id'].unique()
            self.cols = np.zeros(len(self.rows)).astype(np.int32)
        else:
            self.rows = data['user_id'].values
            self.cols = data['item_id'].values
            self.dokmat = set(zip(self.rows, self.cols))
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negative_sampling(self, n_items):
        if self.user_only:
            return
        self.negs = np.random.randint(0, n_items, size=len(self.rows))
        for i in range(len(self.rows)):
            u = self.rows[i]
            if (u, self.negs[i]) in self.dokmat:
                while True:
                    iNeg = np.random.randint(n_items)
                    if (u, iNeg) not in self.dokmat:
                        self.negs[i] = iNeg
                        break

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]
        
class TestRecDataset(Dataset):
    def __init__(self, data):
        super(TestRecDataset, self).__init__()
        self.tst_users = data['user_id'].values
        self.gt_items = data['item_ids'].values
        self.max_len = max([len(items) for items in self.gt_items])

    def __len__(self):
        return len(self.tst_users)
    
    def __getitem__(self, idx):
        return self.tst_users[idx], torch.tensor(self.gt_items[idx] + [-1] * (self.max_len - len(self.gt_items[idx])))  # Padding with -1
    
class TrainExpDataset(Dataset):
    def __init__(self, data):

        super(TrainExpDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
class ValExpDataset(Dataset):
    def __init__(self, data):
        super(ValExpDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class TestExpDataset(Dataset):
    def __init__(self, data):
        super(TestExpDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]