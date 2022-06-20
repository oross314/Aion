import numpy as np
import torch
import torch.nn as nn
from torch import optim

cut = 18
loc = 'nonzero/nonzero'
data, truth = np.load(loc +'data.npy'), np.load(loc +'truth.npy')[:, :cut]
Cinv = np.load(loc + 'Cinv.npy')[:, :cut, :cut]
bins, numt = np.load(loc + 'bins.npy'), np.load(loc + 'numt.npy')

class Net(nn.Module):
    def __init__(self, *args):
        super(Net, self).__init__()
        size, lins,  dropout, self.activation = args  
        
        ## Create List of Linear Layers
        self.lfcs = nn.ModuleList()
        if lins:
            lins = np.append([size,], lins)
            for i in range(len(lins)-1):
                self.lfcs.append( nn.Linear(lins[i], lins[i+1]))
  
        if dropout:
            self.drops = nn.ModuleList()
            for i in range(len(lins)):
                self.drops.append(nn.Dropout(dropout))
        else: self.drops = []    

    def forward(self, output):                                
        #Run linear layers and activations
        for i in range(len(self.lfcs)):
            output = self.lfcs[i](output)
            if self.activation:
                output = self.activation(output)
            if self.drops:
                output = self.drops[i](output)
        return output

    
class Model:
    def __init__(self, *args, **kwargs):
         self.num = args
        
        self.lins, self.activation, optimizer,  self.batch_size, self.lr, = [75, ] * 4, nn.PReLU(), optim.Adam, 40, 1e-3
        self.indata, self.truth, self.C = data, truth, Cinv
        
        keys = ('lr_decay', 'max_epochs', 'saving', 'run_num', 'new_tar', 'lr_min','dropout',  'print_ev', 'train_fac')
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
        
        self.lr_init = self.lr      
        self.train_fac = kwargs.get('train_fac', 4/5)
        self.data = self.data_prep()                
        self.dropout = kwargs.get('dropout', 0)
        lsize = self.indata.shape[-1]
    
        self.model = Net(lsize, self.lins, self.dropout,  self.activation).double()
        
        self.lr_decay = kwargs.get('lr_decay', .998)
        self.lr_min = kwargs.get('lr_min', 1e-6)
        self.optimizer = optimizer(self.model.parameters(), lr = self.lr)
        self.max_epochs = kwargs.get('max_epochs', 2400)   
        self.trainerr, self.testerr, self.err, self.epoch, self.loc, self.save_file = [], [], 0, 0, 0, None #just inits
        
        self.saving = kwargs.get('saving', True)
        self.new_tar = kwargs.get('new_tar', False)
        self.run_num = kwargs.get('run_num', None)
        self.check()     
            
        
                
    def run(self, **kwargs):
        if len(self.trainerr)!=len(self.testerr):
            self.trainerr = self.trainerr[:-1]
            
        keys = ('lr', 'lr_decay', 'epochs', 'saving', 'batch_size', 'lr_min', 'training', 'print_ev')
        for kwarg in kwargs.keys():
            if kwarg not in keys:
                raise Exception(kwarg + ' is not a valid key. Valid keys: ', keys )
                
        train, traintruth, traincov, trainbins, test, testtruth, testcov, testbins = self.data
        self.testtruth = testtruth
        self.max_epochs = kwargs.get('epochs', self.max_epochs)
        self.lr = kwargs.get('lr', self.lr)
        self.saving = kwargs.get('saving', self.saving)
        self.lr_decay = kwargs.get('lr_decay', self.lr_decay)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.lr_min = kwargs.get('lr_min', self.lr_min)
        training = kwargs.get('training', True)
        e0 = self.epoch
        
        while self.epoch < (e0 + self.max_epochs) :
            shuffle = torch.randperm(traintruth.shape[0]) #shuffle training set
            self.lr = max(self.lr * self.lr_decay, self.lr_min)  #lr decay
            
            for i in range(int(np.floor(traintruth.shape[0]/self.batch_size))):
                self.optimizer.param_groups[0]['lr'] = self.lr
                where = shuffle[i * self.batch_size:(i + 1) * self.batch_size] #take batch of training set

                self.output = self.model(train[where])
                
                self.truetrain = traintruth[where].detach().numpy()
                loss = torch.mean(self.cost(torch.squeeze(self.output), traintruth[where], traincov[where], trainbins[where]))
                if training:
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph = True)
                    self.optimizer.step()
                         
            self.trainerr.append(loss.detach().numpy() )
            self.testout = self.model(test)
            
            self.err = self.cost(torch.squeeze(self.testout), testtruth, testcov, testbins ).mean().detach().numpy() 
            
            self.testerr.append(self.err)
                
            if self.saving:
                self.save()
            self.epoch += 1
            
    def cost(prediction, truth, Cinv, bins):
        D = torch.abs(10**prediction - 10**truth )
        loss = torch.matmul(torch.matmul(D.unsqueeze(1), Cinv), D.unsqueeze(2)).squeeze().squeeze()
        return loss/bins

    def save(self): 
        self.save_file = 'tars/' + str(self.num)+'.tar'
        torch.save(self.model.state_dict(), self.save_file)
        vals = pickle.load(open('newvalues.p', 'rb'))
        df = self.params()
        if self.loc in vals.index.tolist():
            vals.loc[self.loc] = df.loc[self.loc]
        else:
            vals = vals.append( df)
            
        pickle.dump(vals, open('newvalues.p', 'wb'))
        
  
    def data_prep(self, shuffle = False):
        bins = torch.sum((10**torch.tensor(self.truth) > 3).int(), axis = 1).unsqueeze(1)
        
        data = torch.tensor(self.indata).double()
        C = torch.tensor(self.C)
        truth = torch.tensor(self.truth)
        
        start = int(np.sum(numt[:self.num]))
        stop = numt[self.num] + start
        test, testtruth, testcov, testbins = data[start:stop], truth[start:stop], C[start:stop], bins[start:stop]

        train = torch.vstack((data[:start], data[stop:]))
        traintruth = torch.vstack((truth[:start], truth[stop:]))
        traincov = torch.vstack((C[:start], C[stop:]))
        trainbins = torch.vstack((bins[:start], bins[stop:]))  
        
        return train, traintruth, traincov, trainbins, test, testtruth, testcov, testbins