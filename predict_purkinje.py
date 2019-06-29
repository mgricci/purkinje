import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import ipdb


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_layers=1):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
       
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        
    def init_hidden(self): 
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim), torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inpt):
        lstm_out, self.hidden = self.lstm(inpt.view(len(inpt), self.batch_size, -1))
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

def split_sequence_batch(data, batch_size, CS_onset, US_onset, p=3, f=1):
    batch = []
    target = []
    num_trials = data.shape[0]
    trial_duration = data.shape[1]
    #CS = np.zeros(data.shape)
    #US = np.zeros(data.shape)
    #CS[:,CS_onset] = 1.0
    #US[:,US_onset] = 1.0
    CSUS = np.zeros(data.shape)
    CSUS[:,CS_onset:US_onset] = 1.0
    
    for b in range(batch_size): 
       trial = np.random.randint(0,num_trials) 
       t = np.random.randint(p, trial_duration - f)
       past_window = np.expand_dims(data[trial,t-p:t],1)
       future_window = data[trial,t:t+f]
       #CS_window = np.expand_dims(CS[trial,t-p:t],1)
       #US_window = np.expand_dims(US[trial,t-p:t],1)
       CSUS_window = np.expand_dims(CSUS[trial,t-p:t],axis=1)
       batch.append(np.concatenate((past_window, CSUS_window, trial*np.ones((p,1))),axis=1))
       target.append(future_window)
    return torch.tensor(batch).transpose(1,0), torch.tensor(target).squeeze().double()

def make_raster(data, CS_onset=None, US_onset=None): 
    num_trials = data.shape[0]
    trials = range(num_trials)
    duration = data.shape[1]

    fig, ax = plt.subplots(figsize=(20,16))
    plt.hlines(trials, np.zeros(num_trials), duration*np.ones(num_trials), alpha=.25)
    ax.set_aspect(50)

    spike_times = np.nonzero(data)
    plt.scatter(spike_times[1], spike_times[0], s=2, color='k')
    plt.savefig('/home/matt/figs/purkinje/raster.png')
    
def generate_data(base_p=.5, num_trials=100, trial_duration=2*1000, CS_onset=1*1000, US_onset=int(1.3*1000), alpha=1.0, beta=0.0, visualize=False):
    start_CS = (np.random.rand(num_trials, CS_onset) < base_p)*1
    US_end = (np.random.rand(num_trials, trial_duration - US_onset) < base_p)*1

    CS_US_p = base_p * np.ones((num_trials, US_onset - CS_onset))
    exp_centered_steps = np.arange(num_trials) - num_trials / 2.0
    p_mask = np.expand_dims(1 / (1+ np.exp(((alpha/num_trials)*exp_centered_steps - beta))),1)
    CS_US_p*= p_mask

    CS_US = (np.random.rand(num_trials, US_onset - CS_onset) < CS_US_p)*1
    
    data = np.concatenate([start_CS, CS_US, US_end],axis=1)

    if visualize:
        make_raster(data)

    return data
    
def train(model, data, CS_onset, US_onset, past, future, num_batches=1000, lr=1e-4, batch_size=32, show_every=25):

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    hist = []
    for b in tqdm(range(num_batches)):
        model.zero_grad()
        batch, target = split_sequence_batch(data,batch_size, CS_onset, US_onset, p=past, f=future)
        y_pred = model(batch.cuda())
        loss = loss_fn(y_pred, target.cuda())
        if b%show_every == 0: 
            print('Loss: {}'.format(loss.data.cpu().numpy()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

if __name__=='__main__':
    # TODO GPU
    trial_duration = 1000
    CS_onset = 500
    US_onset = 800
    past = 10
    future = 1
    data = generate_data(base_p=.05, trial_duration=trial_duration, CS_onset=CS_onset, US_onset=US_onset,alpha=4.0, visualize=True)
    model = LSTM(3,64,batch_size=128,output_dim=future,num_layers=1).double().cuda()
    train(model, data, CS_onset, US_onset, past, future, lr=1e-5,num_batches=10000, show_every=100)
    print('Done')
