import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
plt.style.use('ggplot')
import numpy as np
from scipy.misc import imresize
import torch
from torch import nn
from tqdm import tqdm
import ipdb

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1, num_lstm_layers=1, num_fc_layers=1, num_fc_features=[]):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_lstm_layers = num_lstm_layers
        self.num_fc_layers = num_fc_layers
       
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_lstm_layers)
        if self.num_fc_layers = 1:
            self.fc = nn.Linear((self.hidden_dim, self.output_dim))
        self.fc = nn.ModuleList([])
        
    def init_hidden(self, batch_size): 
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim), torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, inpt):
        batch_size = inpt.shape[1]
        lstm_out, self.hidden = self.lstm(inpt.view(len(inpt), batch_size, -1))
        y_pred = torch.sigmoid(self.linear(lstm_out[-1].view(batch_size, -1)))
        return y_pred

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

def split_sequence_eval(data, t, CS_onset, US_onset, p=3, f=1):
    num_trials = data.shape[0]
    trial_duration = data.shape[1]
    CSUS = np.zeros(data.shape)
    CSUS[:,CS_onset:US_onset] = 1.0
    past_window = np.expand_dims(data[:,t-p:t],-1)
    future_window = np.expand_dims(data[:,t:t+f], -1)
    CSUS_window = np.expand_dims(CSUS[:,t-p:t], -1)
    trials = np.expand_dims(np.expand_dims(np.arange(num_trials),1)*np.ones((num_trials, p)),-1)
    batch = np.concatenate((past_window, CSUS_window, trials), axis=2).transpose(1,0,2)
    target = future_window
    return torch.tensor(batch), torch.tensor(target).squeeze().double()

def make_raster(data, name,CS_onset=None, US_onset=None): 
    num_trials = data.shape[0]
    trials = range(num_trials)
    duration = data.shape[1]
    fig, ax = plt.subplots(figsize=(20,16))
    plt.hlines(trials, np.zeros(num_trials), duration*np.ones(num_trials), alpha=.25)
    ax.set_aspect(50)

    spike_times = np.nonzero(data)
    plt.scatter(spike_times[1], spike_times[0], s=2, color='k')
    plt.savefig('/home/matt/figs/purkinje/{}.png'.format(name))
    plt.close()
    
def generate_data(base_p=.5, num_trials=100, trial_duration=2*1000, CS_onset=1*1000, US_onset=int(1.3*1000), alpha=1.0, beta=0.0, visualize=False):
    start_CS = (np.random.rand(num_trials, CS_onset) < base_p)*1
    US_end = (np.random.rand(num_trials, trial_duration - US_onset) < base_p)*1

    CS_US_p = base_p * np.ones((num_trials, US_onset - CS_onset))
    exp_centered_steps = np.arange(num_trials) - num_trials / 2.0
    p_mask = np.expand_dims(1 / (1+ np.exp(((alpha/num_trials)*exp_centered_steps - beta))),1)
    CS_US_p*= p_mask

    CS_US = (np.random.rand(num_trials, US_onset - CS_onset) < CS_US_p)*1
    
    data = np.concatenate([start_CS, CS_US, US_end],axis=1)
    full_p_mask = np.concatenate([base_p*np.ones_like(start_CS), CS_US_p, base_p*np.ones_like(US_end)], axis=1)
    if visualize:
        plt.imshow(np.flipud(full_p_mask.squeeze()[:,::25]), cmap='gray')
        plt.colorbar()
        plt.savefig('/home/matt/figs/purkinje/train_heatmap.png')
        plt.close()
 
        make_raster(data, 'true_raster.png')
    return data
    
def train_model(model, train_data, test_data, CS_onset, US_onset, past, future, num_batches=1000, lr=1e-4, batch_size=32, show_every=25, save_every=100):

    loss_fn = torch.nn.MSELoss(size_average=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_hist = []
    acc_hist = []
    test_loss_hist = []
    test_acc_hist = []
    for b in tqdm(range(num_batches)):
        model.train()
        model.zero_grad()
        batch, target = split_sequence_batch(train_data,batch_size, CS_onset, US_onset, p=past, f=future)
        model.hidden = model.init_hidden(batch.shape[1])
        y_pred = model(batch.cuda())

        target_p = target.mean(1).unsqueeze(1).cuda()
        loss = loss_fn(y_pred, target_p)
        #accuracy = (target.cuda() == spikes_pred).float().mean()

        npy_loss = loss.data.cpu().numpy()
        #npy_acc  = accuracy.data.cpu().numpy()
        if b%show_every == 0: 
            #Evaluate
            model.eval()
            model.hidden = model.init_hidden(batch.shape[1])

            test_batch, test_target = split_sequence_batch(test_data, batch_size, CS_onset, US_onset, p=past, f=future)             
            test_y_pred = model(test_batch.cuda())

            test_target_p = test_target.mean(1).unsqueeze(1).cuda()
       
            test_loss = loss_fn(test_y_pred, test_target_p)
            #test_accuracy = (test_target.cuda() == test_spikes_pred).float().mean()

            test_npy_loss = test_loss.data.cpu().numpy()
            #test_npy_acc  = test_accuracy.data.cpu().numpy()
            # Plot
            loss_hist.append(npy_loss)
            #acc_hist.append(npy_acc)
            test_loss_hist.append(test_npy_loss)
            #test_acc_hist.append(test_npy_acc)
           
            print('\nLoss: {}'.format(npy_loss))
            plt.plot(np.array([loss_hist, test_loss_hist]).T)
            plt.legend(('Train', 'Test'))
            plt.savefig('/home/matt/figs/purkinje/loss.png')
            plt.close()
        if b%save_every == 0 and b > 0:
            torch.save(model.state_dict(), '/home/matt/models/purkinje/model.pt')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def gen_model(model, data, base_p, num_trials, trial_duration, CS_onset, US_onset, past, future):
    CSUS = torch.zeros(data.shape).cuda().double()
    CSUS[:,CS_onset:US_onset] = 1.0
    print('Generating model raster...')
    raster = []
    for n in tqdm(range(num_trials)):
        t = 0
        seed_spikes = (torch.rand(1, past) < base_p).double()
        spike_history = torch.tensor(seed_spikes)
        current_data = torch.cat([seed_spikes, torch.zeros_like(seed_spikes), n*torch.ones_like(seed_spikes)]).unsqueeze(0).transpose(2,0).transpose(2,1).cuda().double()
        while t < trial_duration - (past):
           model.hidden = model.init_hidden(1)
           y_pred = model.forward(current_data) 
           spikes_pred = y_pred.round().double().transpose(0,1)
           spike_history = torch.cat((spike_history, spikes_pred.data.cpu().transpose(0,1)), dim=1)
           t+=future
           next_spikes = torch.cat((current_data[future:,:,0], spikes_pred),dim=0)
           current_data = torch.cat((next_spikes, CSUS[n,t:t+past].unsqueeze(0).transpose(0,1), n*torch.ones_like(next_spikes)), dim=1).unsqueeze(1)
        raster.append(np.array(spike_history).squeeze())
    make_raster(np.array(raster), 'gen_raster')

def eval_model(model, test_data, trial_duration, CS_onset, US_onset, past, future):
    spike_history = torch.tensor(test_data[:,:past]).double()
    y_pred_history = []
    t = past
    num_trials = test_data.shape[0]
    print('Evaluating model...')
    while t < trial_duration - future:
        batch, target = split_sequence_eval(test_data, t, CS_onset, US_onset, p=past, f=future)
        y_pred = model.forward(batch.cuda())
        y_pred_history.append(y_pred.data.cpu().numpy())
        spikes_pred = (torch.rand((num_trials, future)).cuda().double() < y_pred).double()
        spike_history = torch.cat((spike_history, spikes_pred.data.cpu()), dim=1)
        t+=future
    plt.imshow(np.flipud(np.array(y_pred_history).squeeze().T), cmap='gray')
    #plt.imshow(imresize(np.flipud(np.array(y_pred_history).squeeze().T), (num_trials, int(.5*num_trials)), 'nearest') / 255., cmap='gray')
    yt = np.linspace(0,num_trials, 6)
    plt.yticks(yt, [str(j) for j in np.flipud(yt)])
    xt = np.linspace(0, int((trial_duration - future - past)/float(future)), 6)
    xtl= np.linspace(0, trial_duration, 6)
    plt.xticks(xt, [str(int(j)) for j in xtl])
    plt.colorbar()
    plt.savefig('/home/matt/figs/purkinje/eval_heatmap.png')
    plt.close()
    make_raster(spike_history.cpu().numpy(), 'gen_raster.png')

if __name__=='__main__':
    train = True
    num_trials = 50
    trial_duration = 500
    CS_onset = 150
    US_onset = 350
    base_p = .25
    past = 50
    future = 10
    batch_size = 512
    train_data = generate_data(base_p=base_p, num_trials=num_trials, trial_duration=trial_duration, CS_onset=CS_onset, US_onset=US_onset,alpha=100.0, beta=0, visualize=False)
    test_data = generate_data(base_p=base_p, num_trials=num_trials, trial_duration=trial_duration, CS_onset=CS_onset, US_onset=US_onset,alpha=100.0, beta=0, visualize=True)
    model = LSTM(3,256,batch_size=batch_size,output_dim=1,num_layers=1).double().cuda()
    if train:
        train_model(model, train_data, test_data, CS_onset, US_onset, past, future, batch_size=batch_size, lr=1e-5,num_batches=5000, show_every=25)
    else:
        print('Loading model...')
        model.load_state_dict(torch.load('/home/matt/models/purkinje/model.pt'))
    model.eval()
    eval_model(model, test_data,trial_duration, CS_onset, US_onset, past, future)
 
