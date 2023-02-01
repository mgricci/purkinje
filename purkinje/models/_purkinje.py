import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy.signal import convolve, gaussian
import pdb

class Purkinje(object):
    def __init__(self,
                 R_max=1.0,
                 R_production_rate=.0001,
                 V_hyp=-85,
                 V_rest=-70,
                 V_thresh=-54,
                 V_spike=10,
                 tau_m=5,
                 tau_e=1,
                 tau_i=2,
                 tau_j1=90,
                 tau_j2=200,
                 tau_r=100,
                 s=0.0,
                 J1_0=2.0,
                 J2_0=2.0,
                 J1_thresh=0.0,
                 J2_thresh=0.0,
                 c=0.01,
                 evolution_noise=10.0,
                 a=2000,
                 write_ref_counter_max=2000,
                 read_ref_counter_max=2000,
                 mu=10,
                 sigma=5,
                 max_ISI=1800):
        
        self.R_max  = R_max
        self.R_production_rate = R_production_rate
        self.V_hyp = V_hyp
        self.V_rest = V_rest
        self.V_thresh = V_thresh
        self.V_spike = V_spike
        self.tau_m = tau_m
        self.tau_e = tau_e
        self.tau_i = tau_i
        self.tau_j1 = tau_j1# Smaller: faster recovery --> later trigger --> earlier pause. Larger: slower recovery --> earlier trigger --> later pause
        self.tau_j2 = tau_j2
        self.tau_r = tau_r
        self.s = s
        self.J1_0 = J1_0
        self.J2_0 = J2_0
        self.J1_thresh = J1_thresh
        self.J2_thresh = J2_thresh
        self.c = c
        self.evolution_noise = evolution_noise
        self.a = a #2000
        self.write_ref_counter_max = write_ref_counter_max
        self.read_ref_counter_max = read_ref_counter_max
        self.mu = mu
        self.sigma= sigma
        self.max_ISI = max_ISI
    
        self.all_R  = []
        self.all_P = []
        self.all_V = []
        self.all_I = []
        self.all_E = []
        self.all_H = []
        self.all_J1 = []
        self.all_J2 = []
        self.all_CS = []
        self.all_US = []
                      
        self.all_rasters = []

        self.H = np.zeros((self.max_ISI,))

        
    def V_step(self,V, E, I, P):
        delta = -1*(1. / self.tau_m) * (V - self.V_rest) + E - I + P
        # TODO: problem here? 
        return V + delta

    def E_step(self,E, CS):
        delta = -1*(1 / self.tau_e)*E + self.s*CS
        return E + delta

    def I_step(self,I, h):
        delta = -1*(1 / self.tau_i) * I + h
        return I + delta

    def J1_step(self,J1, CS):
        delta = (1/self.tau_j1)*(self.J1_0 - J1) - CS
        return J1 + delta

    def J2_step(self,J2, CS):
        delta = (1/self.tau_j2)*(self.J2_0 - J2) - CS
        return J2 + delta

    def get_pause_stats(self, inh, thresh, CS_onset, max_type='mode'):

        high = np.argwhere(np.array(inh) > thresh) - CS_onset * 1000
        pause_onset = high[0]
        pause_offset = high[-1]

        if max_type == 'mode':
            pause_max = np.argmax(inh) - CS_onset*1000
        elif max_type == 'mean':
            pause_max = .5 * (pause_offset - pause_onset) - CS_onset * 1000

        return pause_onset, pause_offset, pause_max
    
    def run(self,trials=100, ITI=5.0, CS_onsets=[0.001], CS_offsets=[.5], US_onsets=[1.5],
            US_offsets=[1.6], CS_freq=100, US_freq = 500, probe_start=75, probe_per=5,
            multiple=False,verbose=False, early_stopping_I=np.infty):
        
        CS_onsets  = [1000*CSon for CSon in CS_onsets]
        CS_offsets  = [1000*CSoff for CSoff in CS_offsets]
        US_onsets  = [1000*USon for USon in US_onsets]
        US_offsets = [1000*USoff for USoff in US_offsets]
        ITI       = int(1000*ITI)
        
        kernel = gaussian(99, self.sigma)
        kernel /= kernel.sum()
        # Initialize
        V = self.V_rest
        E = 0
        I = 0
        J1 = self.J1_0
        J2 = self.J1_0
        R = self.R_max
        write_ref_counter = 0
        read_ref_counter = 0
        US_flag = False
        writing = False
        reading = False
        batch_writing = np.zeros((self.max_ISI,))
        spiked=False

        for i in range(trials):
            if verbose:
                print('Trial {}'.format(i))
                
            if multiple:
                trial_CS_onsets = [CS_onsets[i % len(CS_onsets)]]
                trial_CS_offsets = [CS_offsets[i % len(CS_offsets)]]
                trial_US_onsets = [US_onsets[i % len(US_onsets)]]
                trial_US_offsets = [US_offsets[i % len(US_offsets)]]
            else:
                trial_CS_onsets = CS_onsets
                trial_CS_offsets = CS_offsets
                trial_US_onsets = US_onsets
                trial_US_offsets = US_offsets
                
            CS_intervals = []
            for c in range(len(trial_CS_onsets)):
                CS_intervals += range(int(trial_CS_onsets[c]), int(trial_CS_offsets[c]))
                
            US_intervals = []
            if sum(trial_US_onsets) < np.inf and sum(trial_US_offsets) < np.inf:
                for u in range(len(trial_US_onsets)):
                    US_intervals += range(int(trial_US_onsets[u]), int(trial_US_offsets[u]))

            raster = []
            all_CS = []
            trial_R  = []
            trial_P = []
            trial_V = []
            trial_I = []
            trial_E = []
            trial_H = []
            trial_J1 = []
            trial_J2 = []
            trial_CS = []
            trial_US = []
            trial_rasters = []
            trial_h = []
            US_detected = False
            for t in range(ITI): 
                if t > probe_start and ((t - probe_start) % probe_per) == 0:
                    probe = True
                else:
                    probe = False
                    
                # Replenish vesicle
                if not writing:
                    R = min(R + self.R_production_rate*self.R_max, self.R_max)
                    R = max(R, 0)
                # Decrement refratory periods
                read_ref_counter = max(read_ref_counter - 1, 0)
                write_ref_counter = max(write_ref_counter - 1, 0)
                
                # Stimuli
                
                if t in CS_intervals:
                    CS = 1*(((t - CS_intervals[0]) % (1000 / CS_freq)) == 0)
                else:
                    CS = 0
                # Stimuli
                if t in US_intervals and not probe:
                    if not US_detected:
                        US_flag = True
                        US_detected = True
                    US = 1*(((t - US_intervals[0]) % (1000 / US_freq)) == 0)
                else:
                    US = 0
                all_CS.append(CS)

                # Update ctivation energy
                J1 = self.J1_step(J1,CS)
                J2 = self.J2_step(J1,CS)
 
                # If writing threshold is met and w-refractory period is over
                if J1 <= self.J1_thresh and write_ref_counter == 0 and len(US_intervals) > 0:
#                     print(t)
                    #Reset refractory period
                    write_ref_counter = self.write_ref_counter_max
                    # Set mode to writing
                    writing = True
                    write_t = t
                    # Initialize writing batch
                    batch_writing = list(np.zeros((self.max_ISI,)))
             
                # If writing 
                if writing:
                    # Sample batch from reserve
                    new_batch = R / self.tau_r
                    R = R - new_batch
                    # Add to evolution
                    batch_writing = [new_batch] + batch_writing[:-1]
                    # Evolution noise
                    batch_writing = list(np.convolve(np.array(batch_writing), kernel, mode='same'))
                # If reading threshold met and r-refractory period is over
                if J2 <= self.J2_thresh and read_ref_counter == 0: 
                    # Reset refractory period
                    read_ref_counter = self.read_ref_counter_max
                    # Turn on reading mode and turn off writing
                    reading = True
                    read_t = t
                    all_h = []
                    read_index = 0
                    # Read batch and decrement archive
                    batch_reading = self.c*self.H
                    self.H -= self.c*self.H
                  
                if reading:
                    h = self.a * batch_reading[read_index]
                    read_index += 1
                    if read_index == len(self.H) - 1:
                        reading = False
                else:
                    h = 0.0

                # At first US spike
                if US_flag:
                    # Stop writing
                    writing=False
                    # Add writing batch to the archive
                    self.H += batch_writing
                    US_flag = False

                # Dynamics
                P = self.mu * (np.random.rand() < .3)
                E = self.E_step(E,CS)
                I = self.I_step(I,h)
                if spiked:
                    V = self.V_hyp
                    spiked=False
                else:
                    V = self.V_step(V,E,I,P)

                if V > self.V_thresh:
                    spiked=True
                    V = self.V_spike
                    raster.append(1)
                else:
                    raster.append(0)
                trial_P.append(P)
                trial_R.append(R)
                trial_I.append(I)
                trial_E.append(E)
                trial_V.append(V)
                trial_J1.append(J1)
                trial_J2.append(J2)
                trial_CS.append(CS)
                trial_US.append(US)
                trial_h.append(h)

            self.all_H.append(self.H)
            self.all_P.append(trial_P)
            self.all_R.append(trial_R)
            self.all_I.append(trial_I)
            self.all_E.append(trial_E)
            self.all_V.append(trial_V)
            self.all_J1.append(trial_J1)
            self.all_J2.append(trial_J2)
            self.all_CS.append(trial_CS)
            self.all_US.append(trial_US)
            self.all_rasters.append(raster)
            if np.max(trial_I) > early_stopping_I:
                break
