import os
import numpy as np
from scipy.signal import *
import matplotlib
#get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
import csv
from scipy.interpolate import interp1d




class Blink:
    def __init__(self):
        
        self.mode = False # mode True means EEG-IO, otherwise v/r (EEG-VV or EEG-VR) data
        self.data_path = 'data' if self.mode else 'EEG-VR' # or replace w/ EEG-VR
        self.file_idx = 7
        self.fs = 250.0
        self.chan_id = 1

        self.flag_soft = True # if True, consider soft blinks as ground truth

        self.blink_len_max = 2.0 # in seconds
        self.blink_len_min = 0.2 # in seconds


        self.delta_init = 100 # in uvolts

        self.corr_threshold_1 = 0.2
        self.corr_threshold_2 = 0.7

        self.std_threshold_window = int(5*self.fs)  #%  in seconds - for each direction

    def lowpass(self,sig, fc, fs, butter_filt_order):
        B,A = butter(butter_filt_order, np.array(fc)/(fs/2), btype='low')
        return lfilter(B, A, sig, axis=0)

    

    def compute_running_std(self,data_sig, chan_id, fs):
        # Find running std
        std_length = int(0.5*fs) # in seconds
        data_len = len(data_sig)
        running_std = np.zeros([data_len,1])
        idx = 0
        while(idx < len(data_sig) - std_length):
            running_std[idx] = np.std(data_sig[idx:(idx + std_length), chan_id])
            idx = idx + 1
        running_std[idx:-1] = running_std[idx-1]

        # fixing the corrupted signal's std
        for idx in range(data_len):
            if running_std[idx] < 1:
                l_index_lhs = max(0,idx-std_length)
                l_index_rhs = max(0,(idx-std_length-2*self.std_threshold_window-int(fs)))
                r_index_lhs = min(data_len, idx+std_length)
                r_index_rhs = max(0,idx-std_length-int(fs))
                running_std[l_index_lhs:r_index_lhs] = min(running_std[l_index_rhs:r_index_rhs])
                idx=idx+std_length-1

        return running_std

    # Function to find peaks
    def args_init(self,delta_uV):
        args = {}
        args['mintab'], args['maxtab'] = [], []
        args['mn'], args['mx'] = float("inf"), -1*float("inf")
        args['mnpos'], args['mxpos'] = None, None
        args['min_left'], args['min_right'] = [], []
        args['lookformax'] = True
        args['delta'] = delta_uV
        return args


    def peakdet(self,time, value, args):
        foundMin = False
        if value > args['mx']:
            args['mx'] = value
            args['mxpos'] = time
        if value < args['mn']:
            args['mn'] = value
            args['mnpos'] = time
        if args['lookformax']:
            if value < args['mx'] - args['delta']:
                args['maxtab'].append([args['mxpos'], args['mx']])
                args['mn'] = value
                args['mnpos'] = time
                args['lookformax'] = False
        else:
            if value > args['mn'] + args['delta']:
                args['mintab'].append([args['mnpos'], args['mn']])
                args['min_left'].append([-1, -1])
                args['min_right'].append([-1, -1])
                args['mx'] = value
                args['mxpos'] = time
                args['lookformax'] = True                
                foundMin = True
        return foundMin

    ## Finding extreme points

    def find_expoints(self,stat_min2, data_sig, chan_id,running_std):
        # Parameters
        offset_t = 0.00 # in seconds
        win_size = 25
        win_offset = 10
        search_maxlen_t = 1.5 # in seconds


        offset_f = int(offset_t*self.fs)
        search_maxlen_f = int(search_maxlen_t*self.fs)
        iters = int(search_maxlen_f/win_offset)

        data_len = len(data_sig)
        p_blinks_t, p_blinks_val = [], []
        for idx in range(len(stat_min2)):
            # x_indR and x_indL are starting points for left and right window
            x_indR = int(self.fs*stat_min2[idx,0]) + offset_f
            x_indL = int(self.fs*stat_min2[idx,0]) - offset_f
            start_index = max(0, int(self.fs*stat_min2[idx,0]) - self.std_threshold_window)
            end_index = min( int(self.fs*stat_min2[idx,0]) + self.std_threshold_window, data_len)
            stable_threshold = 2*min(running_std[start_index:end_index])
            min_val = stat_min2[idx,1];
            max_val = min_val;
            found1, found2 = 0, 0
            state1, state2 = 0, 0

            for iter in range(iters):
                if(x_indR + win_size > data_len):
                    x_indR = x_indR - (x_indR + win_size - data_len)
                if(x_indL < 0):
                    x_indL = 0
                if (np.std(data_sig[x_indR:x_indR+win_size, chan_id]) < stable_threshold) and state1==1 and data_sig[x_indR, chan_id]>min_val:
                    found1 = 1
                    max_val = max(data_sig[x_indR, chan_id],max_val)
                if (np.std(data_sig[x_indL:x_indL+win_size, chan_id]) < stable_threshold) and state2==1 and data_sig[x_indL + win_size, chan_id]>min_val:
                    found2 = 1
                    max_val = max(data_sig[x_indL + win_size, chan_id],max_val)
                if (np.std(data_sig[x_indR:x_indR+win_size, chan_id]) > 2.5*stable_threshold) and state1==0:
                    state1 = 1
                if (np.std(data_sig[x_indL:x_indL+win_size, chan_id]) > 2.5*stable_threshold) and state2==0:
                    state2 = 1
                if (found1==1) and data_sig[x_indR, chan_id] < (max_val + 2*min_val)/3:
                    found1=0
                if (found2==1) and data_sig[x_indL + win_size, chan_id] < (max_val + 2*min_val)/3:
                    found2=0
                if (found1==0):
                    x_indR = x_indR + win_offset
                if (found2==0):
                    x_indL = x_indL - win_offset;
                if found1==1 and found2==1:
                    break
            if found1==1 and found2==1:
                if (x_indL + win_size)/self.fs > stat_min2[idx,0]:
                    p_blinks_t.append([(x_indL)/self.fs, stat_min2[idx,0], x_indR/self.fs])
                    p_blinks_val.append([data_sig[x_indL, chan_id], stat_min2[idx,1], data_sig[x_indR,chan_id]])         
                else:
                    p_blinks_t.append([(x_indL + win_size)/self.fs, stat_min2[idx,0], x_indR/self.fs])
                    p_blinks_val.append([data_sig[x_indL + win_size, chan_id], stat_min2[idx,1], data_sig[x_indR,chan_id]])


        p_blinks_t = np.array(p_blinks_t)        
        p_blinks_val = np.array(p_blinks_val)  

        return p_blinks_t, p_blinks_val

    def compute_correlation(self,p_blinks_t, data_sig, chan_id, fs):
        total_p_blinks = len(p_blinks_t)
        corr_matrix = np.ones([total_p_blinks, total_p_blinks])
        pow_matrix = np.ones([total_p_blinks, total_p_blinks])
        for idx_i in range(total_p_blinks):
            for idx_j in range(idx_i+1,total_p_blinks):

                blink_i_left = data_sig[int(fs*p_blinks_t[idx_i,0]):int(fs*p_blinks_t[idx_i,1]), chan_id]
                blink_i_right = data_sig[int(fs*p_blinks_t[idx_i,1]):int(fs*p_blinks_t[idx_i,2]), chan_id]

                blink_j_left = data_sig[int(fs*p_blinks_t[idx_j,0]):int(fs*p_blinks_t[idx_j,1]), chan_id]
                blink_j_right = data_sig[int(fs*p_blinks_t[idx_j,1]):int(fs*p_blinks_t[idx_j,2]), chan_id]

                left_interp = interp1d(np.arange(blink_i_left.size), blink_i_left)
                compress_left = left_interp(np.linspace(0,blink_i_left.size-1, blink_j_left.size))
                right_interp = interp1d(np.arange(blink_i_right.size), blink_i_right)
                compress_right = right_interp(np.linspace(0,blink_i_right.size-1, blink_j_right.size))

                sigA = np.concatenate((compress_left, compress_right))
                sigB = np.concatenate((blink_j_left, blink_j_right))

                corr = np.corrcoef(sigA, sigB)[0,1]
                corr_matrix[idx_i, idx_j] = corr
                corr_matrix[idx_j, idx_i] = corr

                if np.std(sigA) > np.std(sigB):
                    pow_ratio = np.std(sigA)/np.std(sigB)
                else:
                    pow_ratio = np.std(sigB)/np.std(sigA)

                pow_matrix[idx_i, idx_j] = pow_ratio
                pow_matrix[idx_j, idx_i] = pow_ratio


        return corr_matrix, pow_matrix