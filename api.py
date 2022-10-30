from flask import Flask, render_template, request,url_for
import sys
from Blink import Blink 
import os
import numpy as np
from scipy.signal import *
import matplotlib
#get_ipython().magic(u'matplotlib notebook')
import matplotlib.pyplot as plt
import csv

#from werkzeug import secure_filename
sys.path.append('/home/prashant/.local/lib/python3.8/site-packages')
app = Flask(__name__)
app.config['UPLOAD_FOLDER']='static'

@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('upload.html')
@app.route('/upload', methods = ['GET', 'POST'])
def upload():
   return render_template('upload.html')
	
@app.route('/uploader_new', methods = ['GET', 'POST'])
def uploader_new():
   if request.method == 'POST':
        f = request.files['file']
        f.save('data/'+f.filename)
       
        blink = Blink()

        # Reading data files
        file_sig = f.filename


        # Loading Data
        if blink.mode:
            data_sig = np.loadtxt(open(os.path.join(blink.data_path,file_sig), "rb"), delimiter=";", skiprows=1, usecols=(0,1,2))
        else:
            data_sig = np.loadtxt(open(os.path.join(blink.data_path,file_sig), "rb"), delimiter=",", skiprows=5, usecols=(0,1,2))
            data_sig = data_sig[0:(int(200*blink.fs)+1),:]
            data_sig = data_sig[:,0:3]
            data_sig[:,0] = np.array(range(0,len(data_sig)))/blink.fs

        # Step1: Low Pass Filter
        data_sig[:,1] = blink.lowpass(data_sig[:,1], 10, blink.fs, 4)
        data_sig[:,2] = blink.lowpass(data_sig[:,2], 10, blink.fs, 4)

        time_min = data_sig[0,0]
        time_max = data_sig[-1,0]

        data_len = len(data_sig)


        # decoding stimulations
        #interval_corrupt, gt_blinks = decode_stim(blink.data_path, file_stim)

        args_chan1 = blink.args_init(blink.delta_init)

        running_std = blink.compute_running_std(data_sig, blink.chan_id, blink.fs)
        for idx in range(len(data_sig[:,0])):
            blink.peakdet(data_sig[idx,0], data_sig[idx, blink.chan_id], args_chan1)

        min_pts = np.array(args_chan1['mintab'])
        p_blinks_t, p_blinks_val = blink.find_expoints(min_pts, data_sig, blink.chan_id,running_std)
        corr_matrix, pow_matrix = blink.compute_correlation(p_blinks_t, data_sig, blink.chan_id, blink.fs)



            
        # fingerprint
        blink_fp_idx = np.argmax(sum(corr_matrix))
        t = corr_matrix[blink_fp_idx,:] > blink.corr_threshold_1
        blink_index = [i for i, x in enumerate(t) if x]

        blink_template_corrmat = corr_matrix[np.ix_(blink_index,blink_index)]
        blink_template_powmat = pow_matrix[np.ix_(blink_index,blink_index)]
        blink_templates_corrWpower = blink_template_corrmat/blink_template_powmat

        blink_var = []
        for idx in blink_index:
            blink_var.append(np.var(data_sig[int(blink.fs*p_blinks_t[idx,0]):int(blink.fs*p_blinks_t[idx,2]), blink.chan_id]))
        from scipy.cluster.hierarchy import linkage
        from scipy.cluster.hierarchy import fcluster

        Z = linkage(blink_templates_corrWpower, 'complete', 'correlation')
        groups = fcluster(Z,2,'maxclust')

        grp_1_blinks_var = [blink_var[i] for i, x in enumerate(groups==1) if x]
        grp_2_blinks_var = [blink_var[i] for i, x in enumerate(groups==2) if x]
        if np.mean(grp_1_blinks_var) > np.mean(grp_2_blinks_var):
            selected_group = 1
        else:
            selected_group = 2
        template_blink_idx = [blink_index[i] for i, x in enumerate(groups==selected_group) if x]

        # computing delta new
        delta_new = 0
        for idx in template_blink_idx:
            delta_new = delta_new + min(p_blinks_val[idx,0], p_blinks_val[idx,2]) - p_blinks_val[idx,1]
        delta_new = delta_new/len(template_blink_idx)

        # 2nd pass

        args_chan1 = blink.args_init(delta_new/3.0)

        for idx in range(len(data_sig[:,0])):
            blink.peakdet(data_sig[idx,0], data_sig[idx, blink.chan_id], args_chan1)

        min_pts = np.array(args_chan1['mintab'])
        p_blinks_t, p_blinks_val = blink.find_expoints(min_pts, data_sig, blink.chan_id,running_std)
        corr_matrix, pow_matrix = blink.compute_correlation(p_blinks_t, data_sig, blink.chan_id, blink.fs)

            
        s_fc = (sum(corr_matrix))
        sort_idx = sorted(range(len(s_fc)), key=lambda k: s_fc[k])

        t = corr_matrix[sort_idx[-1],:] > blink.corr_threshold_2        
        blink_index1 = set([i for i, x in enumerate(t) if x])
        t = corr_matrix[sort_idx[-2],:] > blink.corr_threshold_2        
        blink_index2 = set([i for i, x in enumerate(t) if x])
        t = corr_matrix[sort_idx[-3],:] > blink.corr_threshold_2        
        blink_index3 = set([i for i, x in enumerate(t) if x])

        blink_index = list(blink_index1.union(blink_index2).union(blink_index3))

        blink_template_corrmat = corr_matrix[np.ix_(blink_index,blink_index)]
        blink_template_powmat = pow_matrix[np.ix_(blink_index,blink_index)]
        blink_templates_corrWpower = blink_template_corrmat/blink_template_powmat

        blink_var = []
        for idx in blink_index:
            blink_var.append(np.var(data_sig[int(blink.fs*p_blinks_t[idx,0]):int(blink.fs*p_blinks_t[idx,2]), blink.chan_id]))


        Z = linkage(blink_templates_corrWpower, 'complete', 'correlation')
        groups = fcluster(Z,2,'maxclust')

        grp_1_blinks_var = [blink_var[i] for i, x in enumerate(groups==1) if x]
        grp_2_blinks_var = [blink_var[i] for i, x in enumerate(groups==2) if x]

        if np.mean(grp_1_blinks_var) > np.mean(grp_2_blinks_var) and np.mean(grp_1_blinks_var)/np.mean(grp_2_blinks_var) > 10:
            blink_index = [blink_index[i] for i, x in enumerate(groups==1) if x]
        elif np.mean(grp_2_blinks_var) > np.mean(grp_1_blinks_var) and np.mean(grp_2_blinks_var)/np.mean(grp_1_blinks_var) > 10:
            blink_index = [blink_index[i] for i, x in enumerate(groups==2) if x]

        final_blinks_t = p_blinks_t[blink_index,:]
        final_blinks_val = p_blinks_val[blink_index,:]
        output = [{'start_time':final_blinks_t.T[0],'end_time':final_blinks_t.T[2]}]
        output = str(output)
        return render_template('upload.html',response_text=output)
        #return 'file uploaded successfully'
      
      
if __name__ == '__main__':
   app.run(debug = True,port=1081)