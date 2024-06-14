import torch
import argparse
import json
import os
from Lib import *



import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
# Define the dataset class
class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return (x, y)

# Load the overall data
x = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/inverse_data_interpolated_numpy.p')
P = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/Test_MEM_data.p')
print(x.keys())
omega_fine = x['omega_fine']
omega = x['omega']
tau = x['tau']



def resume(filename, model, device):    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    



def evaluate_data(model, testloader, ERR, device):
    import matplotlib.pyplot as plt
    import numpy 
    model.eval()
    Ehat    = np.zeros([0,151])
    Rhat    = np.zeros([0,2000])
    E_test  = np.zeros([0,151])
    R_test  = np.zeros([0,2000])
    R_err  = np.zeros([0,2000])
    E_err  = np.zeros([0,151])
    R_corrupted = np.zeros([0,2000])
    R_corrupted1 = np.zeros([0,2000])
    for x_batch, y_batch in testloader:
        x_batch = x_batch.double().to(device) 
        y_batch = y_batch.double().to(device) 
        ehat, rhat, evar, rvar, _ = model.forward(x_batch, ERR, device=device)   
        Rhat   = np.concatenate([Rhat, rhat.detach().cpu().numpy()], axis = 0)
        Ehat   = np.concatenate([Ehat, ehat.detach().cpu().numpy()], axis = 0)
        E_test = np.concatenate([E_test, x_batch.detach().cpu().numpy()], axis = 0)
        R_test = np.concatenate([R_test, y_batch.detach().cpu().numpy()], axis = 0)
        E_err = np.concatenate( [ E_err, torch.sqrt( (ehat-evar)**2 ).cpu().detach().numpy() ], axis = 0)
        R_err = np.concatenate( [ R_err, torch.sqrt((rhat-rvar)**2).cpu().detach().numpy() ], axis = 0)

    return Rhat, Ehat, E_test, R_test, E_err, R_err



def evaluate_data__response_(model, x, ERR, n_curves = 1000, device='cpu'):
    import matplotlib.pyplot as plt
    model.eval()
    

    # print('#----------------------------------------------------------------------')
    x_batch =x.double().to(device).view(1,-1)
    # print("the input shape", x_batch.shape)

    # print('#----------------------------------------------------------------------')
    ehat, rhat, evar, rvar, corrupted = model.forward__res__(x_batch, ERR, n_curves)

    # print('#----------------------------------------------------------------------')
    # print(ehat.shape, rhat.shape, evar.shape, rvar.shape, corrupted.shape)
    return ehat, rhat, evar, rvar, corrupted



def Corr_plot(x, y, filename, x_inch, y_inch, xlabel, ylabel, labels,\
              xticks , yticks, xlim = None, ylim = None,\
              save_fig = False, log_scale = False):
    
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    
    def cm2inch(value):
            return value/2.54
        
    small = 14
    med = 16
    large = 18
    
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(x_inch),cm2inch(y_inch)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'text.color' : COLOR,
            'axes.labelcolor' : COLOR,
            'axes.linewidth' : 0.3,
            'xtick.color' : COLOR,
            'ytick.color' : COLOR}

    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'
    large = 24; med = 8; small = 7
    fig,a =  plt.subplots(1,1, sharex = False, dpi = 1000)
    # Some Plot oriented settings 
    a.spines["top"].set_visible(True)    
    a.spines["bottom"].set_visible(True)    
    a.spines["right"].set_visible(True)    
    a.spines["left"].set_visible(True)  
    a.grid(linestyle=':', linewidth=0.5)
    a.get_xaxis().tick_bottom()    
    a.get_yaxis().tick_left()  
    

    print(np.median(y))
    
    a.scatter(x, y, alpha=0.8, c=color[0], edgecolors='none', s=30,\
              label = labels, zorder=1 )
    
    a.plot([np.median(x), np.median(x)],  ylim, '--', lw=0.6, color = COLOR)
    a.plot(xlim, [np.median(y), np.median(y)], '--',  lw=0.6, color = COLOR)
    
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    if log_scale is True:
        a.set_yscale('log')
        a.set_xscale('log')
    a.set_xticks(xticks)
    a.set_yticks(yticks)
    
#     if xlim is not None:
#         a.set_xlim(xlim)
    
    if save_fig is True:
        plt.savefig(filename+'.pdf', dpi = 800,  bbox_inches='tight', pad_inches=0.01) 
    plt.show()

    

def Box_plot(data, labels, lims, filename, yticks,  log_scale = False, spacing = 0.4, title = 'random'):
    import pandas as pd
    
    import matplotlib.pyplot as plt
    def cm2inch(value):
            return value/2.54
    small = 16
    med = 18    
    large = 18
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(20),cm2inch(10)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
                'text.color' : COLOR,
                'axes.labelcolor' : COLOR,
                'axes.linewidth' : 1,
                'xtick.color' : COLOR,
                'ytick.color' : COLOR}

    plt.rcParams.update(params)
    plt.rc('text', usetex = False)
    color = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',\
             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    plt.rcParams['mathtext.fontset'] = 'cm'
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    fig,a =  plt.subplots(1,data.shape[1], sharex = False, dpi = 1000,\
                          gridspec_kw = {'wspace':spacing, 'hspace':spacing})
    print("the title of the plot", title+'\n\n')
    fig.suptitle(title+'\n')
    c = 'slategray'
    import matplotlib.ticker as ticker
    for i in range(data.shape[1]):
        a[i].boxplot(data[:,i], patch_artist=True,
            boxprops=dict(facecolor = 'white', color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='tab:orange'),
            )
        
        a[i].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

        
        a[i].spines["top"].set_visible(True)    
        a[i].spines["bottom"].set_visible(True)    
        a[i].spines["right"].set_visible(True)    
        a[i].spines["left"].set_visible(True)  


        a[i].set_xlabel(labels[i])
        if log_scale is True:
            a[i].set_yscale('log')
        a[i].grid(linestyle=':', linewidth=0.5)
        a[i].set_ylim(lims)
        a[i].set_yticks(yticks) 
        
        #a[i].yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    plt.savefig(filename+'.pdf', dpi = 800,  bbox_inches='tight') 



def plot_uncert(model, Kern, Kern_R, loader, filename, index=0, device='cuda'):
    import matplotlib.pyplot as plt
    import numpy 
    import matplotlib.pyplot as plt
    import seaborn as sns
    j=index
    CB91_Blue = '#05e1e1'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'
    color_list = ['#405952',
    '#9C9B7A',
    '#FFD393',
    '#FF974F',
    '#F54F29',
    ]
    CB91_Grad_BP = ['#2cbdfe', '#2fb9fc', '#33b4fa', '#36b0f8',
                '#3aacf6', '#3da8f4', '#41a3f2', '#449ff0',
                '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
                '#568ae6', '#5986e4', '#5c81e2', '#607de0',
                '#6379de', '#6775dc', '#6a70da', '#6e6cd8',
                '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
                '#7f57cf', '#8353cd', '#864ecb', '#894ac9',
                '#8d46c7', '#9042c5', '#943dc3', '#9739c1',
                '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
                '#a924b7', '#ac20b5', '#b01bb3', '#b317b1']

    small = 16
    med = 18
    large = 20


    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    rc={'axes.titlesize': small,
    'legend.fontsize': small,
    'axes.labelsize': med,
    'axes.titlesize': small,
    'xtick.labelsize': med,
    'ytick.labelsize': med,
    'figure.titlesize': small, 
    'text.color' : COLOR,
    'axes.labelcolor' : COLOR,
    'axes.axisbelow': False,
    'axes.edgecolor': COLOR,
    'axes.facecolor': 'None',
    'axes.grid': False,
    'axes.labelcolor': COLOR,
    'axes.spines.right': False,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    'axes.spines.top': False,
    'figure.facecolor': 'white',
    'lines.solid_capstyle': 'round',
    'patch.edgecolor': 'w',
    'patch.force_edgecolor': True,
    'text.color': COLOR,
    'xtick.bottom': True,
    'xtick.color': COLOR,
    'xtick.direction': 'in',
    'xtick.top': False,
    'ytick.color': COLOR,
    'ytick.direction': 'in',
    'ytick.left': True,
    'ytick.right': False}
    plt.rcParams.update(rc)
    plt.rc('text', usetex = False)
    path = '/home/kraghavan/Projects/Nuclear/Inverse/UQNuc/torch_attention_model_v12/Results/'
    fig, ax = plt.subplots(3,2, figsize=(16,15))
    n_models__ = [0, 1,2,3,4, 5, 6, 7, 8, 9]
    
    for i in range(3):        
        Err= 1e-05*pow(10,i)
        Rhat, Ehat, E_test, R_test, E_err, R_err=\
            evaluate_data(model, loader,\
                Err, device=device)
        ehat    = np.zeros([10,151])
        rhat    = np.zeros([10,2000])
        
        for j in n_models__:
            rbfnet = Network(Kern, Kern_R).double()
            file = path+'models/0_Trained_model_'+str(j)
            rbfnet = resume(file, rbfnet, device)
            rbfnet.to(device)
            ehat[j,:], rhat[j,:]= rbfnet.forward__res__ent_(torch.from_numpy(Ehat[i,:]),\
                Err, index__ = 0,\
                n_curves = 1, device = device)
        
        
        std__ = np.std(rhat, axis = 0)
        curve_R=R_test[j,:]
        yy = Rhat[j, :]
        yerr_R = R_err[j, :]
        curve_E=E_test[j,:]   
        xx = Ehat[j, :]
        yerr_E =E_err[j,:]
        fill_up = yy+yerr_R
        fill_down = yy-yerr_R
        ax[i][0].plot((omega_fine).reshape([-1]), curve_R, '--',\
                      label='R('+str(Err)+')', color='tab:blue')    
        fill_down[fill_down<0]= 0
        ax[i][0].fill_between(omega_fine.reshape([-1]), fill_up, fill_down, alpha=1, color='tab:orange')

        ax[i][0].set_xlim([0,400])
        ax[i][0].legend(loc='upper right')
        ax[i][1].plot( (tau).reshape([-1]), curve_E, label='E', color='tab:blue', linestyle=':')    
        ax[i][1].errorbar(tau.reshape([-1]), xx, yerr=yerr_E, fmt='x', linewidth=2, ms = 1, label='Ehat', color='tab:orange')

        ax[i][1].set_yscale('log')
        ax[i][1].legend(loc='upper right')
        ax[i][0].set_xlabel('$ \\omega (MeV)$')
        ax[i][1].set_xlabel('$ \\tau (MeV^{-1})$')
        ax[i][0].set_ylabel('$ R(\\omega)(MeV^{-1})$')
        ax[i][1].set_ylabel('$ E(\\tau)$')

        ax[i][0].grid(linestyle=':', linewidth=0.5)
        ax[i][1].grid(linestyle=':', linewidth=0.5)

    fig.suptitle('Reconstructions $R, E$ with increasing error $(\\sigma)$ in the input Euclidean', fontsize=16)
    fig.tight_layout()
    plt.savefig(filename+'.pdf', bbox_inches='tight', pad_inches = 0, dpi=1200) 
    plt.show()
    
    