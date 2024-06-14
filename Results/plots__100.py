import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
import torch
sys.path.append("/home/kraghavan/Projects/Nuclear/Inverse/UQNuc/torch_attention_model_v12/")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
from plots_utils import *
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
R = np.concatenate([x['One_Peak_R_interp'],\
    x['Two_Peak_R_interp'] ], axis=0)
_, _, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
# print(E.shape, R.shape)
## The one peak
# remove_out_1 =  [637, 74, 901, 902, 70]
# remove_out_2 = [476, 388, 402, 404, 407]
# remove_out_1 =  [637, 74, 901, 902, 70]
# remove_out_2 = [476, 388, 402, 404, 407]
# The test data (one peak)



# -------------------------------------------------------
## The data
R_test_1 = np.concatenate([  P['One_Peak_R']], axis=0)
# R_test_1 = np.delete(R_test_1, remove_out_1, 0)
R_test_2 = np.concatenate([  P['Two_Peak_R']], axis=0)
# R_test_2 = np.delete(R_test_2, remove_out_2, 0)
E_test_1 = np.concatenate([  P['One_Peak_E']], axis=0)
# E_test_1 = np.delete(E_test_1, remove_out_1, 0)
E_test_2 = np.concatenate([  P['Two_Peak_E']], axis=0)
# E_test_2 = np.delete(E_test_2, remove_out_2, 0)
E_test_1 = torch.from_numpy(E_test_1)
R_test_1 = torch.from_numpy(R_test_1)
E_test_2 = torch.from_numpy(E_test_2)
R_test_2 = torch.from_numpy(R_test_2)
# Gather the reconstructions from MEM
# Gather the reconstructions from MEM
R_MEM_1  = P['One_Peak_R_MEM']
E_MEM_1  = P['One_Peak_E_MEM']
E_MEM_2  = P['Two_Peak_E_MEM']
R_MEM_2  = P['Two_Peak_R_MEM']
# E_MEM_1 = np.delete(E_MEM_1, remove_out_1, 0)
# E_MEM_2 = np.delete(E_MEM_2, remove_out_2, 0)
# R_MEM_1 = np.delete(R_MEM_1, remove_out_1, 0)
# R_MEM_2 = np.delete(R_MEM_2, remove_out_2, 0)
R_MEM = np.concatenate([R_MEM_1, R_MEM_2], axis=0) 
E_MEM = np.concatenate([E_MEM_1, E_MEM_2], axis=0) 
R_test = np.concatenate([R_test_1, R_test_2], axis=0) 
E_test = np.concatenate([E_test_1, E_test_2], axis=0) 
E_test   = torch.from_numpy(E_test)
R_test   = torch.from_numpy(R_test)

#-----------------------------------------------------------------
# The real Euclidean inversion.
#-----------------------------------------------------------------
filename = 'uncert_two'
num = 103
euc = E_test[num, :]
res = R_test[num,:]


# E = np.loadtxt('GFMC__/euc/euc_'+filename+'.out')
# R = np.loadtxt('GFMC__/res/res_'+filename+'.out')
# tau = E[:,0].reshape([-1])
# euc = torch.from_numpy(E[:,1].reshape([1,-1]))
# scale = euc[0]
# euc = euc/scale
scale=1
# print("checking", euc[:,0], scale)

# omega = R[:,0].reshape([-1])
# res_1   = torch.from_numpy(R[:, 1].reshape([-1]))
# res_2 = torch.from_numpy(R[:, 2].reshape([-1]))
# res   = torch.from_numpy(R[:, 3].reshape([-1]))
# print(E.shape, R.shape,tau.shape, omega.shape, res.shape, euc.shape)




#-----------------------------------------------------------------
#### We got our basis and it is good
Kern = Kern.astype('float64')
torch.pi = torch.acos(torch.zeros(1)).item()
path = '/home/kraghavan/Projects/Nuclear/Inverse/UQNuc/torch_attention_model_v12/Results/'
# -----------------------------------------------------------------
rbfPRC = NetworkPRC(Kern, Kern_R)
rbfPRC.load_state_dict(torch.load('/gpfs/jlse-fs0/users/kraghavan/Inverse/UQNuc/torch_attention_model_v11/results/models/modella_both__PRC_00', map_location=device))
rbfPRC.to(device)
rbfnet_1 = Network(Kern, Kern_R).double()
file = path+'models/1_Trained_model_7'
rbfnet_1 = resume(file, rbfnet_1, device)
rbfnet_1.to(device)
# ----------------------------------------------------------------------------
rbfnetUQ_1 = Network(Kern, Kern_R).double()
file = path+'models/1_Trained_model_8'
rbfnetUQ_1 = resume(file, rbfnetUQ_1, device)
rbfnetUQ_1.to(device)

#-----------------------------------------------------------------
rbfnet_2 = Network(Kern, Kern_R).double()
file = path+'models/1_Trained_model_9'
rbfnet_2 = resume(file, rbfnet_2, device)
rbfnet_2.to(device)
## -----------------------------------------------------------------
rbfnetUQ_2 = Network(Kern, Kern_R).double()
file = path+'models/1_Trained_model_2'
rbfnetUQ_2 = resume(file, rbfnetUQ_2, device)
rbfnetUQ_2.to(device)



#-----------------------------------------------------------------
#
# # The Metrics
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager
large = 11; med = 11; small = 11
marker_size = 1.01
lw = 1
inten = 0.4
index = np.arange(0,151, 1)
# list_best_index = [1540, 1600, 1720]
labels = ['Ent-NN', 'Original', 'MaxEnt', 'Phys-NN']
label_type = ['best', 'median', 'worst']
xlims =[[0,750], [0,900], [0,900]]
ylims =[[0,0.0035], [0,0.004], [0,0.1]]



#----------------------------------
# # Gather the reconstructions from my model
# ------------------------------
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager
large = 11; med = 11; small = 11
marker_size = 1.01
lw = 0.1
inten = 0.4
def cm2inch(value):
    return value/2.54
plt.style.use('seaborn-white')
COLOR = 'darkslategray'
params = {'axes.titlesize': small,
        'legend.fontsize': small,
        'figure.figsize': (cm2inch(36),cm2inch(23.5)),
        'axes.labelsize': med,
        'axes.titlesize': small,
        'xtick.labelsize': small,
        'lines.markersize': marker_size,
        'ytick.labelsize': med,
        'figure.titlesize': small, 
            'text.color' : COLOR,
            'axes.labelcolor' : COLOR,
            'axes.linewidth' : 0.5,
            'xtick.color' : COLOR,
            'ytick.color' : COLOR}

plt.rcParams.update(params)
plt.rc('text', usetex = False)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', '#9467bd']
plt.rcParams['mathtext.fontset'] = 'cm'

#--------------------------------------------------
## create plots with numpy array
fig,a =  plt.subplots(2, 3, dpi = 1200, gridspec_kw = {'wspace':0.40, 'hspace':0.40})
for i, Err in enumerate([0.01, 0.001, 0.0001]):
    
    
    ehat, rhat, evar, rvar, corrupted= evaluate_data__response_(rbfnetUQ_1, euc, Err, n_curves = 10000)
    ehat_Q, rhat_Q, _,_,_= evaluate_data__response_(rbfnet_1, euc, Err, n_curves = 10000)
    
    
    for k in range(rvar.shape[0]):
        a[0][i].plot(omega_fine, rvar[k,:], color = 'lightgreen', alpha=0.5,  lw = lw)
    
    
    a[0][i].plot(omega_fine, rhat[0,:], color = 'lightgreen', label = 'UQ-NN', linestyle='-.', linewidth = 10*lw)
    a[0][i].plot(omega_fine, res, color =colors[0],   label = 'Original', linestyle = '-',linewidth = 20*lw)
    a[0][i].plot(omega_fine, rhat_Q[0,:], color = colors[1], linestyle = '--',label = 'Ent-NN', linewidth = 20*lw)
    

    # a[0][i].plot(omega, res_1, color = colors[0],   label = 'Original(2rd)', linewidth = 10*lw)
    # a[0][i].plot(omega, res_2, color=colors[0],
    #              label='Original(1st)', linewidth=10*lw)
    
    #  ehat    = np.zeros([10,151])
    #     rhat    = np.zeros([10,2000])
        
    #     for j in n_models__:
    #         rbfnet = Network(Kern, Kern_R).double()
    #         file = path+'models/0_Trained_model_'+str(j)
    #         rbfnet = resume(file, rbfnet, device)
    #         rbfnet.to(device)
    #         ehat[j,:], rhat[j,:]= rbfnet.forward__res__ent_(torch.from_numpy(Ehat[i,:]),\
    #             Err, index__ = 0,\
    #             n_curves = 1, device = device)
            
            
            
    a[0][i].set_xlim([0,400])
    a[0][i].set_xlabel('$\omega~[\mathrm{MeV}]$')
    a[0][i].set_ylabel('$R(\omega)~[\mathrm{MeV}^{-1}]$')
    a[0][i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
    a[0][i].grid(linestyle=':', linewidth=0.5)
    a[1][i].plot(tau, euc.reshape([-1]), color=colors[0],
                 label='Original', linewidth=20*lw)
    
    
    # for k in range(rvar.shape[0]):
    #     a[1][i].plot(tau, evar[k,:], color = colors[1], linestyle = '-', lw = lw)
    
    
    for k in range(rvar.shape[0]):
        a[1][i].plot(tau, corrupted[k,:]*scale.numpy(), color = 'lightskyblue', alpha=0.05, linestyle = '-.', lw = lw)
    
    # a[1][i].plot(tau, ehat[0, :], color=colors[2],alpha=0.6,
    #               linestyle='-', linewidth=lw)
    
    a[1][i].plot(tau, euc.reshape([-1])*scale.numpy(), color=colors[0],linestyle='--', linewidth=10*lw)
    a[1][i].plot(tau, ehat_Q[0, :]*scale.numpy(), color=colors[1], linestyle = '-.', linewidth=10*lw)
    a[1][i].set_title("the Chi2"+ str( (torch.mean(( euc.reshape([-1,1])-ehat_Q[0,:].reshape([-1,1]) )**2/Err**2)).numpy() ) )
    a[1][i].set_xlim([0,0.0750])
    a[1][i].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$')
    a[1][i].set_ylabel('$E(\\tau)$')
    a[1][i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
    a[1][i].grid(linestyle=':', linewidth=0.5)
    a[0][i].set_title('$\\sigma$='+str(Err))
    
    
a[0][2].legend(loc = 'upper right',ncol=1 )
plt.savefig('Plots/'+filename+'.png', bbox_inches='tight', dpi=120)
plt.close()




# #-----------------------------------------------------------------
# # The real Euclidean inversion.
# #-----------------------------------------------------------------
# filename = 'rr_12b'
# E = np.loadtxt('GFMC__/euc/euc_'+filename+'.out')
# R = np.loadtxt('GFMC__/res/res_'+filename+'.out')
# tau = E[:,0].reshape([-1])
# euc = torch.from_numpy(E[:,1].reshape([1,-1]))
# scale = euc[:, 0]
# euc = euc/scale
# print("checking", euc[:,0], scale)
# omega = R[:,0].reshape([-1])
# res_1   = torch.from_numpy(R[:, 1].reshape([-1]))/scale
# res_2 = torch.from_numpy(R[:, 2].reshape([-1]))/scale
# res   = torch.from_numpy(R[:, 3].reshape([-1]))/scale
# print(E.shape, R.shape,tau.shape, omega.shape, res.shape, euc.shape)

# # num = 34
# # euc = E_test[num, :]
# # res = R_test[num,:]

# #-----------------------------------------------------------------
# ## The Metrics
# from matplotlib.lines import Line2D
# import matplotlib.font_manager as font_manager
# large = 11; med = 11; small = 11
# marker_size = 1.01
# lw = 1
# inten = 0.4
# index = np.arange(0,151, 1)
# # list_best_index = [1540, 1600, 1720]
# labels = ['Ent-NN', 'Original', 'MaxEnt', 'Phys-NN']
# label_type = ['best', 'median', 'worst']
# xlims =[[0,750], [0,900], [0,900]]
# ylims =[[0,0.0035], [0,0.004], [0,0.1]]



# #----------------------------------
# # # Gather the reconstructions from my model
# # ------------------------------
# from matplotlib.lines import Line2D
# import matplotlib.font_manager as font_manager
# large = 11; med = 11; small = 11
# marker_size = 1.01
# lw = 0.1
# inten = 0.4
# def cm2inch(value):
#     return value/2.54
# plt.style.use('seaborn-white')
# COLOR = 'darkslategray'
# params = {'axes.titlesize': small,
#         'legend.fontsize': small,
#         'figure.figsize': (cm2inch(36),cm2inch(13.5)),
#         'axes.labelsize': med,
#         'axes.titlesize': small,
#         'xtick.labelsize': small,
#         'lines.markersize': marker_size,
#         'ytick.labelsize': med,
#         'figure.titlesize': small, 
#             'text.color' : COLOR,
#             'axes.labelcolor' : COLOR,
#             'axes.linewidth' : 0.5,
#             'xtick.color' : COLOR,
#             'ytick.color' : COLOR}

# plt.rcParams.update(params)
# plt.rc('text', usetex = False)
# colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', '#9467bd']
# plt.rcParams['mathtext.fontset'] = 'cm'

# #--------------------------------------------------
# ## create plots with numpy array
# fig,a =  plt.subplots(2, 3, dpi = 1200, gridspec_kw = {'wspace':0.40, 'hspace':0.30})
# for i, Err in enumerate([1e-2, 1e-3, 1e-4]):
    
#     ehat, rhat, evar, rvar, corrupted= evaluate_data__response_(rbfnetUQ_1, euc, Err, n_curves = 10000)
#     ehat_Q, rhat_Q, _,_,_= evaluate_data__response_(rbfnet_1, euc, Err, n_curves = 10000)
#     for k in range(rvar.shape[0]):
#         a[0][i].plot(omega_fine, rvar[k,:], color='lightgreen', alpha=0.5,  lw = lw)
#     a[0][i].plot(omega_fine, rhat[0,:], color = 'lightgreen', label = 'UQ-NN', linestyle='-.', linewidth = 20*lw)  
#     a[0][i].plot(omega_fine, rhat_Q[0,:], color = colors[1], linestyle = '--',label = 'Ent-NN', linewidth = 20*lw)
#     a[0][i].plot(omega_fine, res, color =colors[0],   label = 'Original(Max-Ent)', linestyle = '-',linewidth = 20*lw)
    
 

#     # a[0][i].plot(omega, res_1, color = colors[0],   label = 'Original(2rd)', linewidth = 10*lw)
#     # a[0][i].plot(omega, res_2, color=colors[0],
#     #              label='Original(1st)', linewidth=10*lw)
#     a[0][i].set_xlim([0,400])
#     a[0][i].set_xlabel('$\omega~[\mathrm{MeV}]$')
#     a[0][i].set_ylabel('$R(\omega)~[\mathrm{MeV}^{-1}]$')
#     a[0][i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
#     a[0][i].grid(linestyle=':', linewidth=0.5)
#     a[1][i].plot(tau, euc.reshape([-1]), color=colors[0],
#                  label='Original', linewidth=10*lw)
#     # for k in range(rvar.shape[0]):
#     #     a[1][i].plot(tau, evar[k,:], color = colors[1], linestyle = '-', lw = lw)
#     for k in range(rvar.shape[0]):
#         a[1][i].plot(tau, corrupted[k,:], color = 'lightskyblue', alpha=0.05, linestyle = '-.', lw = lw)
#     # a[1][i].plot(tau, ehat[0, :], color=colors[2],alpha=0.6,
#     #               linestyle='-', linewidth=lw)
#     print("The chi2 is", np.mean(((euc-ehat_Q)**2/1e-04**2), axis = 1) )
#     a[1][i].plot(tau, euc.reshape([-1]), color=colors[0],linestyle='--', linewidth=10*lw)
#     a[1][i].plot(tau, ehat_Q[0, :], color=colors[1], label='Euc(MaxEnt)', linestyle = '--', linewidth=lw)
#     a[1][i].set_xlim([0,0.0750])
#     a[1][i].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$')
#     a[1][i].set_ylabel('$E(\\tau)$')
#     a[1][i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
#     a[1][i].grid(linestyle=':', linewidth=0.5)
#     a[0][i].set_title('$\\sigma$='+str(Err))
# a[0][2].legend(loc = 'upper right',ncol=1 )
# plt.savefig('Plots/'+filename+'.pdf', bbox_inches='tight', dpi=1200)
# plt.close()



