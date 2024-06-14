import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
import torch
sys.path.append("/home/kraghavan/Projects/Nuclear/Inverse/UQNuc/torch_attention_model_v12/")

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



def print__metrics(net, loader, message, device):
    print("-----------------------------------------------------------")
    print(message)
    Err = 1e-04
    Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(net, loader, Err, device)
    # print(Rhat.shape, Ehat.shape, E_test.shape, R_test.shape, E_err.shape, R_err.shape)
    chi2 = np.round(chi2_vec(Ehat, E_test, factor= Err),10)
    Ent = np.round(-1e4*Entropy(R_test, Rhat, int_coeff = Kern_R),10)
    # ist_best_index = [best_five(Ent, 5)[0], best_five(Ent, 5)[1], best_five(-1*Ent, 10)[0]]
    print("------------------------------ The scores -----------------------------")
    ind = np.argsort(-1*Ent)
    print("The outlier is", Ent[ind[:50]], ind[:50])
    print("Min", np.min(Ent) )
    print("Median", np.median(Ent) )
    print("Max", np.max(Ent) )
    print("Mean", Ent.mean())
    print("std", Ent.std())
    print("#######################Chi 2################################# \n")
    print("Min", np.min(chi2) )
    print("Median", np.median(chi2) )
    print("Max", np.max(chi2) )
    print("Mean", chi2.mean())
    print("std", chi2.std())
    print("------------------------------ The scores  -----------------------------")
    # print(chi2[1476], Ent[1476])
    return Rhat, Ehat, R_test, E_test, ind, chi2, Ent

def data_definition():
    # --------------------------------------------------------------------------------------
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
    remove_out_1 = [461, 347, 704]
    remove_out_2 = [476, 271, 484]
#     remove_out_2=[703, 492, 516, 295, 764, 924, 254, 527, 192, 270, 841, 468, 405, 850,   0, 784, 638, 956,
#  341, 749,  41, 387, 135, 649, 693, 464, 247, 802, 872, 774, 523, 558,  15,  25, 920, 973,
#  517, 407,  44, 364, 325, 679, 670, 955, 390, 436,   6, 508, 785, 162]
    # remove_out_1 =  [637, 74, 901, 902, 70]
    # remove_out_2 = [476, 388, 402, 404, 407]
    # The test data (one peak)
    # -------------------------------------------------------
    ## The data
    R_test_1 = np.concatenate([  P['One_Peak_R']], axis=0)
    R_test_1 = np.delete(R_test_1, remove_out_1, 0)
    R_test_2 = np.concatenate([  P['Two_Peak_R']], axis=0)
    R_test_2 = np.delete(R_test_2, remove_out_2, 0)
    E_test_1 = np.concatenate([  P['One_Peak_E']], axis=0)
    E_test_1 = np.delete(E_test_1, remove_out_1, 0)
    E_test_2 = np.concatenate([  P['Two_Peak_E']], axis=0)
    E_test_2 = np.delete(E_test_2, remove_out_2, 0)
    E_test_1 = torch.from_numpy(E_test_1)
    R_test_1 = torch.from_numpy(R_test_1)
    E_test_2 = torch.from_numpy(E_test_2)
    R_test_2 = torch.from_numpy(R_test_2)
    # Gather the reconstructions from MEM
    R_MEM_1  = P['One_Peak_R_MEM']
    # R_MEM_1 = np.delete(R_MEM_1, remove_out_1, 0)
    E_MEM_1  = P['One_Peak_E_MEM']
    # E_MEM_1 = np.delete(E_MEM_1, remove_out_1, 0)
    E_MEM_2  = P['Two_Peak_E_MEM']
    # E_MEM_2 = np.delete(E_MEM_2, remove_out_2, 0)
    R_MEM_2  = P['Two_Peak_R_MEM']
    R_MEM_2 = np.delete(R_MEM_2, remove_out_2, 0)
    E_MEM_1 = np.delete(E_MEM_1, remove_out_1, 0)
    E_MEM_2 = np.delete(E_MEM_2, remove_out_2, 0)
    R_MEM_1 = np.delete(R_MEM_1, remove_out_1, 0)
    R_MEM = np.concatenate([R_MEM_1, R_MEM_2], axis=0) 
    E_MEM = np.concatenate([E_MEM_1, E_MEM_2], axis=0) 
    R_test = np.concatenate([R_test_1, R_test_2], axis=0) 
    E_test = np.concatenate([E_test_1, E_test_2], axis=0) 
    E_test   = torch.from_numpy(E_test)
    R_test   = torch.from_numpy(R_test)
    testset_one = MyDataset(E_test_1, R_test_1)
    testloader_one = DataLoader(testset_one, batch_size=128, shuffle=False)
    testset_two = MyDataset(E_test_2, R_test_2)
    testloader_two = DataLoader(testset_two, batch_size=128, shuffle=False)
    testset= MyDataset(E_test, R_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)
    #### We got our basis and it is good
    Kern = Kern.astype('float64')
    torch.pi = torch.acos(torch.zeros(1)).item() 
    print("sizes", E_MEM_1.shape, E_MEM_2.shape, E_test_1.shape, E_test_2.shape, R_MEM_1.shape, R_MEM_2.shape, R_test_1.shape, R_test_2.shape,)
    return testloader_one, testloader_two, testloader, Kern, Kern_R, omega_fine, omega, tau, R_MEM, E_MEM, R_MEM_1, E_MEM_1, R_MEM_2, E_MEM_2


#-----------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
testloader_one, testloader_two, testloader, Kern, Kern_R, omega_fine, omega,\
    tau, R_MEM, E_MEM, R_MEM_1, E_MEM_1, R_MEM_2, E_MEM_2 = data_definition()
path = '/home/kraghavan/Projects/Nuclear/Inverse/UQNuc/torch_attention_model_v12/Results/'
print("Model Definitions")
# ----------------------------------------------------------------------------
rbfnet_1 = Network(Kern, Kern_R).double()
file = path+'models/0_Trained_model_7'
rbfnet_1 = resume(file, rbfnet_1, device)
rbfnet_1.to(device)
# ----------------------------------------------------------------------------
rbfnetUQ_1 = Network(Kern, Kern_R).double()
file = path+'models/0_Trained_model_8'
rbfnetUQ_1 = resume(file, rbfnetUQ_1, device)
rbfnetUQ_1.to(device)
# -----------------------------------------------------------------
rbfPRC_1 = NetworkPRC(Kern, Kern_R).double()
file = '/gpfs/jlse-fs0/users/kraghavan/Inverse/UQNuc/torch_attention_model_v11/results/models/modella_both__PRC_00'
rbfPRC_1.load_state_dict(torch.load(file))
rbfPRC_1.to(device)


## Model definitions
print("------------------------ Two Peak-------------------------------------")
# -----------------------------------------------------------------
rbfnet_2 = Network(Kern, Kern_R).double()
file = path+'models/1_Trained_model_9'
rbfnet_2 = resume(file, rbfnet_2, device)
rbfnet_2.to(device)
# -----------------------------------------------------------------
rbfnetUQ_2 = Network(Kern, Kern_R).double()
file = path+'models/1_Trained_model_2'
rbfnetUQ_2 = resume(file, rbfnetUQ_2, device)
rbfnetUQ_2.to(device)
# -----------------------------------------------------------------
rbfPRC_2 = NetworkPRC(Kern, Kern_R).double()
file = '/gpfs/jlse-fs0/users/kraghavan/Inverse/UQNuc/torch_attention_model_v11/results/models/modella_both__PRC_00'
rbfPRC_2.load_state_dict(torch.load(file))
rbfPRC_2.to(device)


# print("------------------------ One Peak-------------------------------------")
# # ---------------------------------------------------------------------------
# Rhat, Ehat, R_test, E_test, ind, chi2, Ent = print__metrics(
#     rbfnet_1, testloader_one, "Metrics EntNN -- One Peak", device)
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response_one" +str(i)+".png", dpi=1200)
# # #     plt.close()
# # -----------------------------------------------------------------
# Rhat, Ehat, R_test, E_test, ind, chi2, Ent = print__metrics(
#     rbfnetUQ_1, testloader_one, "Metrics UQNN -- One Peak", device)
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response_one" +str(i)+".png", dpi=1200)
# # #     plt.close()
# # -----------------------------------------------------------------
# Rhat, Ehat, R_test, E_test, ind, chi2, Ent= print__metrics(
#     rbfPRC_1, testloader_one, "Metrics PRC -- One Peak", device)
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response_one" +str(i)+".png", dpi=1200)
# # #     plt.close()


# print("------------------------ Two Peak--------------------------")
# #-----------------------------------------------------------------
# Rhat, Ehat, R_test, E_test, ind, chi2, Ent = print__metrics(
#     rbfnet_2, testloader_two, "Metrics EntNN -- Two Peak", device)
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response_two" +str(i)+".png", dpi=1200)
# # #     plt.close()

# # -----------------------------------------------------------------
# Rhat, Ehat, R_test, E_test, ind, chi2, Ent = print__metrics(
#     rbfnetUQ_2, testloader_two, "Metrics UQNN -- Two Peak", device)
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response_two" + str(i)+".png", dpi=1200)
# # #     plt.close()


# # -----------------------------------------------------------------
# Rhat, Ehat, R_test, E_test, ind, chi2, Ent= print__metrics(
# rbfPRC_2, testloader_two, "Metrics PRC -- Two Peak", device)


# # exit()
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response_two" + str(i)+".png", dpi=1200)
# # #     plt.close()
# # # print("------------------------ Combined--------------------------")
# # # # -----------------------------------------------------------------
# # # rbfnet = Network(Kern, Kern_R).double()
# # # file = path+'Plotting__models/Both/mode_SVD_both_atepoch_100.pt'
# # # rbfnet = resume(file, rbfnet, device)
# # # rbfnet.to(device)
# # # # -----------------------------------------------------------------
# # # rbfnetUQ = Network(Kern, Kern_R).double()
# # # file = path+'Plotting__models/Both/modella_uncert_atepoch_710.pt'
# # # rbfnetUQ = resume(file, rbfnetUQ, device)
# # # rbfnetUQ.to(device)
# # # #-----------------------------------------------------------------
# # # Rhat, Ehat, R_test, E_test, ind, chi2, Ent = print__metrics(
# # #     rbfPRC, testloader, "Metrics EntNN -- combined" )
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response" +str(i)+".png", dpi=1200)
# # #     plt.close()

# # # #-----------------------------------------------------------------
# # # Rhat, Ehat, R_test, E_test, ind, chi2, Ent = print__metrics(\
# # #     rbfnetUQ, testloader,"Metrics UQNN -- combined" )
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response" +str(i)+".png", dpi=1200)
# # #     plt.close()

# # # # -----------------------------------------------------------------
# # # Rhat, Ehat, R_test, E_test, ind, chi2, Ent= print__metrics(
# # #     rbfPRC, testloader, "Metrics PRC -- combined")
# # # ind = np.flip(ind)
# # # for i in range(15):
# # #     print(chi2[ind[i]], Ent[ind[i]], ind[i])
# # #     fig, a = plt.subplots(2, 1)
# # #     a[0].plot(omega_fine, R_test[ind[i]], label='R')
# # #     a[0].plot(omega_fine, Rhat[ind[i]], label='Rhat')
# # #     # a[0].set_xlim([0, 100])
# # #     a[1].plot(tau, E_test[ind[i]], label='E')
# # #     a[1].plot(tau, Ehat[ind[i]], label='Ehat')
# # #     plt.savefig("test__plot_response" +str(i)+".png", dpi=1200)
# # #     plt.close()

print("---------------------------------------------------------------------------")
print("All the scores are above")
print("The next are the plots")
print("---------------------------------------------------------------------------")

## One peak stuff
##--------------------------------------------------------------
print("------------------------Both Peak, plot-------------------------------------")
loader = testloader
# # Gather the reconstructions from my model
Err = 1e-04


Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet_2, loader, Err, device)
Rhat_PRC, Ehat_PRC, E_test_PRC, R_test_PRC, E_err_PRC, R_err_PRC = evaluate_data(
    rbfPRC_2, loader, Err, device)
chi2 = np.round(chi2_vec(Ehat, E_test, factor=Err), 6)
R2 = np.round(1-(get_r2_numpy_manual_vec(Rhat, R_test)), 6)
Ent = np.round(-1*Entropy(R_test, Rhat, int_coeff=Kern_R), 6)
Ent_PRC = np.round(-1*Entropy(R_test, Rhat, int_coeff=Kern_R), 6)
list_best_index = [best_five(Ent, 5)[0],  1440,  1250]
print("The chi2 are", chi2[list_best_index])
print("The entropy are", Ent_PRC[list_best_index])
Rhat = np.expand_dims(Rhat, axis = 0)
R  = np.expand_dims(R_test, axis = 0)
Ehat = np.expand_dims(Ehat, axis = 0)
E  = np.expand_dims(E_test, axis = 0)
Rhat_PRC = np.expand_dims(Rhat_PRC, axis = 0)
R_PRC  = np.expand_dims(R_test_PRC, axis = 0)
Ehat_PRC = np.expand_dims(Ehat_PRC, axis = 0)
E_PRC  = np.expand_dims(E_test_PRC, axis = 0)

n_models__ = np.arange(10)

#--------------------------------------------------------------
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
xlims =[[0,750], [0,200], [0,200]]
ylims =[[0,0.0035], [0,0.005], [0,0.010]]
def cm2inch(value):
    return value/2.54
plt.style.use('seaborn-white')
COLOR = 'darkslategray'
params = {'axes.titlesize': small,
          'legend.fontsize': small,
          'figure.figsize': (cm2inch(32),cm2inch(18)),
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
color = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', '#9467bd']
plt.rcParams['mathtext.fontset'] = 'cm'

# create plots with numpy array
fig,a =  plt.subplots(2,3, dpi = 1200,\
gridspec_kw = {'wspace':0.35, 'hspace':0.35})

for i,element in enumerate(list_best_index):
    mean_R = np.mean(Rhat[:,element,:], axis = 0)
    mean_R_PRC = np.mean(Rhat_PRC[:,element,:], axis = 0)
    std_R  = np.std( Rhat[:,element,:], axis = 0)


    org_R  = R[0, element,:]
    mem_R = R_MEM[element,:]
    omega=omega_fine.reshape([-1])
    tau=tau.reshape([-1])

    mean_E = np.mean(Ehat[:,element,:], axis = 0)
    mean_E_PRC = np.mean(Ehat_PRC[:,element,:], axis = 0)
    std_E  = np.std(Ehat[:,element, :], axis = 0)
    org_E  = E[0, element,:]
    mem_E  = E_MEM[element,:]

    ehat    = np.zeros([10,151])
    rhat    = np.zeros([10,2000])
    for j in n_models__:
        rbfnetT = Network(Kern, Kern_R).double()
        file = path+'models/1_Trained_model_'+str(j)
        rbfnetT = resume(file, rbfnetT, device)
        rbfnetT.to(device)
        ehat[j,:], rhat[j,:], _,_,_= evaluate_data__response_(rbfnetT,\
            torch.from_numpy(org_E), 1e-04, n_curves = 1)
        # print(j)


    stdE = np.std(ehat, axis =0)
    stdR = np.std(rhat, axis =0)*5

    
    a[0][i].plot(omega, org_R,      color = color[0], linewidth = lw, linestyle = '-',label = labels[1])
    a[0][i].plot(omega, mean_R,     color = color[1], linewidth = lw, linestyle = '--',label = labels[0])
    # a[0][i].errorbar(omega, mean_R,     yerr=stdR, color = color[1], linewidth = lw, linestyle = '--')
    a[0][i].fill_between(omega, (mean_R+stdR), (mean_R-stdR), alpha=inten, color = color[1])
    
    a[0][i].plot(omega, mean_R_PRC, color = color[3], linewidth = lw, linestyle = ':',label = labels[3])
    a[0][i].plot(omega, mem_R,      color = color[2], linewidth = lw, linestyle = '-.',label = labels[2])
    
    txt = "{:.2E}".format(Ent[element])
    a[0][i].set_xlabel( '$\omega~[\mathrm{MeV}]$\n  $S_R$: '+str(txt) )
    a[0][i].set_ylabel('$R(\omega)~[\mathrm{MeV}^{-1}]$')


    

    a[1][i].errorbar(tau[index], org_E[index], yerr = 0.0001, fmt = 'x', color = color[0],\
                     ms = 0.00001, linewidth = 0.4, label = labels[1])
    a[1][i].scatter(tau, org_E, color = color[0])
    a[1][i].fill_between(tau, (mean_E+stdE), (mean_E-stdE), alpha=inten, color = color[1])
    a[1][i].plot(tau, mean_E, color = color[1], linewidth = lw, linestyle = '--', label = labels[0])
    a[1][i].plot(tau, mean_E_PRC, color = color[3], linewidth = lw, linestyle = ':', label = labels[3])
    a[1][i].plot(tau, mem_E, color = color[2], linewidth = lw, linestyle = '-.', label = labels[2])
    
    a[0][i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
    a[0][i].spines["top"].set_visible(False)    
    a[0][i].spines["bottom"].set_visible(True)    
    a[0][i].spines["right"].set_visible(False)    
    a[0][i].spines["left"].set_visible(True)  
    a[0][i].grid(linestyle=':', linewidth=0.5)

    a[1][i].spines["top"].set_visible(False)  
    a[1][i].spines["bottom"].set_visible(True)    
    a[1][i].spines["right"].set_visible(False)    
    a[1][i].spines["left"].set_visible(True)  
    a[1][i].grid(linestyle=':', linewidth=0.5)


    a[1][i].set_yscale('log')
    a[1][i].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$')
    a[1][i].set_ylabel('$E(\\tau)$')
    a[0][i].set_xlim(xlims[i])
    a[0][i].set_ylim(ylims[i])
    
    # a[1][i].set_yticks([ 0.01, 0.1, 1])
a[0][2].legend(loc = 'upper right',ncol=2)
plt.savefig('Plots/one_peak_basis.pdf', bbox_inches='tight', pad_inches = 0, dpi=1200) 
plt.close()


exit()

# #-----------------------------------------------------------------------------------
# # Uncertainty Plots
# #-----------------------------------------------------------------------------------
# plot_uncert(rbfnetUQ_2, Kern,Kern_R, testloader_two, filename='Plots/plots_uncert_two', index=250)
# plot_uncert(rbfnetUQ_1, Kern,Kern_R, testloader_one, filename='Plots/plots_uncert_one', index=50)
# # -----------------------------------------------------------------------------------


# #--------------------------------------------------------------------------------------
# # ## BOX PLOTS
# # #---------------------------------------------------------------------------------------------------
loader = testloader_one
# ### Load the required model
## Gather the reconstructions from my model
Err = 1e-04
RhatUQ, EhatUQ, E_testUQ, R_testUQ, E_errUQ, R_errUQ = evaluate_data(rbfnetUQ_1, loader, Err, device)
# # Gather the reconstructions from my model
Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet_1, loader, Err, device)
# # Gather the reconstructions from PRC
Rhat_PRC, Ehat_PRC, E_test_PRC, R_test_PRC, E_err_PRC, R_err_PRC = evaluate_data(rbfPRC_2, loader, Err, device)
chi2MEM = np.round(chi2_vec(E_MEM_1, E_test, factor= Err),6)
EntMEM = np.round(-1*Entropy(R_test, R_MEM_1, int_coeff = Kern_R),6)
chi2 = np.round(chi2_vec(Ehat, E_test, factor=Err), 6)
Ent = np.round(-1*Entropy(R_test, Rhat, int_coeff = Kern_R),6)
chi2UQ = np.round(chi2_vec(EhatUQ, E_testUQ, factor= Err),6)
EntUQ = np.round(-1*Entropy(R_testUQ, RhatUQ, int_coeff = Kern_R),6)
chi2PRC = np.round(chi2_vec(Ehat_PRC,  E_test_PRC, factor= Err),6)
EntPRC = np.round(-1*Entropy(R_test_PRC, Rhat_PRC, int_coeff = Kern_R),6)


# # # --------------------------------------------------------------------------------------
# # # Box plot_responses
labels = ['\nMEM',  '\nPhys-NN', '\nEnt-NN', '\nUQ-NN']
MEM_data = np.concatenate([chi2MEM.reshape([-1,1]), chi2PRC.reshape([-1,1]), chi2.reshape([-1,1]), chi2UQ.reshape([-1,1])], axis = 1)
filename = 'Plots/One_Peak_chi2_BOX'
title = ' One Peak $\chi_E^{2}$\n'
Box_plot(MEM_data, labels, [0.5,50], filename, spacing = 0.6, log_scale = True, yticks = [1, 10, 50], title = title)
#--------------------------------------------------------------------------------------
# Box plot_responses
MEM_data = np.concatenate([EntMEM.reshape([-1,1]), EntPRC.reshape([-1,1]), Ent.reshape([-1,1]), EntUQ.reshape([-1,1])], axis = 1)
filename = 'Plots/ONE_Peak_entropy_BOX'
title = 'One Peak $S_R$\n'
Box_plot(MEM_data, labels, [5e-6,1e-2], filename, spacing = 0.6, log_scale = True, yticks = [1e-05, 1e-04], title=title)



## --------------------------------------------------------------------------------------
######### Two Peak
loader = testloader_two
# # Gather the reconstructions from my model
Err = 1e-04
RhatUQ, EhatUQ, E_testUQ, R_testUQ, E_errUQ, R_errUQ = evaluate_data(rbfnetUQ_2, loader, Err, device)
# # Gather the reconstructions from my model
Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet_2, loader, Err, device)
# # Gather the reconstructions from PRC
Rhat_PRC, Ehat_PRC, E_test_PRC, R_test_PRC, E_err_PRC, R_err_PRC = evaluate_data(rbfPRC_2, loader, Err, device)




chi2MEM = chi2_vec(E_MEM_2, E_test, factor= Err)
EntMEM = -1*Entropy(R_test, R_MEM_2, int_coeff = Kern_R)
chi2 = chi2_vec(Ehat, E_test, factor=Err)
Ent= -1*Entropy(R_test, Rhat, int_coeff = Kern_R)
chi2UQ = chi2_vec(EhatUQ, E_testUQ, factor=Err)
EntUQ = -1*Entropy(R_testUQ, RhatUQ, int_coeff = Kern_R)
chi2PRC = chi2_vec(Ehat_PRC,  E_test_PRC, factor= Err)
EntPRC = -1*Entropy(R_test_PRC, Rhat_PRC, int_coeff = Kern_R)
#--------------------------------------------------------------------------------------------------------------------------------
MEM_data = np.concatenate([chi2MEM.reshape([-1,1]), chi2PRC.reshape([-1,1]), chi2.reshape([-1,1]),  chi2UQ.reshape([-1,1])], axis = 1)
filename = 'Plots/two_Peak_chi2_BOX'
title = ' Two Peak $\chi_E^{2}$ '
Box_plot(MEM_data, labels, [0.4,100], filename, spacing = 0.65, log_scale = True,  yticks = [0.5, 10], title=title)

# --------------------------------------------------------------------------------------------------------------------------------
MEM_data = np.concatenate([EntMEM.reshape([-1,1]), EntPRC.reshape([-1,1]), Ent.reshape([-1,1]), EntUQ.reshape([-1,1])], axis = 1)
filename = 'Plots/two_Peak_entropy_BOX'
title = 'Two Peak $S_R$'
Box_plot(MEM_data, labels, [5e-6,1e-1], filename, spacing = 0.65, log_scale = True, \
         yticks = [1e-05, 1e-03], title=title)










# # #---------------------------------------------------------------------------------------------------
# # #############Correlation plots
# # ##########One Peak

loader = testloader_one
# # Gather the reconstructions from my model
Err = 1e-04
RhatUQ, EhatUQ, E_testUQ, R_testUQ, E_errUQ, R_errUQ = evaluate_data(rbfnetUQ_1, loader, Err, device=device)
# # Gather the reconstructions from my model
Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet_1, loader, Err, device=device)
# # Gather the reconstructions from PRC
Rhat_PRC, Ehat_PRC, E_test_PRC, R_test_PRC, E_err_PRC, R_err_PRC = evaluate_data(rbfPRC_1, loader, Err, device=device)
chi2MEM = np.round(chi2_vec(E_MEM_1, E_test, factor= Err),6)
EntMEM = np.round(-1*Entropy(R_test, R_MEM_1, int_coeff = Kern_R),6)
chi2 = np.round(chi2_vec(Ehat, E_test, factor=Err), 6)
Ent = np.round(-1*Entropy(R_test, Rhat, int_coeff = Kern_R),6)
chi2UQ = np.round(chi2_vec(EhatUQ, E_testUQ, factor=Err), 6)
EntUQ = np.round(-1*Entropy(R_testUQ, RhatUQ, int_coeff = Kern_R),6)
chi2PRC = chi2_vec(Ehat_PRC,  E_test, factor=0.0001)
EntPRC = -1*Entropy(R_test, Rhat_PRC, int_coeff = Kern_R)


# ##################### 
Corr_plot(chi2, Ent, filename = 'Plots/corr_chi2_Entropy_basis_one', x_inch = 4.1, y_inch = 4.1, \
          xlabel = '$\chi^{2}_{E}$',\
          ylabel = '$S_R$', \
          yticks = [0.00001, 0.001], xticks = [10, 100],\
          ylim = [0.00001, 0.002], xlim = [0, 100],\
          labels = 'Ent-NN', save_fig = True, log_scale = True)


# ##################### 
Corr_plot(chi2UQ, EntUQ, filename = 'Plots/corr_chi2_Entropy_UQ_one', x_inch = 4.1, y_inch = 4.1, \
          xlabel = '$\chi^{2}_{E}$',\
          ylabel = '$S_R$', \
          yticks = [0.0001, 0.001], xticks = [10, 100],\
          ylim = [0.0001, 0.002], xlim = [0, 150],\
          labels = 'Ent-NN', save_fig = True, log_scale = True)


# # # # # # # ###############################
# # # # # # # Corr_plot(chi2MEM, EntMEM, filename = 'Plots/corr_chi2_Entropy_MEM_one', x_inch = 4.1, y_inch = 4.1, \
# # # # # # #           xlabel = '$\\overline{\chi^{2}_{E}}$',\
# # # # # # #           ylabel = '$\\overline{S_R}$', \
# # # # # # #           yticks = [0.0001, 0.001, 0.1], xticks = [0.1, 1, 10 ],\
# # # # # # #           ylim = [0.000001, 0.01], xlim = [0.1, 10],\
# # # # # # #           labels = 'Ent-NN', save_fig = True, log_scale = True)


# # # # # # ###############################
# # # # # # # Corr_plot(chi2PRC, EntPRC, filename = 'Plots/corr_chi2_Entropy_PRC_one', x_inch = 4.1, y_inch = 4.1, \
# # # # # # #           xlabel = '$\\chi^{2}_{E}$',\
# # # # # # #           ylabel = '$S_R$', \
# # # # # # #           yticks = [0.00001, 0.0001, 0.001, 0.1], xticks = [10, 100, 200],\
# # # # # # #           ylim = [0.000001, 0.01], xlim = [10, 200],\
# # # # # # #           labels = 'Ent-NN', save_fig = True, log_scale = True)



# # # # # # ###############################
# # # # # # # Box plot_responses
# # # # # # labels = ['MEM',  'Phys-NN', 'Ent-NN', 'UQ-NN']
# # # # # # MEM_data = np.concatenate([chi2MEM.reshape([-1,1]), chi2PRC.reshape([-1,1]), chi2.reshape([-1,1]), chi2UQ.reshape([-1,1])], axis = 1)
# # # # # # filename = 'Plots/One_Peak_chi2_BOX'
# # # # # # title = ' One Peak ( $\chi_E^{2}$ )'
# # # # # # Box_plot(MEM_data, labels, [0.5,100], filename, spacing = 0.8, log_scale = True, yticks = [1, 10, 100, 300, 500], title = title)


# # # # # # ###############################
# # # # # # # Box plot_responses
# # # # # # MEM_data = np.concatenate([EntMEM.reshape([-1,1]), EntPRC.reshape([-1,1]), Ent.reshape([-1,1]), EntUQ.reshape([-1,1])], axis = 1)
# # # # # # filename = 'Plots/ONE_Peak_entropy_BOX'
# # # # # # title = 'One Peak ($S_R$ )'
# # # # # # Box_plot(MEM_data, labels, [0.5,0.9], filename, spacing = 1.3, log_scale = True, yticks = [1e-06, 1e-02], title=title)



# ############################################################
# ####### Two Peak
loader = testloader_two
# # Gather the reconstructions from my model
Err = 1e-04
RhatUQ, EhatUQ, E_testUQ, R_testUQ, E_errUQ, R_errUQ = evaluate_data(rbfnetUQ_2, loader, Err, device=device)
# # Gather the reconstructions from my model
Rhat, Ehat, E_test, R_test, E_err, R_err = evaluate_data(rbfnet_2, loader, Err, device=device)
# # Gather the reconstructions from PRC
Rhat_PRC, Ehat_PRC, E_test_PRC, R_test_PRC, E_err_PRC, R_err_PRC = evaluate_data(rbfPRC_2, loader, Err, device=device)
chi2MEM = np.round(chi2_vec(E_MEM_2, E_test, factor= Err),6)
EntMEM = np.round(-1*Entropy(R_test, R_MEM_2, int_coeff = Kern_R),6)
chi2 = np.round(chi2_vec(Ehat, E_test, factor= Err),6)
Ent = np.round(-1*Entropy(R_test, Rhat, int_coeff = Kern_R),6)
chi2UQ = np.round(chi2_vec(EhatUQ, E_testUQ, factor= Err),6)
EntUQ = np.round(-1*Entropy(R_testUQ, RhatUQ, int_coeff = Kern_R),6)
chi2PRC = chi2_vec(Ehat_PRC,  E_test, factor= Err)
EntPRC = -1*Entropy(R_test, Rhat_PRC, int_coeff = Kern_R)
# #----------------------------------------------------------------------
Corr_plot(chi2UQ, EntUQ, filename = 'Plots/corr_chi2_Entropy_basis_two', x_inch = 4, y_inch = 4, \
          xlabel = '$\chi^{2}_{E}$',\
          ylabel = '$S_R$', \
          yticks = [0.00001, 0.001], xticks = [10, 100],\
          ylim = [0.00001,  0.002], xlim = [0, 1050],\
          labels = 'Ent-NN', save_fig = True, log_scale = True)

# #############################
Corr_plot(chi2UQ, EntUQ, filename = 'Plots/corr_chi2_Entropy_UQ_two', x_inch = 4, y_inch = 4, \
          xlabel = '$\chi^{2}_{E}$',\
          ylabel = '$S_R$', \
          yticks = [0.00001, 0.001], xticks = [10, 100],\
          ylim = [0.00001,  0.002], xlim = [0, 1050],\
          labels = 'Ent-NN', save_fig = True, log_scale = True)

