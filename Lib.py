import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.font_manager as font_manager
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
import numpy as np

import torch_optimizer as optim
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





# Data importing functions
def return_dict(i):
    import pickle
    with (open(i, "rb")) as openfile:
        while True:
            try:
                x = pickle.load(openfile)
            except EOFError:
                break
    return x



# First define a function that finds the nearest neighbors
def x_near(x, x_v):
    x_d = np.abs(x_v - x)
    x_index = np.argsort(x_d)
    return x_index

# Construct the interpolating polynomials up to second order
def polint_quadratic(x, x_v):
    n = x.shape[0]
    n_v = x_v.shape[0]
    poly = np.zeros((n,3))
    x_index = np.zeros((n, n_v), dtype=int)
    for i in range (n):
        x_index[i] = x_near(x[i], x_v)
        x_i = x_v[x_index[i, 0:3]]   
        poly[i,0] = (x[i]-x_i[1])*(x[i]-x_i[2])/(x_i[0]-x_i[1])/(x_i[0]-x_i[2])
        poly[i,1] = (x[i]-x_i[0])*(x[i]-x_i[2])/(x_i[1]-x_i[0])/(x_i[1]-x_i[2])
        poly[i,2] = (x[i]-x_i[0])*(x[i]-x_i[1])/(x_i[2]-x_i[0])/(x_i[2]-x_i[1])
    return poly, x_index

# Actual interpolation function
def interp_quadratic(x, x_v, y_v, poly, x_index):
    y = np.zeros(x.shape[0])
    for i in range (x.shape[0]):
        y[i] = np.sum( poly[i, 0:3] * y_v[x_index[i, 0:3]])
    return y


def interpolate_Alessandro(omega_fine, omega_, R_):
    omega_fine = omega_fine.reshape([-1])
    R_temp = np.zeros([R_.shape[0],omega_fine.shape[0]])
    poly, x_index = polint_quadratic(omega_fine.reshape([-1]), omega_.reshape([-1]))
    print("I am starting the main interpolation loop")
    for i in range(R_.shape[0]):
        R_temp[i,:] = interp_quadratic(omega_fine.reshape([-1]),\
         omega_.reshape([-1]), R_[i,:].reshape([-1]), poly, x_index)
    return R_temp, poly, x_index

def interpolate_integrate_self(tau_, omega_, R_, sigma_E = 0.0001, nw = 2000, wmax =2000):
    n_tau = tau_.shape[0]
    n_points = R_.shape[0]
    # Get the fine grid
    dw = wmax / nw
    omega_fine  = np.zeros(nw)
    for i in range (nw):
        omega_fine[i] = (i + 0.5) * dw
    # Get the integration coefficients.
    omega_fine = omega_fine.reshape([-1,1])
    index = np.argsort(omega_)
    nw = omega_fine.shape[0]
    Kern = np.zeros([n_tau, nw])
    Kern_R = np.zeros([1, nw])
    for i in range(n_tau):
        for j in range((nw-1)):
            Kern[i,j]  =  np.exp(-1*omega_fine[j,0]*tau_[i,0])
    E_      = np.zeros([n_points,n_tau])
    R_temp, poly, x_index  =  interpolate_Alessandro(omega_fine, omega_, R_)
    E_      = np.transpose(np.matmul(Kern, np.transpose(R_temp) ) )
    R_temp[R_temp<1e-08] = 1e-08
    return E_, R_temp, omega_fine, Kern, Kern_R, poly, x_index

def integrate(tau_, omega_, omega_fine, R_, sigma_E = 0.0001, nw = 2000, wmax =2000):
    n_tau = tau_.shape[0]
    n_points = R_.shape[0]
    dw = wmax / nw
    omega_fine = omega_fine.reshape([-1,1])
    index = np.argsort(omega_)
    nw = omega_fine.shape[0]
    Kern = np.zeros([n_tau, nw])
    Kern_R = np.zeros([1, nw])
    for i in range(n_tau):
        for j in range((nw-1)):
            Kern[i,j]  =  np.exp(-1*omega_fine[j,0]*tau_[i,0])*dw   
    E_      = np.zeros([n_points,n_tau])
    # R_[R_<1e-08] = 1e-08
    E_      = np.transpose(np.matmul(Kern, np.transpose(R_) ) )
    return E_, R_, Kern

# The Metric calculation
def chi2_vec(y, yhat, factor):
    return np.mean(((y-yhat)**2/factor**2), axis = 1)


def get_r2_numpy_manual_vec(y_true, y_pred):
    ybar = np.mean(y_true, axis = 1).reshape([-1,1])
    SS_res = np.sum(np.square(y_pred - y_true), axis = 1) 
    SS_tot = np.sum(np.square(y_true - ybar), axis = 1) 
    r2 = (1-(SS_res/(SS_tot+0.00000001)))
    return r2


def Entropy(x, xhat, int_coeff = 1):
    import numpy as np
    return np.sum(xhat-x-xhat*np.log( np.divide(xhat,x) ) , axis = 1)


def best_five(x, n): 
    import numpy as np 
    return np.argsort(x)[:n]


def worst_five(x, n): 
    import numpy as np
    ranked = np.argsort(x)
    return ranked[::-1][:n]


# Let us create some plots with these
def Corr_plot(x, y, filename, x_inch, y_inch, xlabel, ylabel, labels, xlim = None, ylim = None, save_fig = False, log_scale = False):
    
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    
    def cm2inch(value):
            return value/2.54
        
    small = 10
    med = 10
    large = 12
    
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
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
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
    fig,a =  plt.subplots(1,1, sharex = False, dpi = 1000, gridspec_kw = {'wspace':0.7, 'hspace':0.7})
    # Some Plot oriented settings 
    a.spines["top"].set_visible(True)    
    a.spines["bottom"].set_visible(True)    
    a.spines["right"].set_visible(True)    
    a.spines["left"].set_visible(True)  
    a.grid(linestyle=':', linewidth=0.5)
    a.get_xaxis().tick_bottom()    
    a.get_yaxis().tick_left()  
    leg = np.round(np.corrcoef(x,y)[0,1], 3)**2
    a.scatter(x, y, alpha=0.8, c=color[0], edgecolors='none', s=30, label = labels +'(Correlation ='+str(leg)+')' )
    a.set_xlabel(xlabel)
    a.set_ylabel(ylabel)
    # a.get_xaxis().get_major_formatter().scaled['100'] = '%y'
    
    
    if log_scale is True:
        a.set_yscale('log')
    
    if xlim is not None:
        a.set_xlim(xlim)
    
    if ylim is not None:
        a.set_ylim(ylim)
    # a.legend()
    a.legend(bbox_to_anchor=(0.01, -1, 0.3, 0.1), loc = 'upper left',ncol=1, markerscale=1)
    
    if save_fig is True:
        plt.savefig(filename+'.pdf', dpi = 800,  bbox_inches='tight') 
    plt.show()



def Box_plot(data, labels, lims, filename, log_scale = False, spacing = 0.4, title = 'random title'):
    import pandas as pd
    
    import matplotlib.pyplot as plt
    def cm2inch(value):
            return value/2.54
    small = 10
    med = 11
    large = 12
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(6),cm2inch(4)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
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
    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    fig,a =  plt.subplots(1,data.shape[1], sharex = False, dpi = 1000,\
                          gridspec_kw = {'wspace':spacing, 'hspace':spacing})
    c = 'slategray'

    fig.suptitle(title)
    for i in range(data.shape[1]):
        a[i].boxplot(data[:,i], patch_artist=True,
            boxprops=dict(facecolor = 'white', color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color='steelblue'),
            )

        a[i].set_xlabel(labels[i])
        a[i].set_ylim(lims)
        if log_scale is True:
            a[i].set_yscale('log')
        
    plt.savefig(filename+'.pdf', dpi = 800,  bbox_inches='tight') 
    

def vals(chi_2_values):
    print("no. of responses with chi2 E > 5:", len(chi_2_values[chi_2_values>1.5]))
    # chi_2_values = chi_2_values[chi_2_values<1]
    print("Min", np.min(chi_2_values) )
    print("Median", np.median(chi_2_values) )
    print("Max", np.max(chi_2_values) )
    print("Mean", chi_2_values.mean())
    print("std", chi_2_values.std()/np.sqrt(len(chi_2_values)))



# The Metric calculation
def chi2_vec(y, yhat, factor):
    return np.mean(((y-yhat)**2/factor**2), axis = 1)


def get_r2_numpy_manual_vec(y_true, y_pred):
    ybar = np.mean(y_true, axis = 1).reshape([-1,1])
    SS_res = np.sum(np.square(y_pred - y_true), axis = 1) 
    SS_tot = np.sum(np.square(y_true - ybar), axis = 1) 
    r2 = (1-(SS_res/(SS_tot+0.00000001)))
    return r2


def Entropy(x, xhat, int_coeff):
    import numpy as np
    return -1*np.sum(xhat-x-xhat*np.log( np.divide(xhat,x) ) , axis = 1)


def best_five(x, n): 
    import numpy as np 
    return np.argsort(x)[:n]


def worst_five(x, n): 
    import numpy as np
    ranked = np.argsort(x)
    return ranked[::-1][:n]


def plot_responses(data_E, data_R, org_E, list_index, limits, filename_list, \
                   omega, tau, labels, save_fig = False):

    # Let us create some plots with these
    import matplotlib.pyplot as plt
    def cm2inch(value):
            return value/2.54
        
    small = 18
    med = 19
    large = 20
    plt.style.use('seaborn-white')
    COLOR = 'darkslategray'
    params = {'axes.titlesize': small,
            'legend.fontsize': small,
            'figure.figsize': (cm2inch(16),cm2inch(12)),
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'font.family': "sans-serif",
            'font.sans-serif': "Myriad Hebrew",
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

    from matplotlib.lines import Line2D
    import matplotlib.font_manager as font_manager
    for k,element in enumerate(list_index):
        fig,a =  plt.subplots(2,1, sharex = False, dpi = 1000, gridspec_kw = {'wspace':0.3, 'hspace':0.3})
        # Some Plot oriented settings 
        a[0].spines["top"].set_visible(False)    
        a[0].spines["bottom"].set_visible(False)    
        a[0].spines["right"].set_visible(False)    
        a[0].spines["left"].set_visible(True)  
        a[0].grid(linestyle=':', linewidth=0.5)
        a[0].get_xaxis().tick_bottom()    
        a[0].get_yaxis().tick_left()  

        # Some Plot oriented settings 
        a[1].spines["top"].set_visible(False)    
        a[1].spines["bottom"].set_visible(False)    
        a[1].spines["right"].set_visible(False)    
        a[1].spines["left"].set_visible(True)  
        a[1].grid(linestyle=':', linewidth=0.5)
        a[1].get_xaxis().tick_bottom()    
        a[1].get_yaxis().tick_left()  


        for i, datum in enumerate(data_R):
            a[0].plot(omega,  datum[element,:], color = color[i], linewidth = 1.5, linestyle = '-.',\
            label = labels[i] )

        for i, datum in enumerate(data_E):
            chi_value = np.round( chi2_vec( org_E[element,:].reshape([-1,1]),\
            datum[element,:].reshape([-1,1]), 0.0001).mean(),3)
            
            a[1].errorbar(tau, datum[element,:], yerr = 0.0001, fmt = 'x', color = color[i], ms = 1,\
            linewidth = 1, label = '  '+labels[i]+'\n$(\chi^{2}_{E}, $'+str(chi_value)+')')


        a[0].set_xlim(limits[k])
        a[0].set_xlabel('$\omega~[MeV]$')
        a[0].set_ylabel('$R(\omega)~[MeV^{-1}]$')

        a[1].set_yscale('log')
        a[1].set_xlabel('$\\tau~[MeV^{-1}]$')
        a[1].set_ylabel('$E(\\tau)$')

        a[1].legend(bbox_to_anchor=(-0.1, -0.4, 0.3, 0.1), loc = 'upper left',ncol=3, markerscale=2 )
        
        if save_fig is True:
            plt.savefig(filename_list[k]+'.pdf', dpi = 800,  bbox_inches='tight') 
        
        plt.show()



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

# First define a function that finds the nearest neighbors
def x_near(x, x_v):
    x_d = np.abs(x_v - x)
    x_index = np.argsort(x_d)
    return x_index

# Construct the interpolating polynomials up to second order
def polint_quadratic(x, x_v):
    n = x.shape[0]
    n_v = x_v.shape[0]
    poly = np.zeros((n,3))
    x_index = np.zeros((n, n_v), dtype=int)
    for i in range (n):
        x_index[i] = x_near(x[i], x_v)
        x_i = x_v[x_index[i, 0:3]]
        poly[i,0] = (x[i]-x_i[1])*(x[i]-x_i[2])/(x_i[0]-x_i[1])/(x_i[0]-x_i[2])
        poly[i,1] = (x[i]-x_i[0])*(x[i]-x_i[2])/(x_i[1]-x_i[0])/(x_i[1]-x_i[2])
        poly[i,2] = (x[i]-x_i[0])*(x[i]-x_i[1])/(x_i[2]-x_i[0])/(x_i[2]-x_i[1])
    return poly, x_index

# Actual interpolation function
def interp_quadratic(x, x_v, y_v, poly, x_index):
    y = np.zeros(x.shape[0])
    for i in range (x.shape[0]):
        y[i] = np.sum( poly[i, 0:3] * y_v[x_index[i, 0:3]])
    return y


def interpolate_Alessandro(omega_fine, omega_, R_):
    omega_fine = omega_fine.reshape([-1])
    R_temp = np.zeros([R_.shape[0],omega_fine.shape[0]])
    poly, x_index = polint_quadratic(omega_fine.reshape([-1]), omega_.reshape([-1]))
    # /("I am starting the main interpolation loop")
    for i in range(R_.shape[0]):
        R_temp[i,:] = interp_quadratic(omega_fine.reshape([-1]),\
         omega_.reshape([-1]), R_[i,:].reshape([-1]), poly, x_index)
    return R_temp, poly, x_index

def interpolate_integrate_self(tau_, omega_, R_, sigma_E = 0.0001, nw = 2000, wmax =2000):
    n_tau = tau_.shape[0]
    n_points = R_.shape[0]
    # Get the fine grid
    dw = wmax / nw
    omega_fine  = np.zeros(nw)
    for i in range (nw):
        omega_fine[i] = (i + 0.5) * dw
    # Get the integration coefficients.
    omega_fine = omega_fine.reshape([-1,1])
    index = np.argsort(omega_)
    nw = omega_fine.shape[0]
    Kern = np.zeros([n_tau, nw])
    Kern_R = np.zeros([1, nw])
    for i in range(n_tau):
        for j in range((nw-1)):
            Kern_R[0,j] = (omega_fine[j+1,0]-omega_fine[j,0])
            Kern[i,j]  =  np.exp(-1*omega_fine[j,0]*tau_[i,0])*Kern_R[0,j]
    E_      = np.zeros([n_points,n_tau])
    R_temp, poly, x_index  =  interpolate_Alessandro(omega_fine, omega_, R_)
    E_      = np.transpose(np.matmul(Kern, np.transpose(R_temp) ) )
    R_temp[R_temp<1e-08] = 1e-08
    return E_, R_temp, omega_fine, Kern, Kern_R, poly, x_index

def integrate(tau_, omega_, omega_fine, R_, sigma_E = 0.0001, nw = 2000, wmax =2000):
    n_tau = tau_.shape[0]
    n_points = R_.shape[0]
    dw = wmax / nw
    omega_fine = omega_fine.reshape([-1,1])
    index = np.argsort(omega_)
    nw = omega_fine.shape[0]
    Kern = np.zeros([n_tau, nw])
    Kern_R = np.zeros([1, nw])
    for i in range(n_tau):
        for j in range((nw-1)):
            Kern_R[0,j] = (omega_fine[j+1,0]-omega_fine[j,0])
            Kern[i,j]  =  np.exp(-1*omega_fine[j,0]*tau_[i,0])*dw
    E_      = np.zeros([n_points,n_tau])
    R_[R_<1e-08] = 1e-08
    E_      = np.transpose(np.matmul(Kern, np.transpose(R_) ) )
    return E_, R_, Kern, Kern_R

# The Metric calculation
def chi2_vec(y, yhat, factor):
    return np.mean(((y-yhat)**2/factor**2), axis = 1)

def get_r2_numpy_manual_vec(y_true, y_pred):
    ybar = np.mean(y_true, axis = 1).reshape([-1,1])
    SS_res = np.sum(np.square(y_pred - y_true), axis = 1)
    SS_tot = np.sum(np.square(y_true - ybar), axis = 1)
    r2 = (1-(SS_res/(SS_tot+0.00000001)))
    return r2

def Entropy(x, xhat, int_coeff = 1):
    import numpy as np
    return np.sum(xhat-x-xhat*np.log( np.divide(xhat,x) ) , axis = 1)

def best_five(x, n):
    import numpy as np
    return np.argsort(x)[:n]

def worst_five(x, n):
    import numpy as np
    ranked = np.argsort(x)
    return ranked[::-1][:n]


#########################################################################
# Get the older model work out properly
# RBF Layer
class RBF(nn.Module):

    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 0.01)
        nn.init.constant_(self.sigmas, 2)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        # print(size)
        x = input.unsqueeze(1).expand(size)
        # print(x.shape)
        c = self.centres.unsqueeze(0).expand(size)
        # print(c.shape)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        return self.basis_func(distances)

def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(
        alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi
  
def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi
  
def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
        * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3)
           * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases


class NetworkPRC(nn.Module):
    def __init__(self, kern, kern_R):
        super(NetworkPRC, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        self.kern_R = torch.from_numpy(kern_R).to(device)
        self.kern = torch.from_numpy(kern).float().to(device)
        # for i in range(len(layer_widths) - 1):
        #print(layer_widths[i], layer_centres[i])

        layer_widths = [151, 200]
        layer_centres = [2000]
        basis_func = gaussian
        samples = 256

        i = 0
        self.rbf_layers.append(
            RBF(layer_widths[i], layer_widths[i+1], basis_func))
        self.linear_layers.append(
            nn.Linear(layer_widths[i+1], layer_centres[i]))
        self.linear_layers.append(
            nn.Linear(layer_centres[i], layer_centres[i]))

    ##########################################
    def forward(self, x, fac_var=0.001):
        out = x.float()
        ##########################################
        # for i in range(len(self.rbf_layers)):
        i = 0
        out = self.rbf_layers[i](out)
        out = torch.relu(self.linear_layers[i](out))
        i = 1
        out = torch.exp(self.linear_layers[i](out))
        ##########################################
        # We have convert this code to pytorch
        # print(out.shape, self.kern.shape)
        Ehat = torch.matmul(self.kern, out.transpose(0, 1)).transpose(0, 1)
        # print("EHat", Ehat.shape)
        correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000)
        ##########################################
        # print(correction_term.shape, Ehat.shape, out.shape)
        Rhat = torch.div(out, correction_term)
        # print(out.shape, self.kern.shape)
        ##########################################
        multout = torch.matmul(self.kern, Rhat.transpose(0, 1))
        # print("multout", multout.shape)
        Ehat = multout.transpose(0, 1)
        # print("multout", Ehat.shape)
        return Ehat, Rhat, Ehat, Rhat, x

    ####################################################################################
    def loss_func(self, Ehat, ENhat, Rhat,  E, EN, R, fac):
        if fac[2]>0:
            # The R loss
            non_integrated_entropy = (Rhat-R-torch.mul(Rhat, torch.log(torch.div(Rhat, R))))
            loss_R = -torch.mean(non_integrated_entropy)
            loss_E = torch.mean( torch.mul((Ehat- E).pow(2), (1/(0.0001*0.0001))))\
                + torch.mean( torch.mul((EN- ENhat).pow(2), (1/(fac[2]*fac[2]))) ) 
            return (fac[0]*loss_R+fac[1]* loss_E), loss_E, loss_R
        else:
            # The R loss
            non_integrated_entropy = (Rhat-R-torch.mul(Rhat, torch.log(torch.div(Rhat, R))))
            loss_R = -torch.mean(non_integrated_entropy)
            loss_E = torch.mean( torch.mul((Ehat- E).pow(2), (1/(0.0001*0.0001))))
            return (fac[0]*loss_R+fac[1]* loss_E), loss_E, loss_R

    def evaluate_METRICS(self, testloader, fac_var, save_dir,  epoch, filee):
        self.eval()
        ### Final Numbers 
        Ent_list =[]
        chi2=[]
        for x_batch, y_batch in testloader:
            y_batch = y_batch.float().to(device)
            x_batch = x_batch.float().to(device)
            x_hat, y_hat, _, yvar, xvar = self.forward(x_batch, fac_var)
            # Calculate Entropy
            my_list = Entropy(y_batch.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), 1 )
            [Ent_list.append(item) for item in my_list]
            my_list = chi2_vec(x_batch.cpu().detach().numpy(), x_hat.cpu().detach().numpy(), 1e-04)
            [chi2.append(item) for item in my_list]


        print("#######################ENTROPIES#################################")
        # print("Entropy shapes", len(Ent_list_one), len(Ent_list_two) )
        Ent_list=np.array(Ent_list)
        Ent_list=np.array(Ent_list)
        print("Min", np.min(Ent_list) )
        print("Median", np.median(Ent_list) )
        print("Max", np.max(Ent_list) )
        print("Mean", Ent_list.mean())
        print("std", Ent_list.std())
        print("#######################Chi 2################################# \n")
        chi2=np.array(chi2)
        print("Min", np.min(chi2) )
        print("Median", np.median(chi2) )
        print("Max", np.max(chi2) )
        print("Mean", chi2.mean())
        print("std", chi2.std())

        import matplotlib.pyplot as plt
        import numpy 
        self.eval()
        import matplotlib.pyplot as plt
        import seaborn as sns

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

        small = 18
        med = 20
        large = 22
        plt.style.use('seaborn-white')
        COLOR = 'dimgrey'
        rc={'axes.titlesize': small,
            'legend.fontsize': small,
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'text.color' : COLOR,
            'axes.labelcolor' : COLOR,
            'axes.axisbelow': False,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': 'dimgrey',
            'xtick.bottom': False,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',
            'xtick.top': False,
            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': False,
            'ytick.right': False}
        plt.rcParams.update(rc)
        plt.rc('text', usetex = False)
        fig, ax = plt.subplots(3,1, figsize=(16,15) )
        ax[0].hist( 1e-05*chi2, color=CB91_Blue, label='chi squared')    
        ax[1].hist( -1e05*Ent_list, color=CB91_Green, label='Entropy')     
        ax[2].hist( 1e-05*chi2-1e05*Ent_list, color=CB91_Purple, label='chi squared+Entropy')   
        ax[0].legend(loc='upper right')
        ax[1].legend(loc='upper right')
        ax[2].legend(loc='upper right')
        fig.suptitle('Histograms', fontsize=16)
        fig.tight_layout()
        plt.savefig(save_dir+filee+str(epoch)+'_hist.png', dpi=300)
        plt.close()
        return Ent_list, chi2


    def evaluate_plots(self, testloader, omega_fine, tau, epoch, save_dir, filee, fac):
        import matplotliba.pyplot as plt
        import numpy 
        self.eval()
        import matplotlib.pyplot as plt
        import seaborn as sns

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
        COLOR = 'dimgrey'
        rc={'axes.titlesize': small,
            'legend.fontsize': small,
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'text.color' : COLOR,
            'axes.labelcolor' : COLOR,
            'axes.axisbelow': False,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': 'dimgrey',
            'xtick.bottom': False,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',
            'xtick.top': False,
            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': False,
            'ytick.right': False}
        plt.rcParams.update(rc)
        plt.rc('text', usetex = False)
        for j in range(5):
            fig, ax = plt.subplots( 4,2, figsize=(16,15) )
            for x_batch, y_batch in testloader:
                y_batch = y_batch.float().to(device)
                x_batch = x_batch.float().to(device)
                for i in range(4):
                    x_hat, y_hat, _, yvar, xvar = self.forward(x_batch, fac*pow(10,i))
                    # print(x_hat.shape, y_hat.shape, yvar.shape, xvar.shape)
                    ## PLOT THINGS ABOUT THE R
                    curve=y_batch[j,:].cpu().detach().numpy()
                    ax[i][0].plot((omega_fine).reshape([-1]), curve, '--', label='R('+str(fac*pow(10,i))+')', color='blue')    
                    Rhat = y_hat.cpu().detach().numpy()[j, :]
                    yerr = abs(Rhat-yvar.cpu().detach().numpy()[j, :])
                    fill_up = Rhat+yerr
                    fill_down = Rhat-yerr
                    fill_down[fill_down<0]= 0
                    ax[i][0].fill_between(omega_fine.reshape([-1]), fill_up, fill_down, alpha=1, color='orange')
                    ax[i][0].set_xlim([0,400])
                    ax[i][0].legend(loc='upper right')
                    ## PLOT THINGS ABOUT THE E
                    curve=x_batch[j,:].cpu().detach().numpy()
                    ax[i][1].plot( (tau).reshape([-1]), curve, label='E', color='blue')    
                    Ehat = x_hat.cpu().detach().numpy()[j, :]
                    yerr = abs(Ehat-xvar.cpu().detach().numpy()[j, :])
                    # print(Ehat.shape, tau.shape)
                    ax[i][1].errorbar(tau.reshape([-1]), Ehat, yerr=yerr, fmt='x', linewidth=2, ms = 1, label='Ehat', color='orange')
                    ax[i][1].set_yscale('log')
                    ax[i][1].legend(loc='upper right')
                    ax[i][0].set_xlabel('$ \\omega (MeV)$')
                    ax[i][1].set_xlabel('$ \\tau (MeV^{-1})$')
                    ax[i][0].set_ylabel('$ R(\\omega)(MeV^{-1})$')
                    ax[i][1].set_ylabel('$ E(\\tau)$')
                fig.suptitle('Reconstructions $R, E$ with increasing error $(\\sigma)$ in the input Euclidean', fontsize=16)
                fig.tight_layout()
                plt.savefig(save_dir+filee+str(epoch)+'_'+str(j)+".png", dpi=300)
                plt.close()
                break
        return


    ####################################################################################
    def fit(self, trainloader, testloader_one, testloader_two,  omega, tau, epochs,\
        batch_size, lr, save_dir, model_name, flag, load, last_checkpoint, configs):

        self.train()
        obs = len(trainloader)*batch_size
        LR = []
        LE = []
        # optimiser = torch.optim.RMSprop(self.parameters(),\
        #     lr=lr, weight_decay=0.0001)
        
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.99)
        print("The flag is printed here", flag)

        ####################################################################################
        if load == 1:
            checkpoint = torch.load(model_name+'_atepoch_'+str(last_checkpoint)+'.pt')
            self.load_state_dict(checkpoint['model_state_dict'])
            optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
            last_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            self.train()

        epoch = last_epoch
        epochs +=last_epoch
        print(last_epoch, epochs, epoch)
        for j in range(last_epoch):
            if epoch % int(round( (epochs)*0.2))==0:
                if factor_E<1:
                        factor_R=  factor_R*sched_factor
                        factor_E = factor_E*sched_factor
                        fac_var  = fac_var*sched_factor
                else:
                    factor_R = 1e5
                    factor_E = 1
                    fac_var  = 1
        factor_R= configs['factor_R']
        factor_E =configs['factor_E']
        fac_var  = configs['fac_var']
        sched_factor=configs['sched_factor']
        print(factor_R, factor_E, fac_var)

        while epoch < epochs:
            epoch += 1
            current_loss=0
            current_loss_E = 0
            current_loss_R = 0
            batches = 0
            progress = 0
            factor_R= configs['factor_R']
            factor_E =configs['factor_E']
            fac_var  = configs['fac_var']
            sched_factor=configs['sched_factor']
            ####################################################################################
            ####################################################################################
            if epoch % int(round( (epochs)*0.2))==0:
                if factor_E<1:
                    factor_R=  factor_R*sched_factor
                    factor_E = factor_E*sched_factor
                    fac_var  = fac_var*sched_factor
                else:
                    factor_R = 1e5
                    factor_E = 1
                    fac_var  = 1
            
            ####################################################################################
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()

                ####################################################################################
                x_batch = x_batch.double()
                x_batch   = x_batch.to(device)

                y_batch = y_batch.double().to(device)
                
                xhat, yhat, xNhat, _, x_batch_N = self.forward(x_batch)
                loss, E_L, R_L = self.loss_func( xhat, xNhat, yhat, x_batch,\
                                                x_batch_N, y_batch, \
                                                [factor_R, factor_E, fac_var] )
                current_loss      += (1/batches) * (loss.cpu().item() - current_loss)
                current_loss_E    += (1/batches) * (E_L.cpu().item() - current_loss_E)
                current_loss_R    += (1/batches) * (R_L.cpu().item() - current_loss_R)
                loss.backward()
                optimiser.step()
                progress += x_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f' %(epoch,\
                progress, obs, current_loss, current_loss_R, current_loss_E) )
                # print('\rEpoch: %d, Progress: %d/%d,\
                # Loss: %f' %(epoch, progress, obs, current_loss))
                sys.stdout.flush()
                # profiler.step()
                ####################################################################################
            
            
            # Printing and saving code 
            if epoch % 1==0:
                scheduler.step()
            if epoch % 1==0:
                if flag==0:
                    print("########################################################")
                    print("\n One Peak")
                    # torch.save(self.state_dict(), model_name)
                    _,_=self.evaluate_METRICS(testloader_one, fac_var, save_dir, epoch, filee='one_peak')
                    # self.evaluate_plots(testloader_one, omega, tau, epoch,\
                    #      save_dir, filee='one_peak', fac=fac_var)
                    
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss,
                    }, model_name+'_atepoch_'+str(epoch)+'.pt')


                elif flag==1:
                    print("########################################################")
                    print("\n Two Peak")
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss,
                    }, model_name+'_atepoch_'+str(epoch)+'.pt')
                    # torch.save(self.state_dict(), model_name)
                    _,_=self.evaluate_METRICS(testloader_two, fac_var, save_dir, epoch, filee='two_peak')
                    # self.evaluate_plots(testloader_two, omega, tau, epoch, save_dir, filee='two_peak', fac=fac_var)
                    print("########################################################")
                else:
                    print("save the checkpoint")
                    torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'loss': loss,
                    }, model_name+'_atepoch_'+str(epoch)+'.pt')


                    print("########################################################")
                    print("\n One Peak")
                    _,_=self.evaluate_METRICS(testloader_one, fac_var, save_dir, epoch, filee='one_peak')
                    # self.evaluate_plots(testloader_one, omega, tau, epoch,\
                    #      save_dir, filee='one_peak', fac=fac_var)
                    print("########################################################")
                    print("Two Peak")
                    _,_=self.evaluate_METRICS(testloader_two, fac_var, save_dir, epoch, filee='two_peak')
                    # self.evaluate_plots(testloader_two, omega, tau, epoch,\
                    #      save_dir, filee='two_peak', fac=fac_var)    
                    # torch.save(self.state_dict(), model_name)
                    print("########################################################")
        
        return self



#####################################################################
# Get the new model
class Network_selector(nn.Module):
    def __init__(self, input_shape=151, k=2, truncated=40):
        super(Network_selector, self).__init__()
        self.l1  =nn.Linear(in_features=input_shape, out_features=truncated)
        self.l2  =nn.Linear(in_features=truncated, out_features=truncated)
        self.l3  =nn.Linear(in_features=truncated, out_features=truncated)

    ##########################################
    def forward(self, x):
        x = x.double()
        x = torch.nn.Tanh()(self.l1(x))
        identity = x
        x = torch.nn.Tanh()(self.l2(x)) + identity
        skip =x
        alpha = self.l3(x)+skip
        return alpha

class Network(nn.Module):
    ##########################################
    def __init__(self, kern, kern_R, truncation = 40, input_shape=151, device="cpu"):
        super(Network, self).__init__()
        self.selector = Network_selector()
        self.kern=torch.from_numpy(kern)
        _, s, vh  = torch.svd(self.kern)
        print("the shapex", s.shape, vh.shape)
        self.U=vh[:,0:truncation]
        print("the U shape", self.U.shape)


    ##########################################
    def checkpoint(self, filename, optimiser, factor_R, factor_E, fac_var, epoch, loss):
        torch.save({'step': epoch,
                    'loss': loss,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimiser.state_dict(),
                    'factor_R':factor_R,
                    'factor_E' : factor_E,
                    'fac_var':  fac_var,
                    }, filename)

    
    ##########################################
    def resume(self, filename, optimiser):
        checkpoint = torch.load(filename)
        optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        self.selector.load_state_dict(checkpoint['model_state_dict'])
        factor_R = checkpoint['factor_R'] 
        factor_E = checkpoint['factor_E']
        fac_var = checkpoint['fac_var']
        return self, optimiser, last_epoch, factor_E, factor_R, fac_var
    

    ##########################################
    def forward(self, x, sigma, device='cpu'):
        # print(device)
        x = x.float().to(device)
        self.U=self.U.double().to(device)
        select=self.selector(x).double().to(device)
        self.kern = self.kern.to(device)
        Rhat = torch.exp(torch.matmul(select, self.U.transpose(0, 1)) ).to(device)

        # Normalize E
        ####################################################################################
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1).to(device)
        correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000).to(device)
        Rhat = torch.div(Rhat, correction_term).to(device)
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1).to(device)
        
        # print("The value of sigma is", sigma)
        if sigma>0:
            
            # print("The value of sigma is", sigma)
            x_batch_N = (x+torch.normal(0, sigma, size=x.shape).to(device) )
            # select_3=self.selector(x_batch_N.pow(3)).double().to(device)
            # select_2=self.selector(x_batch_N.pow(2)).double().to(device)
            # select_1=self.selector(x_batch_N.pow(1)).double().to(device)
            select=self.selector(x_batch_N.pow(1)).double().to(device)
            Rhat_N = torch.exp(torch.matmul(select, self.U.transpose(0,1))  ).to(device)
            
            # Normalize E
            ####################################################################################
            Ehat_N = torch.matmul(self.kern, Rhat_N.transpose(0, 1)).transpose(0, 1).to(device)
            correction_term = Ehat_N[:, 0].view(-1, 1).repeat(1, 2000).to(device)
            Rhat_N = torch.div(Rhat_N, correction_term).to(device)
            multout_N = torch.matmul(self.kern, Rhat_N.transpose(0, 1)).to(device)
            Ehat_N = multout_N.transpose(0, 1).to(device)
            return Ehat, Rhat, Ehat_N, Rhat_N, x_batch_N
        
        else:
            return Ehat, Rhat, Ehat, Rhat, x



    ####################################################################################
    def forward__res__ent_(self, x, sigma, index__ = 0, n_curves = 1000, device="cpu"):
        x = x.float().to(device)
        self.U=self.U.double().to(device)
        select=self.selector(x).double().to(device)
        self.kern = self.kern.to(device)
        Rhat = torch.exp(torch.matmul(select, self.U.transpose(0, 1)) ).to(device)
        print(Rhat.shape)
        
        # Normalize E
        ####################################################################################
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1).to(device)
        correction_term = Ehat[:, 0].view(-1, 1).repeat(1, 2000).to(device)
        Rhat = torch.div(Rhat, correction_term).to(device)
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1).to(device)
        print(Ehat.shape, Rhat.shape)
        print("exiting this evaluation")
        return Ehat.detach().cpu().numpy().reshape([1,151]), Rhat.detach().cpu().numpy().reshape([1,2000])


    ####################################################################################
    def forward__res__(self, x, sigma, index__ = 0, n_curves = 1000):
        #--------------------------------------------------------------------------------------
        x = x.float().to(device).view(1,-1)
        print("the actual input",x.shape)
        self.U=self.U.double().to(device)   
        #--------------------------------------------------------------------------------------
        select=self.selector(x).double().to(device)
        Rhat = torch.exp(torch.matmul(select, self.U.transpose(0, 1)) )
        print("At the onset", select.shape, Rhat.shape, x.shape)
        #--------------------------------------------------------------------------------------
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1)


        print("out of the matmul", Ehat.shape, Rhat.shape)
        correction_term = Ehat[:,0].view(-1,1).repeat(1, 2000)
        Rhat = torch.div(Rhat, correction_term)
        Ehat = torch.matmul(self.kern, Rhat.transpose(0, 1)).transpose(0, 1)
        print(Ehat.shape, Rhat.shape)

        #--------------------------------------------------------------------------------------
        x_100 =torch.zeros(n_curves, x.shape[1]).to(device)
        for i in range(n_curves):
            x_100[i,:]= x+torch.normal(0, sigma, size=x.shape).to(device) 
        select = self.selector(x_100).double().to(device)
        print(select.shape)
        Rhat_N = torch.exp(torch.matmul(select, self.U.transpose(0,1))  )
        print(Rhat_N.shape, x_100.shape)
        

        # print("before Euc", self.kern.shape, Rhat_N.shape)
        #--------------------------------------------------------------------------------------
        Ehat_N = torch.matmul(self.kern, Rhat_N.transpose(0, 1)).transpose(0, 1)
        correction_term = Ehat_N[:, 0].view(-1, 1).repeat(1, 2000)
        Rhat_N = torch.div(Rhat_N, correction_term)
        multout_N = torch.matmul(self.kern, Rhat_N.transpose(0, 1))
        Ehat_N = multout_N.transpose(0, 1)

        print(Ehat_N.shape, x_100.shape, Rhat_N.shape)

        return Ehat.detach().cpu().numpy(), Rhat.detach().cpu().numpy(), Ehat_N.detach().cpu().numpy(),\
            Rhat_N.detach().cpu().numpy(), x_100.detach().cpu().numpy()
    


    ####################################################################################
    def loss_func(self, Ehat, ENhat, Rhat,  E, EN, R, fac, device='cpu'):
        if fac[2]>0:
            # The R loss
            non_integrated_entropy = (Rhat-R-torch.mul(Rhat, torch.log(torch.div(Rhat, R))))
            loss_R = -torch.mean(non_integrated_entropy)
            loss_E = torch.mean( torch.mul((Ehat- E).pow(2), (1/(0.0001*0.0001))))\
                   + torch.mean( torch.mul((EN- ENhat).pow(2), (1/(fac[2]*fac[2]))) ) 
            return (fac[0]*loss_R+fac[1]* loss_E), loss_E, loss_R
        else:
            # The R loss
            non_integrated_entropy = (Rhat-R-torch.mul(Rhat, torch.log(torch.div(Rhat, R))))
            loss_R = -1*torch.mean(non_integrated_entropy)
            loss_E =  torch.mean( torch.mul((Ehat- E).pow(2), (1/(1e-04*1e-04))))
            return (fac[0]*loss_R+fac[1]*loss_E), loss_E, loss_R

    def evaluate_metrics_models(self, data, device):
        testloader, fac_var = data 
        self.eval()
        ### Final Numbers 
        Ent_list =[]
        chi2=[]
        x_hat= []
        y_hat =[]
        for x_batch, y_batch in testloader:
            y_batch = y_batch.float().to(device)
            x_batch = x_batch.float().to(device)
            x_hat, y_hat, _, _, _ = self.forward(x_batch, fac_var, device=device)
            # Calculate Entropy
            my_list = -1*Entropy(y_batch.cpu().detach().numpy(), y_hat.cpu().detach().numpy(), 1e04 )
            [Ent_list.append(item) for item in my_list]
            my_list = chi2_vec(x_batch.cpu().detach().numpy(), x_hat.cpu().detach().numpy(), 1e-04)
            [chi2.append(item) for item in my_list]
        
        return sum(Ent_list)/len(Ent_list), sum(chi2)/len(chi2)
       


    def evaluate_plots(self, testloader, omega_fine, tau, epoch, save_dir, filee, fac):
        import matplotlib.pyplot as plt
        import numpy 
        self.eval()
        import matplotlib.pyplot as plt
        import seaborn as sns

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
        COLOR = 'dimgrey'
        rc={'axes.titlesize': small,
            'legend.fontsize': small,
            'axes.labelsize': med,
            'axes.titlesize': small,
            'xtick.labelsize': small,
            'ytick.labelsize': med,
            'figure.titlesize': small, 
            'text.color' : COLOR,
            'axes.labelcolor' : COLOR,
            'axes.axisbelow': False,
            'axes.edgecolor': 'lightgrey',
            'axes.facecolor': 'None',
            'axes.grid': False,
            'axes.labelcolor': 'dimgrey',
            'axes.spines.right': False,
            'axes.spines.bottom': False,
            'axes.spines.left': False,
            'axes.spines.top': False,
            'figure.facecolor': 'white',
            'lines.solid_capstyle': 'round',
            'patch.edgecolor': 'w',
            'patch.force_edgecolor': True,
            'text.color': 'dimgrey',
            'xtick.bottom': False,
            'xtick.color': 'dimgrey',
            'xtick.direction': 'out',
            'xtick.top': False,
            'ytick.color': 'dimgrey',
            'ytick.direction': 'out',
            'ytick.left': False,
            'ytick.right': False}
        plt.rcParams.update(rc)
        plt.rc('text', usetex = False)
        for j in range(5):
            fig, ax = plt.subplots( 4,2, figsize=(16,15) )
            for x_batch, y_batch in testloader:
                y_batch = y_batch.float().to(device)
                x_batch = x_batch.float().to(device)
                for i in range(4):
                    x_hat, y_hat, _, yvar, xvar = self.forward(x_batch, fac*pow(100,i))
                    ## PLOT THINGS ABOUT THE R
                    curve=y_batch[j,:].cpu().detach().numpy()
                    ax[i][0].plot((omega_fine).reshape([-1]), curve, '--', label='R('+str(fac*pow(100,i))+')', color='blue')    
                    Rhat = y_hat.cpu().detach().numpy()[j, :]
                    yerr = abs(Rhat-yvar.cpu().detach().numpy()[j, :])
                    fill_up = Rhat+yerr
                    fill_down = Rhat-yerr
                    fill_down[fill_down<0]= 0
                    ax[i][0].fill_between(omega_fine.reshape([-1]), fill_up, fill_down, alpha=1, color='orange')
                    ax[i][0].set_xlim([0,400])
                    ax[i][0].legend(loc='upper right')
                    ## PLOT THINGS ABOUT THE E
                    curve=x_batch[j,:].cpu().detach().numpy()
                    ax[i][1].plot( (tau).reshape([-1]), curve, label='E', color='blue')    
                    Ehat = x_hat.cpu().detach().numpy()[j, :]
                    yerr = abs(Ehat-xvar.cpu().detach().numpy()[j, :])
                    ax[i][1].errorbar(tau.reshape([-1]), Ehat, yerr=yerr, fmt='x', linewidth=2, ms = 1, label='Ehat', color='orange')
                    ax[i][1].set_yscale('log')
                    ax[i][1].legend(loc='upper right')
                    ax[i][0].set_xlabel('$ \\omega (MeV)$')
                    ax[i][1].set_xlabel('$ \\tau (MeV^{-1})$')
                    ax[i][0].set_ylabel('$ R(\\omega)(MeV^{-1})$')
                    ax[i][1].set_ylabel('$ E(\\tau)$')
                fig.suptitle('Reconstructions $R, E$ with increasing error $(\\sigma)$ in the input Euclidean', fontsize=16)
                fig.tight_layout()
                plt.savefig(save_dir+filee+str(epoch)+'_'+str(j)+".png", dpi=300)
                plt.close()
                break
        return



    
#     ####################################################################################
#     def fit_batch(self, data, checkpoint, device, configs):
#         (X_train, y_train, E_test, R_test), flag, model_name = data 
#         last_epoch, batch_size, optimiser = checkpoint
#         # obs = len(trainloader)*batch_size
#         print("obtaining", x_batch.shape, y_batch.shape)
#         print("starting at epoch", last_epoch)
#         epoch = last_epoch
#         factor_R= configs['factor_R']
#         factor_E =configs['factor_E']
#         fac_var  = configs['fac_var']
        
#         # Parameters for L-BFGS training
#         max_iter = 200                      # note each iteration is NOT an epoch
#         ghost_batch = 128
#         batch_size = 8192
#         overlap_ratio = 0.25                # should be in (0, 0.5)
#         lr = 1


#         #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.999)
#         self.train()
#         # optimiser = torch.optim.LBFGS(self.parameters(),
#         #             history_size=100, 
#         #             max_iter=500, 
#         #             line_search_fn="strong_wolfe")
        
#         # Define optimizer
#         optimiser = LBFGS(self.parameters(), lr=lr, history_size=10, line_search='None', debug=True)

#         # Main training loop
#         Ok_size = int(overlap_ratio * batch_size)
#         Nk_size = int((1 - 2 * overlap_ratio) * batch_size)

#         # sample previous overlap gradient
#         random_index = np.random.permutation(range(X_train.shape[0]))
#         Ok_prev = random_index[0:Ok_size]
#         g_Ok_prev, obj_Ok_prev = get_grad(optimiser, X_train[Ok_prev], y_train[Ok_prev], opfun)

# # main loop
# for n_iter in range(max_iter):
    
#     # training mode
#     model.train()
    
#     # sample current non-overlap and next overlap gradient
#     random_index = np.random.permutation(range(X_train.shape[0]))
#     Ok = random_index[0:Ok_size]
#     Nk = random_index[Ok_size:(Ok_size + Nk_size)]
    
#     # compute overlap gradient and objective
#     g_Ok, obj_Ok = get_grad(optimizer, X_train[Ok], y_train[Ok], opfun)
    
#     # compute non-overlap gradient and objective
#     g_Nk, obj_Nk = get_grad(optimizer, X_train[Nk], y_train[Nk], opfun)
    
#     # compute accumulated gradient over sample
#     g_Sk = overlap_ratio * (g_Ok_prev + g_Ok) + (1 - 2 * overlap_ratio) * g_Nk
        
#     # two-loop recursion to compute search direction
#     p = optimizer.two_loop_recursion(-g_Sk)
                
#     # perform line search step
#     lr = optimizer.step(p, g_Ok, g_Sk=g_Sk)
    
#     # compute previous overlap gradient for next sample
#     Ok_prev = Ok
#     g_Ok_prev, obj_Ok_prev = get_grad(optimizer, X_train[Ok_prev], y_train[Ok_prev], opfun)
    
#     # curvature update
#     optimizer.curvature_update(g_Ok_prev, eps=0.2, damping=True)
    
    
        
#         while epoch < configs['epochs']:
#             epoch += 1
#             x_batch = x_batch.double().to(device)
#             y_batch = y_batch.double().to(device)
#             E_test = E_test.double().to(device)
#             R_test = R_test.double().to(device)
#             def closure():
#                 optimiser.zero_grad()
#                 xhat, yhat, xNhat, _, x_batch_N= self.forward(x_batch, fac_var,\
#                                                         device = device)
#                 loss, _, _ = self.loss_func( xhat, xNhat, yhat, x_batch,\
#                                             x_batch_N, y_batch, \
#                                                 [factor_R, factor_E, fac_var], device = device)
#                 loss.backward()
                
#                 return loss
            
#             optimiser.step(closure)
#             ####################################################################################
#             xhat, yhat, xNhat, _, x_batch_N= self.forward(x_batch, fac_var,\
#                                                         device = device)
#             loss, E_L, R_L = self.loss_func( xhat, xNhat, yhat, x_batch,\
#                                             x_batch_N, y_batch, \
#                                                 [factor_R, factor_E, fac_var], device = device)
            
            
#             # test_loss calculation
            
#             xhat, yhat, xNhat, _, x_batch_N= self.forward(E_test, fac_var,\
#                                                         device = device)
#             lossT, E_LT, R_LT = self.loss_func( xhat, xNhat, yhat, E_test,\
#                                             x_batch_N, R_test, \
#                                                 [factor_R, factor_E, fac_var], device = device)
            
            
#             # loss.backward()
#             # optimiser.step()
            
#             # current_loss      += (1/batches) * (loss.cpu().item() - current_loss)
#             # current_loss_E    += (1/batches) * (E_L.cpu().item() - current_loss_E)
#             # current_loss_R    += (1/batches) * (R_L.cpu().item() - current_loss_R) 
                
                
#             ####################################################################################
#             # if epoch % 10 ==0:
#                 # scheduler.step()
#             sys.stdout.write('\rProgress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f, Losstest: %f, Loss_Rtest: %f, Loss_Etest: %f' %(\
#             epoch, configs['epochs'], loss, R_L*1e10, E_L, lossT, R_LT*1e10, E_LT) )
#             # print("0")
#             sys.stdout.flush()
#             # ####################################################################################
#             # ## saving code 
#             # # if epoch % 50==0:
#             if flag==0:
#                 self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
#             elif flag==1:
#                 # print(loss.cpu().item())
#                 self.checkpoint(model_name+"second", optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
#             else:
#                 self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
        
#         return self

    def get_grad(self, optimizer, X_Sk, y_Sk, opfun, ghost_batch=128, device=device, fac= [0,0,0]):
        optimizer.zero_grad()
        xhat, yhat, xNhat, _, x_batch_N= opfun(X_Sk)
        loss_fn, _, _ = self.loss_func( xhat, xNhat, yhat, X_Sk, x_batch_N, y_Sk, fac, device = device)
        loss_fn.backward()
        # gather flat gradient
        grad = optimizer._gather_flat_grad()
        return grad, loss_fn


####################################################################################
    def fit(self, data, checkpoint, device, configs):
        trainloader, flag, model_name = data 
        last_epoch, batch_size, optimiser = checkpoint
        obs = len(trainloader)*batch_size
        print("starting at epoch", last_epoch)
        epoch = last_epoch
        factor_R= configs['factor_R']
        factor_E =configs['factor_E']
        fac_var  = configs['fac_var']
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.999)
        self.train()
        while epoch < configs['epochs']:
            epoch += 1
            current_loss=0
            current_loss_E = 0
            current_loss_R = 0
            batches = 0
            progress = 0
            ####################################################################################
            for x_batch, y_batch in trainloader:
                batches += 1
                x_batch = x_batch.double().to(device)
                y_batch = y_batch.double().to(device)
                
                optimiser.zero_grad()
                ####################################################################################
                xhat, yhat, xNhat, _, x_batch_N = self.forward(x_batch, fac_var,\
                                                            device = device)
                loss, E_L, R_L = self.loss_func( xhat, xNhat, yhat, x_batch,\
                                                x_batch_N, y_batch, \
                                                 [factor_R, factor_E, fac_var], device = device)
                loss.backward()
                optimiser.step()
                current_loss      += (1/batches) * (loss.cpu().item() - current_loss)
                current_loss_E    += (1/batches) * (E_L.cpu().item() - current_loss_E)
                current_loss_R    += (1/batches) * (R_L.cpu().item() - current_loss_R) 
                
            ####################################################################################
            if epoch % 10 ==0:
                # scheduler.step()
                sys.stdout.write('\rProgress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f' %(\
                epoch, configs['epochs'], current_loss, current_loss_R*1e10, current_loss_E) )
                sys.stdout.flush()
            # ####################################################################################
            # ## saving code 
            # # if epoch % 50==0:
            if flag==0:
                self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
            elif flag==1:
                # print(loss.cpu().item())
                self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
            else:
                self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
        
        return self

    # ####################################################################################
    # def fit(self, data, checkpoint, device, configs):
    #     (x_batch, y_batch, E_test, R_test), flag, model_name = data 
    #     last_epoch, batch_size, optimiser = checkpoint
    #     # obs = len(trainloader)*batch_size
    #     print("obtaining", x_batch.shape, y_batch.shape)
    #     print("starting at epoch", last_epoch)
    #     epoch = last_epoch
    #     factor_R= configs['factor_R']
    #     factor_E =configs['factor_E']
    #     fac_var  = configs['fac_var']
    #     #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.999)
    #     self.train()
    #     optimiser = torch.optim.LBFGS(self.parameters(),
    #                 history_size=100, 
    #                 max_iter=500, 
    #                 line_search_fn="strong_wolfe")
        
    #     # optimiser = SdLBFGS0(self.parameters(), lr=1e-04)
    #     no_samples = x_batch.shape[0]
    #     ghost_batch= 8192
    #     x_batch = x_batch.double().to(device)
    #     y_batch = y_batch.double().to(device)
    #     E_test = E_test.double().to(device)
    #     R_test = R_test.double().to(device)
            
    #     while epoch < configs['epochs']:
    #         epoch += 1
            
    #         for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples / ghost_batch), 1)):
    #             batch_x = x_batch[subsmpl]
    #             batch_y = y_batch[subsmpl]
                
    #             # optimiser.zero_grad()
    #             # xhat, yhat, xNhat, _, x_batch_N= self.forward(batch_x, fac_var,\
    #             #                                         device = device)
    #             # loss, _, _ = self.loss_func( xhat, xNhat, yhat, batch_x,\
    #             #                             x_batch_N, batch_y, \
    #             #                                 [factor_R, factor_E, fac_var], device = device)
    #             # loss.backward()
                    
                    
                    
    #             def closure():
    #                 optimiser.zero_grad()
    #                 xhat, yhat, xNhat, _, x_batch_N= self.forward(batch_x, fac_var,\
    #                                                         device = device)
    #                 loss, _, _ = self.loss_func( xhat, xNhat, yhat, batch_x,\
    #                                             x_batch_N, batch_y, \
    #                                                 [factor_R, factor_E, fac_var], device = device)
    #                 loss.backward()
    #                 return loss
                
    #             optimiser.step(closure)
            
            
            
            
    #         ####################################################################################
    #         xhat, yhat, xNhat, _, x_batch_N= self.forward(x_batch, fac_var,\
    #                                                     device = device)
    #         loss, E_L, R_L = self.loss_func( xhat, xNhat, yhat, x_batch,\
    #                                         x_batch_N, y_batch, \
    #                                             [factor_R, factor_E, fac_var], device = device)
            
            
    #         # test_loss calculation
            
    #         xhat, yhat, xNhat, _, x_batch_N= self.forward(E_test, fac_var,\
    #                                                     device = device)
    #         lossT, E_LT, R_LT = self.loss_func( xhat, xNhat, yhat, E_test,\
    #                                         x_batch_N, R_test, \
    #                                             [factor_R, factor_E, fac_var], device = device)
            
            
    #         # loss.backward()
    #         # optimiser.step()
            
    #         # current_loss      += (1/batches) * (loss.cpu().item() - current_loss)
    #         # current_loss_E    += (1/batches) * (E_L.cpu().item() - current_loss_E)
    #         # current_loss_R    += (1/batches) * (R_L.cpu().item() - current_loss_R) 
                
                
    #         ####################################################################################
    #         # if epoch % 10 ==0:
    #             # scheduler.step()
    #         sys.stdout.write('\rProgress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f, Losstest: %f, Loss_Rtest: %f, Loss_Etest: %f' %(\
    #         epoch, configs['epochs'], loss, R_L*1e10, E_L, lossT, R_LT*1e10, E_LT) )
    #         print("0")
    #         sys.stdout.flush()
    #         # ####################################################################################
    #         # ## saving code 
    #         # # if epoch % 50==0:
    #         if flag==0:
    #             self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
    #         elif flag==1:
    #             # print(loss.cpu().item())
    #             self.checkpoint(model_name+"second", optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
    #         else:
    #             self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
        
    #     return self
    
    
    # ####################################################################################
    # def fit(self, data, checkpoint, device, configs):
    #     (x_batch, y_batch, E_test, R_test), flag, model_name = data 
    #     last_epoch, batch_size, optimiser = checkpoint
    #     # obs = len(trainloader)*batch_size
    #     print("obtaining", x_batch.shape, y_batch.shape)
    #     print("starting at epoch", last_epoch)
    #     epoch = last_epoch
    #     factor_R= configs['factor_R']
    #     factor_E =configs['factor_E']
    #     fac_var  = configs['fac_var']
    #     #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.999)
    #     self.train()
    #     optimiser = torch.optim.LBFGS(self.parameters(),
    #                 history_size=100, 
    #                 max_iter=500, 
    #                 line_search_fn="strong_wolfe")
        

    #     no_samples = x_batch.shape[0]
    #     ghost_batch=1208
    #     while epoch < configs['epochs']:
    #         epoch += 1
    #         x_batch = x_batch.double().to(device)
    #         y_batch = y_batch.double().to(device)
    #         E_test = E_test.double().to(device)
    #         R_test = R_test.double().to(device)
    #         def closure():
    #             optimiser.zero_grad()
    #             loss=0.0
    #             for subsmpl in np.array_split(np.arange(no_samples), max(int(no_samples / ghost_batch), 1)):
    #                 batch_x = x_batch[subsmpl]
    #                 batch_y = y_batch[subsmpl]

    #                 xhat, yhat, xNhat, _, x_batch_N= self.forward(batch_x, fac_var,\
    #                                                         device = device)
    #                 loss_num, _, _ = self.loss_func( xhat, xNhat, yhat, batch_x,\
    #                                             x_batch_N, batch_y, \
    #                                                 [factor_R, factor_E, fac_var], device = device)
                    
    #                 loss +=loss_num*(len(subsmpl)/no_samples)
                    
                
    #             loss.backward()
    #             return loss
            
    #         optimiser.step(closure)
    #         ####################################################################################
    #         xhat, yhat, xNhat, _, x_batch_N= self.forward(x_batch, fac_var,\
    #                                                     device = device)
    #         loss, E_L, R_L = self.loss_func( xhat, xNhat, yhat, x_batch,\
    #                                         x_batch_N, y_batch, \
    #                                             [factor_R, factor_E, fac_var], device = device)
            
            
    #         # test_loss calculation
            
    #         xhat, yhat, xNhat, _, x_batch_N= self.forward(E_test, fac_var,\
    #                                                     device = device)
    #         lossT, E_LT, R_LT = self.loss_func( xhat, xNhat, yhat, E_test,\
    #                                         x_batch_N, R_test, \
    #                                             [factor_R, factor_E, fac_var], device = device)
            
            
    #         # loss.backward()
    #         # optimiser.step()
            
    #         # current_loss      += (1/batches) * (loss.cpu().item() - current_loss)
    #         # current_loss_E    += (1/batches) * (E_L.cpu().item() - current_loss_E)
    #         # current_loss_R    += (1/batches) * (R_L.cpu().item() - current_loss_R) 
                
                
    #         ####################################################################################
    #         # if epoch % 10 ==0:
    #             # scheduler.step()
    #         sys.stdout.write('\rProgress: %d/%d, Loss: %f, Loss_R: %f, Loss_E: %f, Losstest: %f, Loss_Rtest: %f, Loss_Etest: %f' %(\
    #         epoch, configs['epochs'], loss, R_L*1e10, E_L, lossT, R_LT*1e10, E_LT) )
    #         # print("0")
    #         sys.stdout.flush()
    #         # ####################################################################################
    #         # ## saving code 
    #         # # if epoch % 50==0:
    #         if flag==0:
    #             self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
    #         elif flag==1:
    #             # print(loss.cpu().item())
    #             self.checkpoint(model_name+"second", optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
    #         else:
    #             self.checkpoint(model_name, optimiser, factor_R, factor_E, fac_var, epoch, loss.cpu().item())
        
    #     return self


    ####################################################################################
    def plot(self, data, configs):
        testloader, flag, save_dir, omega, tau, epoch = data 
        print("plotting at epoch", epoch)
        fac_var  = configs['fac_var']
        if flag==0:
            print("########################################################")
            print("\n One Peak")
            Ent_list, chi2 =self.evaluate_METRICS(testloader_one, fac_var, save_dir, epoch, filee='one_peak')
            self.evaluate_plots(testloader_one, omega, tau, epoch,\
                 save_dir, filee='one_peak', fac=fac_var)
        elif flag==1:
            print("########################################################")
            print("\n Two Peak")
            Ent_list, chi2 =self.evaluate_METRICS(testloader_two, fac_var, save_dir, epoch, filee='two_peak')
            self.evaluate_plots(testloader_two, omega, tau, epoch, save_dir, filee='two_peak', fac=fac_var)
            print("########################################################")
        else:
            testloader_one, testloader_two = testloader
            print("########################################################")
            print("\n One Peak")
            Ent_one, chi2_one=self.evaluate_METRICS(testloader_one, fac_var, save_dir, epoch, filee='one_peak')
            self.evaluate_plots(testloader_one, omega, tau, epoch, save_dir, filee='one_peak', fac=fac_var)
            print("########################################################")
            print("Two Peak")
            Ent_two, chi2_two=self.evaluate_METRICS(testloader_two, fac_var, save_dir, epoch, filee='two_peak')
            self.evaluate_plots(testloader_two, omega, tau, epoch, save_dir, filee='two_peak', fac=fac_var)    
            print("########################################################")    
            Ent_list = Ent_one.extend(Ent_two)
            chi2 = chi2_one.extend(chi2_two)

        return Ent_list, chi2