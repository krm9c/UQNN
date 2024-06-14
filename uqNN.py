import torch
from Lib import *
import argparse
import json
import os
import signal
import sys
import itertools

#-------------------------------------------
## Gather the reconstructions from my model
# ------------------------------------------
from matplotlib.lines import Line2D
import matplotlib.font_manager as font_manager
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import sys
import torch
sys.path.append("/home/kraghavan/Projects/Nuclear/Inverse/UQNuc/torch_attention_model_v12/")


import torch.multiprocessing as mp
torch.set_default_dtype(torch.float64)
class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """
    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by
          `params.dict['learning_rate']"""
        return self.__dict__



# ------------------------------------------------------------------------
# ------------------------------------------------------------------------
def load_checkpoint(path, learning_rate, data, device="cpu"):
    step = 0
    loss = 0
    Kern, Kern_R = data
    rbfnet = Network(Kern, Kern_R, device=device).to(torch.float64)                 
    rbfnet.to(device)
    # optimiser = optim.SWATS(rbfnet.parameters(), lr=learning_rate, nesterov=True)
    optimiser = optim.QHAdam(rbfnet.parameters(), lr=learning_rate)
    if os.path.exists(path):
        print("load from checkpoint")
        checkpoint = torch.load(path, map_location=device)
        loss = checkpoint["loss"]
        step= checkpoint["step"]
        # optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Starting step is", step)
        # for g in optimiser.param_groups:
        #     print("The group lr",  g['lr'])
        #     g['lr'] = learning_rate
        rbfnet.load_state_dict(checkpoint["model_state_dict"] )
    return step, rbfnet, optimiser, loss




# ------------------------------------------------------------------------
def return_data(params):
    n_data=params['n_points']
    batche=params['batche']
    ## Theta
    # x = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/inverse_data_interpolated_numpy.p')
    ## JLSE
    x = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/inverse_data_interpolated_numpy.p')

    ## ThetaGPU
    # P = return_dict('/grand/NuQMC/UncertainityQ/theta_JLSE_Port/Test_MEM_data.p')
    ## JLSE
    P = return_dict('/gpfs/jlse-fs0/users/kraghavan/Inverse/Test_MEM_data.p')

    remove_out_1 = [461, 347, 704]
    remove_out_2 = [476, 271, 484]
    

    if flag == 0:
        print("one peak")
        tau = x['tau']
        omega_fine=x['omega_fine']
        omega=x['omega']
        R = np.concatenate([x['One_Peak_R_interp'][0:n_data]], axis=0)
        E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
        Kern = Kern.astype('float64')
        print(torch.cuda.is_available())
        torch.pi = torch.acos(torch.zeros(1)).item() 
        x = torch.from_numpy(E)
        y = torch.from_numpy(R)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batche, shuffle=True)

        R_test_1 = np.concatenate([  P['One_Peak_R']], axis=0)
        R_test_1 = np.delete(R_test_1, remove_out_1, 0)
        E_test_1 = np.concatenate([  P['One_Peak_E']], axis=0)
        E_test_1 = np.delete(E_test_1, remove_out_1, 0)
        E_test_1 = torch.from_numpy(E_test_1)
        R_test_1 = torch.from_numpy(R_test_1)
        testset_one = MyDataset(E_test_1, R_test_1)
        testloader= DataLoader(testset_one, batch_size=batche, shuffle=False)
        return trainloader, testloader, Kern, Kern_R,tau, omega_fine, E_test_1, R_test_1
    elif flag==1:
        print("two peaks")
        tau = x['tau']
        omega_fine=x['omega_fine']
        omega=x['omega']
        R = np.concatenate([ x['Two_Peak_R_interp'][0:n_data] ], axis=0)
        E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
        Kern = Kern.astype('float64')
        print(torch.cuda.is_available())
        torch.pi = torch.acos(torch.zeros(1)).item() 
        x = torch.from_numpy(E)
        y = torch.from_numpy(R)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batche, shuffle=True)

        R_test_2 = np.concatenate([  P['Two_Peak_R']], axis=0)
        R_test_2 = np.delete(R_test_2, remove_out_2, 0)
        E_test_2 = np.concatenate([  P['Two_Peak_E']], axis=0)
        E_test_2 = np.delete(E_test_2, remove_out_2, 0)
        E_test_2 = torch.from_numpy(E_test_2)
        R_test_2 = torch.from_numpy(R_test_2)
        testset = MyDataset(E_test_2, R_test_2)
        testloader = DataLoader(testset, batch_size=batche, shuffle=False)
        print("returning", E_test_2.shape, R_test_2.shape)
        return trainloader, testloader, Kern, Kern_R,tau, omega_fine, E_test_2, R_test_2

            
    elif flag==2:
        print("both peaks")
        tau = x['tau']
        omega_fine=x['omega_fine']
        omega=x['omega']
        R = np.concatenate([x['One_Peak_R_interp'][0:n_data], x['Two_Peak_R_interp'][0:n_data] ], axis=0)
        E, R, Kern, Kern_R = integrate(tau, omega, omega_fine, R)
        Kern = Kern.astype('float64')
        torch.pi = torch.acos(torch.zeros(1)).item() 
        x = torch.from_numpy(E)
        y = torch.from_numpy(R)
        trainset = MyDataset(x, y)
        trainloader = DataLoader(trainset, batch_size=batche, shuffle=True)

        R_test_1= np.concatenate([  P['One_Peak_R']], axis=0)
        R_test_1 = np.delete(R_test_1, remove_out_1, 0)
        R_test_2 = np.concatenate([  P['Two_Peak_R']], axis=0)
        R_test_2 = np.delete(R_test_2, remove_out_2, 0)
        E_test_1 = np.concatenate([  P['One_Peak_E']], axis=0)
        E_test_1 = np.delete(E_test_1, remove_out_1, 0)
        E_test_2 = np.concatenate([  P['Two_Peak_E']], axis=0)
        E_test_2 = np.delete(E_test_2, remove_out_2, 0)
        E_test = np.concatenate([E_test_1, E_test_2], axis =0)
        R_test = np.concatenate([R_test_1, R_test_2], axis =0)
        E_test = torch.from_numpy(E_test)
        R_test = torch.from_numpy(R_test)
        testset = MyDataset(E_test, R_test)
        testloader= DataLoader(testset, batch_size=batche, shuffle=False)
        
        return trainloader,testloader, Kern, Kern_R, tau, omega_fine,\
            np.concatenate([E_test_1, E_test_2], axis=0),\
            np.concatenate([R_test_1, R_test_2], axis=0) 

# -------------------------------------------------------------------------------------------------------------  
def train_model(stuff):
    i, model, device, path, data, params = stuff
    (trainloader, tau, omega) = data
    (prev_epoch, rbfnet, optimiser, prev_loss) = model
    batche=params['batche']

    def sigterm_handler(*args):
        unlock_model(path)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    if is_locked(path):
        return
    print(
        f"Launch training of model {i} on process {mp.current_process().pid}",
        flush=True,
    )
    # Create a lock for this model so only this process/job can work on this model
    lock_model(path) 
    # data = (trainloader, testloader, testloader, omega,\
    #           tau, flag, save_dir, model_path) 
    # checkpoint = (prev_epoch, batche, optimiser)
    print(rbfnet, path)
    rbfnet.fit( (trainloader, params['flag'], path), 
    (prev_epoch, batche, optimiser), device=device, configs=params)
    unlock_model(path)


def plot_model(stuffs, x,y, omega_fine, tau, flag=0, sigma=1e-04, N=50):
    print("plotting")
    #print(x.shape, y.shape, omega_fine.shape, tau.shape, sigma)
    omega_fine=omega_fine.reshape([-1])
    tau=tau.reshape([-1])
    scale = x[0]
    ehat    = np.zeros([N,151])
    rhat    = np.zeros([N,2000])
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    for stuff  in stuffs:
        i,  device, model = stuff
        (_, rbfnet, _, _) = model
        ehat[i,:], rhat[i,:]= rbfnet.forward__res__ent_(x, sigma, index__ = 0, n_curves = 1, device = device)
        #print("i", i)
    # print("I finished this loop")
    #-----------------------------------------------------------------
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
    colors_ = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', '#9467bd']
    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, rhat.shape[0])]
    plt.rcParams['mathtext.fontset'] = 'cm'

    fig,a =  plt.subplots(2, 1, dpi = 1200, gridspec_kw = {'wspace':0.40, 'hspace':0.40})
    a= a.flatten()
    for k in range(rhat.shape[0]):
        a[0].plot(omega_fine, rhat[k,:], color = colors[k], alpha=1,  lw = lw)
    
    a[0].plot(omega_fine, y.reshape([-1]), color =colors_[0],   label = 'Original', linestyle = '-',linewidth = 20*lw)
    a[0].set_xlim([0,400])
    a[0].set_xlabel('$\omega~[\mathrm{MeV}]$')
    a[0].set_ylabel('$R(\omega)~[\mathrm{MeV}^{-1}]$')
    a[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
    a[0].grid(linestyle=':', linewidth=0.5)


    if not isinstance(scale, np.ndarray):
        scale = scale.numpy()

    for k in range(rhat.shape[0]):
        a[1].plot(tau, ehat[k,:]*scale, color = colors[k], alpha=1, linestyle = '-.', lw = lw)
    
    a[1].plot(tau, x.reshape([-1])*scale, color =colors[1],   label = 'Original', linestyle = '-',linewidth = 20*lw)
    a[1].set_xlim([0,0.0750])
    a[1].set_xlabel('$\\tau~[\mathrm{MeV}^{-1}]$')
    a[1].set_ylabel('$E(\\tau)$')
    a[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.03f'))
    a[1].grid(linestyle=':', linewidth=0.5)
    a[0].set_title('$\\sigma$='+str(sigma))

    
    # a[1].legend(loc = 'upper right',ncol=1 )
    plt.savefig(str(flag)+'plot_model__.png', bbox_inches='tight', dpi=120)
    plt.close()

    return ehat, rhat



# -------------------------------------------------------------------------------------------------------------  
def get_metrics(stuffs, testloader, flag=0, sigma=1e-04,  N=50):
    chi2    = np.zeros(len(N))
    Ent_list   = np.zeros(len(N))
    for stuff  in stuffs:
        i, device, model = stuff
        print(i)
        (_, rbfnet, _, _) = model
        Ent_list[i-1], chi2[i-1]= \
        rbfnet.evaluate_metrics_models((testloader, sigma), device)


     #-----------------------------------------------------------------
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

    fig,a =  plt.subplots(1, 2, dpi = 1200, gridspec_kw = {'wspace':0.40, 'hspace':0.40})
    a= a.flatten()
    a[0].boxplot(Ent_list)
    a[0].set_yscale('log')
    a[0].set_title('Entropy')
    a[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.06f'))
    a[0].grid(linestyle=':', linewidth=0.5)

    a[1].boxplot(chi2)
    
    a[1].set_title('$\\chi^2$')
    # a[1].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.06f'))
    a[1].grid(linestyle=':', linewidth=0.5)
    a[1].legend(loc = 'upper right',ncol=1 )
    plt.savefig( str(flag)+'plot_model__metrics.png', bbox_inches='tight', dpi=120)
    plt.close()
    return Ent_list, chi2




########################################################################
# We want to be able to launch multiple jobs to train subsets of the
# ensemble of models concurrently. As such, we use a filesystem based
# semaphore to ensure a model is not being worked on by more than one job.
def lock_model(model_path):
    lock_file = open(model_path + ".lock", "w")
    lock_file.write(f"{mp.current_process().pid}")
    lock_file.close()


def unlock_model(model_path):
    try:
        os.remove(model_path + ".lock")
    except:
        print("There was no need to unlock")

def is_locked(model_path):
    return os.path.isfile(model_path + ".lock")


########################################################################
if __name__ == '__main__':
    # Load the parameters from json file
    parser = argparse.ArgumentParser(prog="uq.py", description="Tests the uq model for the inversion" )
    parser.add_argument('-t', '--train', default=False, action='store_true', help='do training')
    parser.add_argument('-p', '--plot', default=False, action='store_true', help='generate plots with test data')
    parser.add_argument('-path', '--path',default='RUN_1',help="Path to the json file")
    parser.add_argument('-flag','--flag',default='false',help="one peak, two peak, uncertainty, original")
    parser.add_argument('-method', '--method',default='UQ', help="one peak, two peak, uncertainty, original")
    parser.add_argument("-cuda","--cuda",default=False,action="store_true",help="consider using CUDA if it is available",)
    parser.add_argument("-np","--num-processes", default=None,help="number of processes to use (default is to auto detect)",)

    args = parser.parse_args()
    json_path = os.path.join(str(args.path)+'/parameters_json/paramsUQ.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict
    if args.path is not None:
        params['path'] = str(args.path)

    if args.flag is not None:
        params['flag'] = int(args.flag)

    if args.method is not None:
        params['method'] = str(args.method)

    if params["flag"] == 3:
        params['fac_var']=-1

    if args.cuda is not None:
        use_cuda = args.cuda
        if use_cuda:
            use_cuda = torch.cuda.is_available()

        if use_cuda:
            devices = [
                torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())
            ]
            num_devices = torch.cuda.device_count()
            num_processes = num_devices
            if args.num_processes is not None:
                num_processes = int(args.num_processes)
        else:
            num_processes = 4
            if args.num_processes is not None:
                num_processes = int(args.num_processes)
            devices = [torch.device("cpu")] * num_processes
            num_devices = len(devices)
    print(
        f"use_cuda = {use_cuda}, num_devices = {num_devices}, num_processes = {num_processes}",
        flush=True,
    )
    print(parser, params)

    models = []
    paths__ = []
    # n_models__ = [0]
    n_models__ = list(np.arange(10))
    global flag 
    global method

    flag = params['flag']
    method = params['method']
    batche=params['batche']
    epochs=params['epochs']
    trainloader, testloader, Kern, Kern_R, tau, omega, E__test, R__test = return_data(params=params)    
    mp.set_start_method("spawn")
    if args.train:
        device_list=[]
        for i in n_models__:
            model_path = f"../Results/models/{flag}_UQTrained_model_{str(i)}"
            
            checkpoint = load_checkpoint(model_path, params['learning_rate'], (Kern, Kern_R), device=devices[i % num_devices])
            device_list.append(devices[i % num_devices])
            models.append(checkpoint)
            paths__.append(model_path) 

        stuff = zip(range(len(models)), models, device_list, paths__,\
                itertools.repeat((trainloader, tau, omega)), itertools.repeat(params) )
    
        with mp.Pool(num_processes) as pool:
            pool.map(train_model, stuff)

    print("We are starting to plot")
# ------------------------------------------------------------------------------------------------------------------
    
    
# -------------------------------------------------------------------------------------------------------------  
    if args.plot:
        device_list=[]
        for i in n_models__:
            model_path = f"../Results/models/{flag}_UQTrained_model_{str(i)}"
            paths__.append(model_path) 
            checkpoint = load_checkpoint(model_path, params['learning_rate'], (Kern, Kern_R), device=devices[i % num_devices])
            device_list.append(devices[i % num_devices])
            models.append(checkpoint)
        
        
        ehat, rhat  = plot_model(zip(range(len(models)),  device_list,  models),\
                                E__test[100,:].reshape([1,-1]), R__test[100,:].reshape([1,-1]),\
                                omega, tau,flag=params['flag'], sigma= 1e-04  )
        Ent_list, chi2 = get_metrics( zip(range(len(models)),  device_list, models), testloader, flag=params['flag'], N =n_models__)
        print(ehat.shape, rhat.shape)
        print(Ent_list, chi2)
