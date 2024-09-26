from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.ETauDataset import ETauDataset
import numpy as np
from Optimisation.helpers_opt import loss

def run_optimization(dataset_rate, dataset_eff, rate_budget):
    config = load_cfg_file()
    PNetparam_t3 = float(config['OPT']['PNet_t3'])

    def f(a):
        a = np.append(a, [PNetparam_t3])

        N_den, N_num = dataset_rate.get_Nnum_Nden_ETauPNet(a)
        rate = (N_num/N_den)*L1A_physics

        N_den, N_num = dataset_eff.ComputeEffAlgo_ETauPNet(a)
        eff_algo = (N_num/N_den)

        print(f'Rate: {rate}')
        print(f'Algo Eff: {eff_algo}') 
        print(f'Output score: {- eff_algo + loss(rate, rate_budget)}')
        print('---------------------------------------------------------------------------------------------------------')
        return - eff_algo + loss(rate, rate_budget)

    res = minimize(f, [0.95, 0.95], bounds=((0.8, 0.99), (0.8, 0.99)), method="L-BFGS-B", options={"eps": [0.01, 0.01]})
    return res.x

if __name__ == '__main__':

    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    HLT_name = config['HLT']['HLTname']
    L1A_physics = float(config['RUNINFO']['L1A_physics'])
    Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)
    PNetparam_t3 = float(config['OPT']['PNet_t3']) # optimise only t1 and t2

    FileNameList_eff = f"/afs/cern.ch/work/s/skeshri/TauHLT/Braden/TauHLTOptimzation/PnetAtHLT/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" # optimisation with ZprimeToTauTau_M-4000 data
    FileNameList_rate = files_from_path(Rate_path)[0] # only one otherwise it's too long (gives good aprosimation for the rate)

    dataset_eff = ETauDataset(FileNameList_eff)
    dataset_rate = ETauDataset(FileNameList_rate)

    optim_param = run_optimization(dataset_rate, dataset_eff, rate_budget = 15)
    optim_param = np.append(optim_param, [PNetparam_t3])

    print('Optimisation finished:')
    print("Optimized parameters:", optim_param)

    N_den, N_num = dataset_rate.get_Nnum_Nden_ETauPNet(optim_param)
    rate_opt = (N_num/N_den)*L1A_physics

    print("Approx. Rate:", rate_opt)

    N_den, N_num = dataset_eff.ComputeEffAlgo_ETauPNet(optim_param)
    algo_eff = (N_num/N_den)

    print("Approx. Algo Eff:", algo_eff)

