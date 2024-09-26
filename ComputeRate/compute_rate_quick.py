from helpers import files_from_path, load_cfg_file, compute_ratio_witherr
import os

#compute rate with only 1 Ephemeral data file to give a quick aproximation of the rate (no need to run ProduceRateFiles task for that)

config = load_cfg_file()
RefRun = int(config['RUNINFO']['ref_run'])
HLT_name = config['HLT']['HLTname']
L1A_physics = float(config['RUNINFO']['L1A_physics'])
Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)

PNet_mode = config['OPT']['PNet_mode']
if PNet_mode == 'false':
    PNetMode = False
else:
    PNetMode = True
    PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]

FileNameList_rate = files_from_path(Rate_path)[0] # only one otherwise it's too long

if HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
    from HLTClass.DiTauDataset import DiTauDataset

    dataset_rate = DiTauDataset(FileNameList_rate)

    if PNetMode:
        N_den, N_num = dataset_rate.get_Nnum_Nden_DiTauPNet(PNetparam)
        print(f'Quick rate computation for DiTau path with PNet param {PNetparam}:')
    else:
        N_den, N_num = dataset_rate.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()
        print(f'Quick rate computation for {HLT_name}:')

if HLT_name == 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3':
    from HLTClass.SingleTauDataset import SingleTauDataset

    dataset_rate = SingleTauDataset(FileNameList_rate)

    if PNetMode:
        N_den, N_num = dataset_rate.get_Nnum_Nden_SingleTauPNet(PNetparam)
        print(f'Quick rate computation for SingleTau path with PNet param {PNetparam}:')
    else:
        N_den, N_num = dataset_rate.get_Nnum_Nden_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3()
        print(f'Quick rate computation for {HLT_name}:')

if HLT_name == 'HLT_DoubleTauOrSingleTau':
    from HLTClass.DoubleORSingleTauDataset import DoubleORSingleTauDataset

    dataset_rate = DoubleORSingleTauDataset(FileNameList_rate)

    if PNetMode:
        N_den, N_num = dataset_rate.get_Nnum_Nden_DoubleORSinglePNet(PNetparam)
    else:
        N_den, N_num = dataset_rate.get_Nnum_Nden_HLT_DoubleORSingleDeepTau()

if HLT_name == 'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1':
    from HLTClass.ETauDataset import ETauDataset

    dataset_rate = ETauDataset(FileNameList_rate)

    if PNetMode:
        N_den, N_num = dataset_rate.get_Nnum_Nden_ETauPNet(PNetparam)
    else:
        N_den, N_num = dataset_rate.get_Nnum_Nden_ETauDeepNet()


rate, rate_low, rate_up = compute_ratio_witherr(N_num, N_den)

print(f"Rate : {rate*L1A_physics}")
print(f"Rate_up : {rate_up*L1A_physics}")
print(f"Rate_down : {rate_low*L1A_physics}")
