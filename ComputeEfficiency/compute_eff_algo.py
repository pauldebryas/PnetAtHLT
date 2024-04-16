from helpers import load_cfg_file, compute_ratio_witherr, files_from_path
import os

#compute Algo eff quickly

config = load_cfg_file()
HLT_name = config['HLT']['HLTname']
Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)

PNet_mode = config['OPT']['PNet_mode']
if PNet_mode == 'false':
    PNetMode = False
else:
    PNetMode = True
    PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]


# only one file otherwise it's too long
FileNameList_eff = f"/afs/cern.ch/work/s/skeshri/TauHLT/Braden/TauHLTOptimzation/PnetAtHLT/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
#FileNameList_eff = files_from_path(f"/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/EfficiencyDen/{HLT_name}/")

if HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
    from HLTClass.DiTauDataset import DiTauDataset
    dataset_eff = DiTauDataset(FileNameList_eff)

    if PNetMode:
        print(f'Quick EffAlgo computation for DiTau path with PNet param {PNetparam}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_DiTauPNet(PNetparam)
    else:
        print(f'Quick EffAlgo computation for {HLT_name}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()

if HLT_name == 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3':
    from HLTClass.SingleTauDataset import SingleTauDataset
    dataset_eff = SingleTauDataset(FileNameList_eff)

    if PNetMode:
        print(f'Quick EffAlgo computation for SingleTau path with PNet param {PNetparam}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_SingleTauPNet(PNetparam)
    else:
        print(f'Quick EffAlgo computation for {HLT_name}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_HLTLooseDeepTauPFTauHPS180_L2NN_eta2p1_v3()    

if HLT_name == 'HLT_Ele24_eta2p1_WPTight_Gsf_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1':
    from HLTClass.ETauDataset import ETauDataset
    dataset_eff = ETauDataset(FileNameList_eff)

    if PNetMode:
        print(f'Quick EffAlgo computation for SingleTau path with PNet param {PNetparam}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_ETauPNet(PNetparam)
    else:
        print(f'Quick EffAlgo computation for {HLT_name}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_ETauDeepNet()    

if HLT_name == 'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1':
    from HLTClass.MuTauDataset import MuTauDataset
    dataset_eff = MuTauDataset(FileNameList_eff)

    if PNetMode:
        print(f'Quick EffAlgo computation for SingleTau path with PNet param {PNetparam}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_MuTauPNet(PNetparam)
    else:
        print(f'Quick EffAlgo computation for {HLT_name}:')
        N_den, N_num = dataset_eff.ComputeEffAlgo_MuTauDeepNet()    

EffAlgo, EffAlgo_low, EffAlgo_up = compute_ratio_witherr(N_num, N_den)
print(f"Eff : {EffAlgo}")
print(f"Eff_up : {EffAlgo_up}")
print(f"Eff_down : {EffAlgo_low}")
