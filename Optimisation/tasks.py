#!/usr/bin/env python
import law
import os
import shutil
import json

from law_customizations import Task, HTCondorWorkflow
from helpers import files_from_path, load_cfg_file, hadd_anatuple
from HLTClass.MuTauDataset import MuTauDataset
#from HLTClass.DiTauJetDataset import DiTauJetDataset
#from HLTClass.DiTauDataset import DiTauDataset
#from HLTClass.DoubleORSingleTauDataset import DoubleORSingleTauDataset
#from Optimisation.run_fullcombo import Threshold_optimiser
#from Optimisation.run_fulldeeptau import Threshold_optimiser as Th_opt_deeptau

"""
class SaveOptimisationResults(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''

    config = load_cfg_file()
    output_path = config['DATA']['result_opt']

    def create_branch_map(self):
        branches = {}
        os.makedirs(self.output_path, exist_ok=True)

        max_param_1 = {
            26: 0.7,
            27: 0.65,
            28: 0.6,
            29: 0.55,
            30: 0.5
        }

        max_param_2 = {
            # 26: 0.7,
            26: 0.5,
            27: 0.65,
            28: 0.6,
            29: 0.55,
            30: 0.5
        }

        min_param_1 = {
            # 26: 0.5,
            26: 0.6,
            27: 0.45,
            28: 0.4,
            29: 0.35,
            30: 0.3
        }

        # min_param_2 = 0.1
        min_param_2 = 0.4

        # min_param_1 = 0.2
        # # min_param_1 = 0.59
        # min_param_2 = 0.2
        # # min_param_2 = 0.59
        # max_param = 1.

        step = 0.01
        pt_cuts = [26, 27, 28]
        # pt_cuts = [26, 27, 28, 29, 30]
        index = 0
        for pt in pt_cuts:
            for i in range(int(round((max_param_1[pt] - min_param_1[pt]) / step)) + 1):
                for j in range(int(round((max_param_2[pt] - min_param_2) / step)) + 1):
                    param = (round(min_param_1[pt] + i * step, 2), round(min_param_2 + j * step, 2))
                    if param[1] > param[0]:
                        continue
                    branches[index] = (pt, param[0], param[1])
                    index += 1
        
        branches[index] = ("deeptau", )
        return branches
    
    def output(self):
        p = self.branch_data
        path = os.path.join(self.output_path, f'results_{"_".join([str(elem) for elem in p])}.json')
        return law.LocalFileTarget(path)

    def run(self):
        HLT_name = "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60"
        L1A_physics = float(self.config['RUNINFO']['L1A_physics'])

        RefRun = list(filter(None, (x.strip() for x in self.config['RUNINFO']['ref_run'].splitlines())))
        tag = f'Run_{RefRun[0]}' if len(RefRun) == 1 else f'Run_{"_".join(RefRun)}'
        Rate_path = os.path.join(self.config['DATA']['RateDenPath'], tag)
        Eff_path = os.path.join(self.config['DATA']['EffDenPath'], HLT_name)

        # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
        FileNameList_eff = [
            f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125.root"
        ]
        FileNameList_rate = files_from_path(Rate_path)
        FileNameList_rate = FileNameList_rate[0]

        params = self.branch_map[self.branch]
        
        if "deeptau" not in params:
            th = Threshold_optimiser()
        else:
            th = Th_opt_deeptau()
        th.dataset_eff = DiTauJetDataset(FileNameList_eff)
        th.dataset_rate = DiTauJetDataset(FileNameList_rate)

        if "deeptau" not in params:
            eff, rate = th.f((params[1], params[2]), params[0])
            eff_noditaujet, rate_noditaujet = th.f((params[1], params[2]), params[0], False)
        else:
            eff, rate = th.f()
            eff_noditaujet, rate_noditaujet = th.f(False)

        d = {
            "params": params, "eff": eff, "rate": rate,
            "eff_noditaujet": eff_noditaujet, "rate_noditaujet": rate_noditaujet
        }

        with open(self.output().path, "w+") as out:
            json.dump(d, out)


class SaveOptimisationResultsDiTau(SaveOptimisationResults):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''

    config = load_cfg_file()
    output_path = config['DATA']['result_opt_ditau']
    add_deeptau = True
    def create_branch_map(self):
        branches = {}
        os.makedirs(self.output_path, exist_ok=True)

        param_ranges = [
            (0.4, 0.65),
            (0.4, 0.55)
        ]

        # min_param_1 = 0.2
        # # min_param_1 = 0.59
        # min_param_2 = 0.2
        # # min_param_2 = 0.59
        # max_param = 1.

        step = 0.01
        # pt_cuts = [26, 27, 28, 29, 30]
        index = 0
        for i in range(int(round((param_ranges[0][1] - param_ranges[0][0]) / step)) + 1):
            for j in range(int(round((param_ranges[1][1] - param_ranges[1][0]) / step)) + 1):
                param = (round(param_ranges[0][0] + i * step, 2), round(param_ranges[1][0] + j * step, 2))
                if param[1] > param[0]:
                    continue
                branches[index] = (param[0], param[1])
                index += 1

        if self.add_deeptau:
            branches[index] = ("deeptau", )

        return branches

    def run(self):
        config = load_cfg_file()
        HLT_name = "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1"
        L1A_physics = float(self.config['RUNINFO']['L1A_physics'])

        RefRun = list(filter(None, (x.strip() for x in self.config['RUNINFO']['ref_run'].splitlines())))
        tag = f'Run_{RefRun[0]}' if len(RefRun) == 1 else f'Run_{"_".join(RefRun)}'
        Rate_path = os.path.join(self.config['DATA']['RateDenPath'], tag)
        Eff_path = os.path.join(self.config['DATA']['EffDenPath'], HLT_name)

        # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
        FileNameList_eff = [
            f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125.root"
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
        ]
        FileNameList_rate = files_from_path(Rate_path)
        FileNameList_rate = FileNameList_rate[0]
        dataset_eff = DiTauDataset(FileNameList_eff)
        dataset_rate = DiTauDataset(FileNameList_rate)

        params = self.branch_map[self.branch]
        if not "deeptau" in params:
            N_den, N_num = dataset_rate.get_Nnum_Nden_DiTauPNet(params)
            rate = (N_num/N_den)*L1A_physics

            N_den, N_num = dataset_eff.ComputeEffAlgo_DiTauPNet(params)
            eff = (N_num/N_den)
        else:
            N_den, N_num = dataset_rate.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()
            rate = (N_num/N_den)*L1A_physics

            N_den, N_num = dataset_eff.ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()
            eff = (N_num/N_den)

        d = {
            "params": params, "eff": eff, "rate": rate,
        }

        with open(self.output().path, "w+") as out:
            json.dump(d, out)


class SaveOptimisationResultsDiTauSingleTau(SaveOptimisationResultsDiTau):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''

    config = load_cfg_file()
    output_path = config['DATA']['result_opt_ditau_singletau']
    add_deeptau = False
    par_double = [0.57, 0.44, 0]

    den_double = False

    def create_branch_map(self):
        branches = {}
        os.makedirs(self.output_path, exist_ok=True)

        param_ranges = [
            (0.9, 1.1),
            (0.8, 1.)
        ]

        # min_param_1 = 0.2
        # # min_param_1 = 0.59
        # min_param_2 = 0.2
        # # min_param_2 = 0.59
        # max_param = 1.

        step = 0.01
        # pt_cuts = [26, 27, 28, 29, 30]
        index = 0
        for i in range(int(round((param_ranges[0][1] - param_ranges[0][0]) / step)) + 1):
            for j in range(int(round((param_ranges[1][1] - param_ranges[1][0]) / step)) + 1):
                param = (round(param_ranges[0][0] + i * step, 2), round(param_ranges[1][0] + j * step, 2))
                if param[1] > param[0]:
                    continue
                branches[index] = (param[0], param[1])
                index += 1

        if self.add_deeptau:
            branches[index] = ("deeptau", )

        return branches

    def run(self):
        HLT_name = "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1"
        L1A_physics = float(self.config['RUNINFO']['L1A_physics'])

        RefRun = list(filter(None, (x.strip() for x in self.config['RUNINFO']['ref_run'].splitlines())))
        tag = f'Run_{RefRun[0]}' if len(RefRun) == 1 else f'Run_{"_".join(RefRun)}'
        Rate_path = os.path.join(self.config['DATA']['RateDenPath'], tag)
        Eff_path = os.path.join(self.config['DATA']['EffDenPath'], HLT_name)

        # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
        FileNameList_eff = [
            f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125.root"
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
        ]
        FileNameList_rate = files_from_path(Rate_path)
        FileNameList_rate = FileNameList_rate[0]
        dataset_eff = DoubleORSingleTauDataset(FileNameList_eff)
        dataset_rate = DoubleORSingleTauDataset(FileNameList_rate)

        params = self.branch_map[self.branch]
        if not "deeptau" in params:
            N_den, N_num = dataset_rate.get_Nnum_Nden_DoubleORSinglePNet(params)
            rate = (N_num/N_den)*L1A_physics

            N_den, N_num = dataset_eff.ComputeEffAlgo_DoubleORSinglePNet(params)
            eff = (N_num/N_den)
        else:
            raise ValueError("Not implemented")
            N_den, N_num = dataset_rate.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()
            rate = (N_num/N_den)*L1A_physics

            N_den, N_num = dataset_eff.ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()
            eff = (N_num/N_den)

        d = {
            "params": params, "eff": eff, "rate": rate,
        }

        with open(self.output().path, "w+") as out:
            json.dump(d, out)


class SaveOptimisationResultsDiTauSingleTauDenDouble(SaveOptimisationResultsDiTauSingleTau):
    config = load_cfg_file()
    output_path = config['DATA']['result_opt_ditau_singletau_dendouble']
    add_deeptau = False
    par_double = [0.57, 0.44, 0]
    den_double = True
"""



class SaveOptimisationResultsMuTau(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''

    config = load_cfg_file()
    output_path = config['DATA']['result_opt_Etau']
    add_deeptau = True
    par_double = [0.57, 0.44, 0]

    den_double = False

    def create_branch_map(self):
        branches = {}
        os.makedirs(self.output_path, exist_ok=True)

        
        param_ranges = [
            (0.45, 0.60),
            (0.35, 0.50)
        ]

        # min_param_1 = 0.2
        # # min_param_1 = 0.59
        # min_param_2 = 0.2
        # # min_param_2 = 0.59
        # max_param = 1.

        step = 0.01
        # pt_cuts = [26, 27, 28, 29, 30]
        index = 0
        for i in range(int(round((param_ranges[0][1] - param_ranges[0][0]) / step)) + 1):
            for j in range(int(round((param_ranges[1][1] - param_ranges[1][0]) / step)) + 1):
                param = (round(param_ranges[0][0] + i * step, 2), round(param_ranges[1][0] + j * step, 2))
                if param[1] > param[0]:
                    continue
                branches[index] = (param[0], param[1])
                index += 1

        if self.add_deeptau:
            branches[index] = ("deeptau", )

        return branches

    def run(self):
        #HLT_name = "HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1"
        HLT_name = "HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1"
        L1A_physics = float(self.config['RUNINFO']['L1A_physics'])

        RefRun = list(filter(None, (x.strip() for x in self.config['RUNINFO']['ref_run'].splitlines())))
        tag = f'Run_{RefRun[0]}' if len(RefRun) == 1 else f'Run_{"_".join(RefRun)}'
        Rate_path = os.path.join(self.config['DATA']['RateDenPath'], tag)
        Eff_path = os.path.join(self.config['DATA']['EffDenPath'], HLT_name)

        # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
        FileNameList_eff = [
            f"/afs/cern.ch/work/s/skeshri/TauHLT/Braden/Forked/TauTriggerDev/EfficiencyDen/{HLT_name}/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-3p00.root"
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
            # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
        ]
        FileNameList_rate = files_from_path(Rate_path)
        FileNameList_rate = FileNameList_rate[0]
        dataset_eff = MuTauDataset(FileNameList_eff)
        dataset_rate = MuTauDataset(FileNameList_rate)

        params = self.branch_map[self.branch]
        if not "deeptau" in params:
            N_den, N_num = dataset_rate.get_Nnum_Nden_MuTauPNet( params)
            rate = (N_num/N_den)*L1A_physics

            N_den, N_num = dataset_eff.ComputeEffAlgo_MuTauPNet(params)
            eff = (N_num/N_den)
        else:
            #raise ValueError("Not implemented")
            N_den, N_num = dataset_rate.get_Nnum_Nden_MuTauDeepNet()
            rate = (N_num/N_den)*L1A_physics

            N_den, N_num = dataset_eff.ComputeEffAlgo_MuTauDeepNet()
            eff = (N_num/N_den)

        d = {
            "params": params, "eff": eff, "rate": rate,
        }

        with open(self.output().path, "w+") as out:
            json.dump(d, out)


    def output(self):
        p = self.branch_data
        path = os.path.join(self.output_path, f'results_{"_".join([str(elem) for elem in p])}.json')
        return law.LocalFileTarget(path)
