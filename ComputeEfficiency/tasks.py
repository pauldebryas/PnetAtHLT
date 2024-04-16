#!/usr/bin/env python
import law
import os
import shutil

from law_customizations import Task, HTCondorWorkflow
from helpers import files_from_path, load_cfg_file, hadd_anatuple

class SaveEventsDenEfficiency(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where Events passing denominator selection are saved 
    '''
    config = load_cfg_file()
    MCDataPath = config['DATA']['SamplesPath']
    tmp_folder = config['DATA']['tmpPath']
    HLT_name = config['HLT']['HLTname']
    MCDataFolderNames = list(filter(None, (x.strip() for x in config['DATA']['MCDataFolderNames'].splitlines())))
    output_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)
    
    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(len(self.MCDataFolderNames)):
            branches[i] = os.path.join(self.output_path, f'{self.MCDataFolderNames[i]}.root')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder {self.MCDataFolderNames[self.branch]}-----------------------------------------------")
        MCFolderName = os.path.join(self.MCDataPath, self.MCDataFolderNames[self.branch])
        output_root_file = os.path.join(self.output_path, f'{self.MCDataFolderNames[self.branch]}.root')
        output_tmp_folder = os.path.join(self.tmp_folder, self.HLT_name, self.MCDataFolderNames[self.branch])
        if os.path.exists(output_tmp_folder):
            print(f'A tmp folder which store tmp anatuple files exist already: being deleted')
            shutil.rmtree(output_tmp_folder)
        os.makedirs(output_tmp_folder)

        HLT_config = ['HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1', 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3', 'HLT_DoubleTauOrSingleTau','HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1','HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1']
        if self.HLT_name not in HLT_config:
            print(f'HLT name {self.HLT_name} not implemented in the code')
            raise

        # Produce tmp files
        FileNameList = files_from_path(MCFolderName)
        for FileName in FileNameList:
            print(f"Producing tmp file for {os.path.basename(FileName)}:")
            output_tmp_file = os.path.join(output_tmp_folder, os.path.basename(FileName))

            if self.HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
                from HLTClass.DiTauDataset import DiTauDataset
                MC_dataset = DiTauDataset(FileName)
                MC_dataset.save_Event_Nden_eff_DiTau(output_tmp_file)

            if self.HLT_name == 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3':
                from HLTClass.SingleTauDataset import SingleTauDataset
                MC_dataset = SingleTauDataset(FileName)
                MC_dataset.save_Event_Nden_eff_SingleTau(output_tmp_file)

            if self.HLT_name == 'HLT_DoubleTauOrSingleTau':
                from HLTClass.DoubleORSingleTauDataset import DoubleORSingleTauDataset
                MC_dataset = DoubleORSingleTauDataset(FileName)
                MC_dataset.save_Event_Nden_eff_DoubleORSingleTau(output_tmp_file)

            if self.HLT_name == 'HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1':
                from HLTClass.ETauDataset import ETauDataset
                MC_dataset = ETauDataset(FileName)
                MC_dataset.save_Event_Nden_eff_ETau(output_tmp_file)

            if self.HLT_name == 'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1':
                from HLTClass.MuTauDataset import MuTauDataset
                MC_dataset = MuTauDataset(FileName)
                MC_dataset.save_Event_Nden_eff_MuTau(output_tmp_file)

        # Hadd the tmp files to a single root file
        hadd_anatuple(output_tmp_folder, output_root_file)
        shutil.rmtree(output_tmp_folder)
        return

class ProduceEfficiencyFiles(Task, HTCondorWorkflow, law.LocalWorkflow):
    '''
    Produce root file where informations of events that pass numerator/denominator selection are saved
    '''
    config = load_cfg_file()
    HLT_name = config['HLT']['HLTname']
    input_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)
    MCDataFolderNames = list(filter(None, (x.strip() for x in config['DATA']['MCDataFolderNames'].splitlines())))

    PNet_mode = config['OPT']['PNet_mode']
    if PNet_mode == 'false':
        PNetMode = False
        output_path = os.path.join(config['DATA']['result_eff'], HLT_name, HLT_name)
    else:
        PNetMode = True
        PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]
        output_path = os.path.join(config['DATA']['result_eff'], HLT_name, f'PNetTresh_{config["OPT"]["PNet_t1"]}_{config["OPT"]["PNet_t2"]}_{config["OPT"]["PNet_t3"]}')

    # requires SaveEventsDenEfficiency for Den events
    def workflow_requires(self):
        return { "Counter": SaveEventsDenEfficiency.req(self, branch=self.branch) }

    def requires(self):
        return self.workflow_requires()
    
    def create_branch_map(self):
        os.makedirs(self.output_path, exist_ok=True)
        branches = {}
        for i in range(len(self.MCDataFolderNames)):
            branches[i] = os.path.join(self.output_path, f'{self.MCDataFolderNames[i]}.root')
        return branches
    
    def output(self):
        path = os.path.join(self.branch_data)
        return law.LocalFileTarget(path)

    def run(self):
        print(f"-----------------------------------------------Processing folder {self.MCDataFolderNames[self.branch]}-----------------------------------------------")
        input_root_file = os.path.join(self.input_path, f'{self.MCDataFolderNames[self.branch]}.root')
        output_root_file = os.path.join(self.output_path, f'{self.MCDataFolderNames[self.branch]}.root')

        if not os.path.exists(input_root_file):
            raise('Input root file does not exist')

        HLT_config = ['HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1', 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3', 'HLT_DoubleTauOrSingleTau','HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1','HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1']
        if self.HLT_name not in HLT_config:
            print(f'HLT name {self.HLT_name} not implemented in the code')
            raise

        if self.HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
            from HLTClass.DiTauDataset import DiTauDataset

            MC_dataset = DiTauDataset(input_root_file)
            if self.PNetMode:
                MC_dataset.produceRoot_DiTauPNet(output_root_file, self.PNetparam)
            else:
                MC_dataset.produceRoot_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1(output_root_file)

        if self.HLT_name == 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3':
            from HLTClass.SingleTauDataset import SingleTauDataset

            MC_dataset = SingleTauDataset(input_root_file)
            if self.PNetMode:
                MC_dataset.produceRoot_SingleTauPNet(output_root_file, self.PNetparam)
            else:
                MC_dataset.produceRoot_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(output_root_file)

        if self.HLT_name == 'HLT_DoubleTauOrSingleTau':
            from HLTClass.DoubleORSingleTauDataset import DoubleORSingleTauDataset

            MC_dataset = DoubleORSingleTauDataset(input_root_file)
            if self.PNetMode:
                MC_dataset.produceRoot_DoubleORSinglePNet(output_root_file, self.PNetparam)
            else:
                MC_dataset.produceRoot_DoubleORSingleDeepTau(output_root_file)
        
        if self.HLT_name == 'HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1':
            from HLTClass.ETauDataset import ETauDataset

            MC_dataset = ETauDataset(input_root_file)
            if self.PNetMode:
                MC_dataset.produceRoot_ETauPNet(output_root_file, self.PNetparam)
            else:
                MC_dataset.produceRoot_ETauDeepNet(output_root_file)
        
        if self.HLT_name == 'HLT_IsoMu20_eta2p1_LooseDeepTauPFTauHPS27_eta2p1_CrossL1':
            from HLTClass.MuTauDataset import MuTauDataset

            MC_dataset = MuTauDataset(input_root_file)
            if self.PNetMode:
                MC_dataset.produceRoot_MuTauPNet(output_root_file, self.PNetparam)
            else:
                MC_dataset.produceRoot_MuTauDeepNet(output_root_file)
