import awkward as ak
import numpy as np
from HLTClass.dataset import Dataset
from HLTClass.dataset import get_L1Taus, get_Taus, get_Jets, get_GenTaus, hGenTau_selection, matching_Gentaus, matching_L1Taus_obj, compute_PNet_charge_prob, get_L1Egamma, get_Electrons,get_GenElectrons, GenElectron_selection, matching_GenElectrons, matching_L1Egamma_obj
from helpers import delta_r


# ------------------------------ functions for ETau with PNet ---------------------------------------------------------------

def compute_PNet_WP_ETau(tau_pt, par):
    # return PNet WP for ETau
    t1 = par[0]
    t2 = par[1]
    t3 = 0.001
    t4 = 0
    x1 = 130
    x2 = 200
    x3 = 500
    x4 = 1000
    PNet_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    PNet_WP = ak.where((tau_pt <= ones*x1) == False, PNet_WP, ones*t1)
    PNet_WP = ak.where((tau_pt >= ones*x4) == False, PNet_WP, ones*t4)
    PNet_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, PNet_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    PNet_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, PNet_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    PNet_WP = ak.where(((tau_pt >= ones*x3) & (tau_pt < ones*x4))== False, PNet_WP, (t4 - t3) / (x4 - x3) * (tau_pt - ones*x3) + ones*t3)
    return PNet_WP


def Jet_selection_Tau(events, par, apply_PNET_WP = True):
    # return mask for Jet passing selection for ETau path
    Jet_pt_corr = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
    Jets_mask = (events['Jet_pt'].compute() >= 30) & (np.abs(events['Jet_eta'].compute()) <= 2.3) & (Jet_pt_corr >= 30)
    if apply_PNET_WP:
        probTauP = events['Jet_PNet_probtauhp'].compute()
        probTauM = events['Jet_PNet_probtauhm'].compute()
        Jets_mask = Jets_mask & ((probTauP + probTauM) >= compute_PNet_WP_ETau(Jet_pt_corr, par)) & (compute_PNet_charge_prob(probTauP, probTauM) >= 0)
    return Jets_mask

def Electron_selection(events):
    Electron_mask = (events['Electron_pt'].compute() >= 20) & (np.abs(events['Electron_eta'].compute()) <= 2.3) & (events['Electron_passSingleElectron'].compute() | events['Electron_passETau'].compute())
    return Electron_mask


def matching_dR_Min0p3(L1Taus, Obj, dR_matching_min = 0.3):
    obj_inpair, l1taus_inpair = ak.unzip(ak.cartesian([Obj, L1Taus], nested=True))
    dR_obj_l1taus = delta_r(obj_inpair, l1taus_inpair)
    mask_obj_l1taus = (dR_obj_l1taus > dR_matching_min)
    
    mask = ak.any(mask_obj_l1taus, axis=-1)
    return mask



def evt_sel_ETau(events, par, is_gen = False):
    # Selection of event passing condition of ETau with PNet HLT path + mask of objects passing those conditions

    L1_EG_mask = L1_LooseIsoEG22er2p1_selection(events) | L1_LooseIsoEG24er2p1_selcetion(events) 
    L1_Tau_mask = L1_IsoTau26er2p1_selection(events) | L1_Tau70er2p1_selection(events) | L1_IsoTau27er2p1_selcetion(events)
    
    Tau_mask = Jet_selection_Tau(events, par, apply_PNET_WP = True)

    Electron_mask = Electron_selection(events)
    ETau_evt_mask = (ak.sum(L1_EG_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Electron_mask, axis=-1) >= 1)
    
    L1Taus = get_L1Taus(events)
    L1Taus_Tau = L1Taus[L1_Tau_mask]
    
    L1Egamma = get_L1Egamma(events)
    L1Egamma_EG = L1Egamma[L1_EG_mask]
    
    Jets = get_Jets(events)
    Jets_Tau = Jets[Tau_mask]
    
    Electrons = get_Electrons(events)
    Selected_Electrons = Electrons[Electron_mask]

    matching_dR_Min0p3_L1 = matching_dR_Min0p3(L1Taus_Tau, L1Egamma_EG, dR_matching_min=0.3)

    ETau_evt_mask = ETau_evt_mask & matching_dR_Min0p3_L1

    if is_gen:
        # if MC data, at least 1 GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenElectron_mask = GenElectron_selection(events)
        ETau_evt_mask = ETau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= 1) & (ak.sum(GenElectron_mask, axis=-1) >= 1)

    # matching
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_Tau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_Tau, Jets_Tau, GenTaus_Tau)
        # at least 1 GenTau should match L1Tau/Taus
        evt_Taumask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 1)

        GenElectrons = get_GenElectrons(events)
        GenElectrons_Electron  = GenElectrons[GenElectron_mask]
        matchingGenElectrons_mask = matching_GenElectrons(L1Egamma_EG, Selected_Electrons, GenElectrons_Electron)
        # at least 1 GenTau should match L1Tau/Taus
        evt_Electronmask_matching = (ak.sum(matchingGenElectrons_mask, axis=-1) >= 1)

    else:
        matchingJets_mask = matching_L1Taus_obj(L1Taus_Tau, Jets_Tau)
        # at least 1 Tau should match L1Tau
        evt_Taumask_matching = (ak.sum(matchingJets_mask, axis=-1) >= 1)

        matchingElectrons_mask = matching_L1Egamma_obj(L1Egamma_EG, Selected_Electrons)
        evt_Electronmask_matching = (ak.sum(matchingElectrons_mask, axis=-1) >= 1)

    ETau_evt_mask = ETau_evt_mask & evt_Taumask_matching & evt_Electronmask_matching
    if is_gen: 
        return ETau_evt_mask, matchingGentaus_mask, matchingGenElectrons_mask
    else:
        return ETau_evt_mask, matchingJets_mask, matchingElectrons_mask
    
# ------------------------------ functions for HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1 ---------------------------------------------------
def compute_DeepTau_WP_ETau(tau_pt):
    # return DeepTau WP for LooseDeepTauPFTauHPS30_eta2p1 
    t1 = 0.649
    t2 = 0.441
    t3 = 0.05
    x1 = 35
    x2 = 100
    x3 = 300
    Tau_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    Tau_WP = ak.where((tau_pt <= ones*x1) == False, Tau_WP, ones*t1)
    Tau_WP = ak.where((tau_pt >= ones*x3) == False, Tau_WP, ones*t3)
    Tau_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, Tau_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    Tau_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, Tau_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    return Tau_WP

def Tau_selection_Tau(events, apply_DeepTau_WP = True):
    # return mask for Tau passing selection for HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1
    tau_pt = events['Tau_pt'].compute()
    Tau_mask = (tau_pt >= 30) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
    if apply_DeepTau_WP:
        Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= compute_DeepTau_WP_ETau(tau_pt))
    return Tau_mask

def evt_sel_HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1(events, is_gen = False):
    # Selection of event passing condition of HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1 + mask of objects passing those conditions

    L1_EG_mask = L1_LooseIsoEG22er2p1_selection(events) | L1_LooseIsoEG24er2p1_selcetion(events) 
    L1_Tau_mask = L1_IsoTau26er2p1_selection(events) | L1_Tau70er2p1_selection(events) | L1_IsoTau27er2p1_selcetion(events)
    
    Tau_mask = Tau_selection_Tau(events)

    Electron_mask = Electron_selection(events)
    # at least 1 L1tau/ recoJet should pass
    ETau_evt_mask = (ak.sum(L1_EG_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Electron_mask, axis=-1) >= 1)
    
    L1Taus = get_L1Taus(events)
    L1Taus_Tau = L1Taus[L1_Tau_mask]
    
    L1Egamma = get_L1Egamma(events)
    L1Egamma_EG = L1Egamma[L1_EG_mask]
    
    Taus = get_Taus(events)
    Taus_Tau = Taus[Tau_mask]
    
    Electrons = get_Electrons(events)
    Selected_Electrons = Electrons[Electron_mask]

    matching_dR_Min0p3_L1 = matching_dR_Min0p3(L1Taus_Tau, L1Egamma_EG, dR_matching_min=0.3)

    ETau_evt_mask = ETau_evt_mask & matching_dR_Min0p3_L1
    print("ETau_evt_mask",ETau_evt_mask)

    if is_gen:
        # if MC data, at least 1 GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenElectron_mask = GenElectron_selection(events)
        ETau_evt_mask = ETau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= 1) & (ak.sum(GenElectron_mask, axis=-1) >= 1)

    # matching
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_Tau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_Tau, Taus_Tau, GenTaus_Tau)
        # at least 1 GenTau should match L1Tau/Taus
        evt_Taumask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 1)

        GenElectrons = get_GenElectrons(events)
        GenElectrons_Electron  = GenElectrons[GenElectron_mask]
        matchingGenElectrons_mask = matching_GenElectrons(L1Egamma_EG, Selected_Electrons, GenElectrons_Electron)
        # at least 1 GenTau decay to electron should match L1Tau/Taus
        evt_Electronmask_matching = (ak.sum(matchingGenElectrons_mask, axis=-1) >= 1)

    else:
        matchingTaus_mask = matching_L1Taus_obj(L1Taus_Tau, Taus_Tau)
        # at least 1 Tau should match L1Tau
        evt_Taumask_matching = (ak.sum(matchingTaus_mask, axis=-1) >= 1)

        matchingElectrons_mask = matching_L1Egamma_obj(L1Egamma_EG, Selected_Electrons)
        # at least 1 L1EG should match L1Tau
        evt_Electronmask_matching = (ak.sum(matchingElectrons_mask, axis=-1) >= 1)

    ETau_evt_mask = ETau_evt_mask & evt_Taumask_matching & evt_Electronmask_matching
    if is_gen: 
        return ETau_evt_mask, matchingGentaus_mask, matchingGenElectrons_mask
    else:
        return ETau_evt_mask, matchingTaus_mask, matchingElectrons_mask

# ------------------------------ Common functions for Ditau path ---------------------------------------------------------------
def L1Tau_Tau130er2p1_selection(events):
    L1Taus = get_L1Taus(events)
    # return mask for L1tau passing Tau120er2p1 selection
    L1_Tau120er2p1_mask  = (events['L1Tau_pt'].compute() >= 130) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131)
    return L1_Tau130er2p1_mask

def L1_LooseIsoEG22er2p1_selection(events):
    L1_LooseIsoEG22er2p1 = (events['L1Egamma_pt'].compute() >= 22) & (events['L1Egamma_eta'].compute() <= 2.131) & (events['L1Egamma_eta'].compute() >= -2.131)
    return L1_LooseIsoEG22er2p1

def L1_IsoTau26er2p1_selection(events):
    L1_IsoTau26er2p1 = (events['L1Tau_pt'].compute() >= 26) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) & (events['L1Tau_hwIso'].compute() > 0)
    return L1_IsoTau26er2p1

def L1_Tau70er2p1_selection(events):
    L1_Tau70er2p1 = (events['L1Tau_pt'].compute() >= 70) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) 
    return L1_Tau70er2p1

def L1_LooseIsoEG24er2p1_selcetion(events):
    L1_LooseIsoEG24er2p1 = (events['L1Egamma_pt'].compute() >= 24) & (events['L1Egamma_eta'].compute() <= 2.131) & (events['L1Egamma_eta'].compute() >= -2.131)
    return L1_LooseIsoEG24er2p1

def L1_IsoTau27er2p1_selcetion(events):
    L1_IsoTau27er2p1 = (events['L1Tau_pt'].compute() >= 27) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) & (events['L1Tau_hwIso'].compute() > 0)
    return L1_IsoTau27er2p1


def Denominator_Selection_ETau(GenLepton):
    # return mask for event passing minimal Gen requirements for ETau HLT (1 hadronic Taus with min vis. pt and eta and 1 Tau decayed to electron)
    mask_GenTau = (GenLepton['kind'] == 5)
    mask_GenEle = (GenLepton['kind'] == 3)
    ev_mask = (ak.sum(mask_GenTau, axis=-1) == 1) &  (ak.sum(mask_GenEle, axis=-1) == 1) # 1 Gen tau and one Gen Ele should pass this requirements
    return ev_mask

class ETauDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)
    
    # ------------------------------ functions to Compute Rate ---------------------------------------------------------------

    def get_Nnum_Nden_ETauDeepNet(self):
        print(f'Computing rate for HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        ETau_evt_mask, matchingTaus_mask, matchingElectrons_mask = evt_sel_HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1(events, is_gen = False)
        N_num = len(events[ETau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_ETauPNet(self, par):
        print(f'Computing Rate for ETau path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")
        ETau_evt_mask, matchingJets_mask, matchingElectrons_mask = evt_sel_ETau(events, par, is_gen = False)
        N_num = len(events[ETau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num
    
    def get_Nnum_Nden_ETauDeepNet_nPV(self, nPV):
        print(f'Computing rate for HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1 for nPV = {nPV}:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        events = events[events['nPFPrimaryVertex'].compute() == nPV]
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        ETau_evt_mask, matchingTaus_mask, matchingElectrons_mask = evt_sel_HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1(events, is_gen = False)
        N_num = len(events[ETau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_ETauPNet_nPV(self, par, nPV):
        print(f'Computing Rate for ETau path with param: {par} and for nPV = {nPV}:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        events = events[events['nPFPrimaryVertex'].compute() == nPV]
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        ETau_evt_mask, matchingJets_mask, matchingElectrons_mask = evt_sel_ETau(events, par, is_gen = False)
        N_num = len(events[ETau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

# ------------------------------ functions for ComputeEfficiency ---------------------------------------------------------------

    def save_Event_Nden_eff_ETau(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for ETau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask = Denominator_Selection_ETau(GenLepton)
        print(f"Number of events with exactly 1 hadronic Tau and one Electron (kind=TauDecayedToHadrons and TauDecayedTo Electron): {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def produceRoot_ETauDeepNet(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        GenElectron_mask = GenElectron_selection(events)
        GenElectrons = get_GenElectrons(events)
        Electron_Den  = GenElectrons[GenElectron_mask]

        mask_den_selection = ( ak.num(Tau_Den['pt']) >=1 ) & ( ak.num(Electron_Den['pt']) >=1 )
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        ETau_evt_mask, matchingGentaus_mask, matchingElectrons_mask = evt_sel_HLT_Ele24_eta2p1_WPTight_Gsf_LooseDeepTauPFTauHPS30_eta2p1_CrossL1(events, is_gen = True)
        Tau_Num = (Tau_Den[matchingGentaus_mask])[ETau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num = events[ETau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return

    def produceRoot_ETauPNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        GenElectron_mask = GenElectron_selection(events)
        GenElectrons = get_GenElectrons(events)
        Electron_Den  = GenElectrons[GenElectron_mask]

        mask_den_selection = ( ak.num(Tau_Den['pt']) >=1 ) & ( ak.num(Electron_Den['pt']) >=1 )
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        ETau_evt_mask, matchingGentaus_mask, matchingElectrons_mask = evt_sel_ETau(events, par, is_gen = True)

        Tau_Num = (Tau_Den[matchingGentaus_mask])[ETau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")

        #events = events[SingleTau_evt_mask]
        events_Num = events[ETau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return
    
    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_ETauPNet(self, par):

        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()

        # Selection of L1/Gen and Jets objects without PNET WP
        L1_EG_mask = L1_LooseIsoEG22er2p1_selection(events) | L1_LooseIsoEG24er2p1_selcetion(events) 
        L1_Tau_mask = L1_IsoTau26er2p1_selection(events) | L1_Tau70er2p1_selection(events) | L1_IsoTau27er2p1_selcetion(events)
        Tau_mask = Jet_selection_Tau(events, par, apply_PNET_WP = False)
        GenTau_mask = hGenTau_selection(events)
        Electron_mask = Electron_selection(events)
        GenElectron_mask = GenElectron_selection(events)

        # at least 1 L1tau/ Jet/ GenTau should pass
        ETau_evt_mask = (ak.sum(L1_EG_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Electron_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        L1Taus = get_L1Taus(events)
        L1Taus_Tau = L1Taus[L1_Tau_mask]

        L1Egamma = get_L1Egamma(events)
        L1Egamma_EG = L1Egamma[L1_EG_mask]

        Jets = get_Jets(events)
        Jets_Tau = Jets[Tau_mask]

        Electrons = get_Electrons(events)
        Selected_Electrons = Electrons[Electron_mask]   

        GenTaus = get_GenTaus(events)
        GenTaus_Sel = GenTaus[GenTau_mask]

        GenElectrons = get_GenElectrons(events)
        GenElectrons_Electrons  = GenElectrons[GenElectron_mask]
    
        matching_dR_Min0p3_L1 = matching_dR_Min0p3(L1Taus_Tau, L1Egamma_EG, dR_matching_min=0.3)

        ETau_evt_mask = ETau_evt_mask & matching_dR_Min0p3_L1        

        #matching        
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Jets_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1)  

        matchingGenElectrons_mask = matching_GenElectrons(L1Egamma_EG, Selected_Electrons, GenElectrons_Electron)
        evt_Electronmask_matching = (ak.sum(matchingGenElectrons_mask, axis=-1) >= 1)

        ETau_evt_mask = ETau_evt_mask & evt_Taumask_matching & evt_Electronmask_matching

        N_den = len(events[ETau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        Tau_mask = Jet_selection_Tau(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        ETau_evt_mask = (ak.sum(L1_EG_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Electron_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1) 
        ETau_evt_mask = ETau_evt_mask & matching_dR_Min0p3_L1 

        #matching
        Jets_Tau = Jets[Tau_mask]
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Jets_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1) 
        ETau_evt_mask = ETau_evt_mask & evt_Taumask_matching & evt_Electronmask_matching

        N_num = len(events[ETau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def ComputeEffAlgo_ETauDeepNet(self):

        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()

        # Selection of L1/Gen and Jets objects without PNET WP
        L1_EG_mask = L1_LooseIsoEG22er2p1_selection(events) | L1_LooseIsoEG24er2p1_selcetion(events) 
        L1_Tau_mask = L1_IsoTau26er2p1_selection(events) | L1_Tau70er2p1_selection(events) | L1_IsoTau27er2p1_selcetion(events)
        Tau_mask = Tau_selection_Tau(events, apply_DeepTau_WP = False)
        GenTau_mask = hGenTau_selection(events)
        Electron_mask = Electron_selection(events)
        GenElectron_mask = GenElectron_selection(events)

        # at least 1 L1tau/ Jet/ GenTau should pass
        ETau_evt_mask = (ak.sum(L1_EG_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Electron_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        L1Taus = get_L1Taus(events)
        L1Taus_Tau = L1Taus[L1_Tau_mask]

        L1Egamma = get_L1Egamma(events)
        L1Egamma_EG = L1Egamma[L1_EG_mask]

        Taus = get_Taus(events)
        Taus_Tau = Taus[Tau_mask]

        Electrons = get_Electrons(events)
        Selected_Electrons = Electrons[Electron_mask]   

        GenTaus = get_GenTaus(events)
        GenTaus_Sel = GenTaus[GenTau_mask]

        GenElectrons = get_GenElectrons(events)
        GenElectrons_Electrons  = GenElectrons[GenElectron_mask]
    
        matching_dR_Min0p3_L1 = matching_dR_Min0p3(L1Taus_Tau, L1Egamma_EG, dR_matching_min=0.3)

        ETau_evt_mask = ETau_evt_mask & matching_dR_Min0p3_L1        

        #matching        
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Taus_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1)  

        matchingGenElectrons_mask = matching_GenElectrons(L1Egamma_EG, Selected_Electrons, GenElectrons_Electron)
        evt_Electronmask_matching = (ak.sum(matchingGenElectrons_mask, axis=-1) >= 1)

        ETau_evt_mask = ETau_evt_mask & evt_Taumask_matching & evt_Electronmask_matching

        N_den = len(events[ETau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        Tau_mask = Tau_selection_Tau(events, apply_DeepTau_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        ETau_evt_mask = (ak.sum(L1_EG_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Electron_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1) 
        ETau_evt_mask = ETau_evt_mask & matching_dR_Min0p3_L1 

        #matching
        Taus_Tau = Taus[Tau_mask]
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Taus_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1) 
        ETau_evt_mask = ETau_evt_mask & evt_Taumask_matching & evt_Electronmask_matching

        N_num = len(events[ETau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num


