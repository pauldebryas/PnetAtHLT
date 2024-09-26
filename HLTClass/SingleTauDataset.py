import awkward as ak
import numpy as np
from HLTClass.dataset import Dataset
from HLTClass.dataset import get_L1Taus, get_Taus, get_Jets, get_GenTaus, hGenTau_selection, matching_Gentaus, matching_L1Taus_obj, compute_PNet_charge_prob

# ------------------------------ functions for SingleTau with PNet ---------------------------------------------------------------
def compute_PNet_WP_SingleTau(tau_pt, par):
    # return PNet WP for SingleTau
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

def Jet_selection_SingleTau(events, par, apply_PNET_WP = True):
    # return mask for Jet passing selection for SingleTau path
    Jet_pt_corr = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
    Jets_mask = (events['Jet_pt'].compute() >= 30) & (np.abs(events['Jet_eta'].compute()) <= 2.3) & (Jet_pt_corr >= 30)
    if apply_PNET_WP:
        probTauP = events['Jet_PNet_probtauhp'].compute()
        probTauM = events['Jet_PNet_probtauhm'].compute()
        Jets_mask = Jets_mask & ((probTauP + probTauM) >= compute_PNet_WP_SingleTau(Jet_pt_corr, par)) & (Jet_pt_corr >= 130) & (compute_PNet_charge_prob(probTauP, probTauM) >= 0)
    return Jets_mask

def evt_sel_SingleTau(events, par, is_gen = False):
    # Selection of event passing condition of SingleTau with PNet HLT path + mask of objects passing those conditions

    L1Tau_Tau120er2p1L2NN_mask = L1Tau_Tau120er2p1_selection(events) & L1Tau_L2NN_selection_SingleTau(events)
    SingleTau_mask = Jet_selection_SingleTau(events, par)
    # at least 1 L1tau/ recoJet should pass
    SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1)
    if is_gen:
        # if MC data, at least 1 GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        SingleTau_evt_mask = SingleTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= 1)

    # matching
    L1Taus = get_L1Taus(events)
    Jets = get_Jets(events)
    L1Taus_SingleTau = L1Taus[L1Tau_Tau120er2p1L2NN_mask]
    Jets_SingleTau = Jets[SingleTau_mask]
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTau_SingleTau = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_SingleTau, Jets_SingleTau, GenTau_SingleTau)
        # at least 1 GenTau should match L1Tau/Jets
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 1)
    else:
        matchingJets_mask = matching_L1Taus_obj(L1Taus_SingleTau, Jets_SingleTau)
        # at least 1 Jet should match L1Tau
        evt_mask_matching = (ak.sum(matchingJets_mask, axis=-1) >= 1)
    
    SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching
    if is_gen: 
        return SingleTau_evt_mask, matchingGentaus_mask
    else:
        return SingleTau_evt_mask, matchingJets_mask
# ------------------------------ functions for LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3 ---------------------------------------------------
def compute_DeepTau_WP_SingleTau(tau_pt):
    # return DeepTau WP for LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3
    t1 = 0.6072
    t2 = 0.125
    x1 = 180
    x2 = 500
    Tau_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    Tau_WP = ak.where((tau_pt <= ones*x1) == False, Tau_WP, ones*t1)
    Tau_WP = ak.where((tau_pt >= ones*x2) == False, Tau_WP, ones*t2)
    Tau_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, Tau_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    return Tau_WP

def Tau_selection_SingleTau(events, apply_DeepTau_WP = True):
    # return mask for Tau passing selection for LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3
    tau_pt = events['Tau_pt'].compute()
    Tau_mask = (tau_pt >= 180) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
    if apply_DeepTau_WP:
        Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= compute_DeepTau_WP_SingleTau(tau_pt))
    return Tau_mask

def evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(events, is_gen = False):
    # Selection of event passing condition of LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3 + mask of objects passing those conditions

    L1Tau_Tau120er2p1L2NN_mask = L1Tau_Tau120er2p1_selection(events) & L1Tau_L2NN_selection_SingleTau(events)
    SingleTau_mask = Tau_selection_SingleTau(events)
    # at least 1 L1tau/ recoTau should pass
    SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1)
    print("SingleTau_evt_mask",SingleTau_evt_mask)
    if is_gen:
        # if MC data, at least 1 GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        SingleTau_evt_mask = SingleTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= 1)

    # matching
    L1Taus = get_L1Taus(events)
    Taus = get_Taus(events)
    L1Taus_SingleTau = L1Taus[L1Tau_Tau120er2p1L2NN_mask]
    Taus_SingleTau = Taus[SingleTau_mask]
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTau_SingleTau = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_SingleTau, Taus_SingleTau, GenTau_SingleTau)
        # at least 1 GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 1)
    else:
        matchingTaus_mask = matching_L1Taus_obj(L1Taus_SingleTau, Taus_SingleTau)
        # at least 1 Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingTaus_mask, axis=-1) >= 1)
    
    SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching
    if is_gen: 
        return SingleTau_evt_mask, matchingGentaus_mask
    else:
        return SingleTau_evt_mask, matchingTaus_mask
    
# ------------------------------ Common functions for Single tau path ---------------------------------------------------------------

def L1Tau_Tau120er2p1_selection(events):
    L1Taus = get_L1Taus(events)
    # return mask for L1tau passing Tau120er2p1 selection
    L1_Tau120er2p1_mask  = (events['L1Tau_pt'].compute() >= 120) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131)
    return L1_Tau120er2p1_mask

def L1Tau_L2NN_selection_SingleTau(events):
    # return mask for L1tau passing L2NN selection for LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3
    L1_L2NN_mask = ((events['L1Tau_l2Tag'].compute() > 0.8517) | (events['L1Tau_pt'].compute() >= 250))
    return L1_L2NN_mask

def Denominator_Selection_SingleTau(GenLepton):
    # return mask for event passing minimal Gen requirements for SingleTau HLT (exactly one hadronic Taus)
    mask = (GenLepton['kind'] == 5)
    ev_mask = ak.sum(mask, axis=-1) == 1
    return ev_mask




class SingleTauDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)
    
    # ------------------------------ functions to Compute Rate ---------------------------------------------------------------

    def get_Nnum_Nden_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(self):
        print(f'Computing rate for HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        SingleTau_evt_mask, matchingTaus_mask = evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(events, is_gen = False)
        N_num = len(events[SingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_SingleTauPNet(self, par):
        print(f'Computing Rate for SingleTau path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        SingleTau_evt_mask, matchingJets_mask = evt_sel_SingleTau(events, par, is_gen = False)
        N_num = len(events[SingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num
    
    def get_Nnum_Nden_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3_nPV(self, nPV):
        print(f'Computing rate for HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3 for nPV = {nPV}:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        events = events[events['nPFPrimaryVertex'].compute() == nPV]
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        SingleTau_evt_mask, matchingTaus_mask = evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(events, is_gen = False)
        N_num = len(events[SingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_SingleTauPNet_nPV(self, par, nPV):
        print(f'Computing Rate for SingleTau path with param: {par} and for nPV = {nPV}:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        events = events[events['nPFPrimaryVertex'].compute() == nPV]
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        SingleTau_evt_mask, matchingJets_mask = evt_sel_SingleTau(events, par, is_gen = False)
        N_num = len(events[SingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

# ------------------------------ functions for ComputeEfficiency ---------------------------------------------------------------

    def save_Event_Nden_eff_SingleTau(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for diTau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask = Denominator_Selection_SingleTau(GenLepton)
        print(f"Number of events with exactly 1 hadronic Tau (kind=TauDecayedToHadrons): {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def produceRoot_HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        mask_den_selection = ak.num(Tau_Den['pt']) >=2
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        SingleTau_evt_mask, matchingGentaus_mask = evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(events, is_gen = True)
        Tau_Num = (Tau_Den[matchingGentaus_mask])[SingleTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num = events[SingleTau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return

    def produceRoot_SingleTauPNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        mask_den_selection = ak.num(Tau_Den['pt']) >=2
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        SingleTau_evt_mask, matchingGentaus_mask = evt_sel_SingleTau(events, par, is_gen = True)

        Tau_Num = (Tau_Den[matchingGentaus_mask])[SingleTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")

        #events = events[SingleTau_evt_mask]
        events_Num = events[SingleTau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return
    
    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_SingleTauPNet(self, par):

        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        L1Taus = get_L1Taus(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_Tau120er2p1L2NN_mask = L1Tau_Tau120er2p1_selection(events) & L1Tau_L2NN_selection_SingleTau(events)
        SingleTau_mask = Jet_selection_SingleTau(events, par, apply_PNET_WP = False)
        GenTau_mask = hGenTau_selection(events)
        # at least 1 L1tau/ Jet/ GenTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        #matching
        L1Taus_SingleTau = L1Taus[L1Tau_Tau120er2p1L2NN_mask]
        Jets_SingleTau = Jets[SingleTau_mask]
        GenTaus_Sel = GenTaus[GenTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Jets_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  

        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        N_den = len(events[SingleTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        SingleTau_mask = Jet_selection_SingleTau(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        #matching
        Jets_SingleTau = Jets[SingleTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Jets_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1) 
        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        N_num = len(events[SingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def ComputeEffAlgo_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3(self):

        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        L1Taus = get_L1Taus(events)
        Taus = get_Taus(events)
        GenTaus = get_GenTaus(events)

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_Tau120er2p1L2NN_mask = L1Tau_Tau120er2p1_selection(events) & L1Tau_L2NN_selection_SingleTau(events)
        SingleTau_mask = Tau_selection_SingleTau(events, apply_DeepTau_WP = False)
        GenTau_mask = hGenTau_selection(events)
        # at least 1 L1tau/ Jet/ GenTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        #matching
        L1Taus_SingleTau = L1Taus[L1Tau_Tau120er2p1L2NN_mask]
        Taus_SingleTau = Taus[SingleTau_mask]
        GenTaus_Sel = GenTaus[GenTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Taus_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  # at least 1 Taus should match L1Tau

        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        N_den = len(events[SingleTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        SingleTau_mask = Tau_selection_SingleTau(events, apply_DeepTau_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        #matching
        Taus_SingleTau = Taus[SingleTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Taus_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  # at least 1 Taus should match L1Tau
        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        N_num = len(events[SingleTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

