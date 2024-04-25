import awkward as ak
import numpy as np
from HLTClass.dataset import Dataset
from HLTClass.dataset import get_L1Taus, get_Taus, get_Jets, get_GenTaus, hGenTau_selection, matching_Gentaus, matching_L1Taus_obj, compute_PNet_charge_prob, get_L1Muons, get_Muons, get_GenMuons, GenMuon_selection, matching_GenMuons, matching_L1Muons_obj
from helpers import delta_r


def matching_dR_Min0p5(Obj1, Obj2, dR_matching_min = 0.5):
    obj1_inpair, obj2_inpair = ak.unzip(ak.cartesian([Obj2, Obj1], nested=True))
    dR_obj2_obj1 = delta_r(obj2_inpair, obj1_inpair)
    mask_obj2_obj1 = (dR_obj2_obj1 > dR_matching_min)

    mask = ak.any(mask_obj2_obj1, axis=-1)
    return mask


# ------------------------------ functions for MuTau with PNet ---------------------------------------------------------------
def compute_PNet_WP_MuTau(tau_pt, par):
    # return PNet WP for MuTau
    t1 = par[0]
    t2 = par[1]
    t3 = 0.001
    t4 = 0
    x1 = 27
    x2 = 100
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
    # return mask for Jet passing selection for MuTau path
    Jet_pt_corr = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
    Jets_mask = (events['Jet_pt'].compute() >= 27) & (np.abs(events['Jet_eta'].compute()) <= 2.3) & (Jet_pt_corr >= 27)
    if apply_PNET_WP:
        probTauP = events['Jet_PNet_probtauhp'].compute()
        probTauM = events['Jet_PNet_probtauhm'].compute()
        Jets_mask = Jets_mask & ((probTauP + probTauM) >= compute_PNet_WP_MuTau(Jet_pt_corr, par)) & (compute_PNet_charge_prob(probTauP, probTauM) >= 0)
    return Jets_mask

def Muon_selection(events):
    Muon_mask = (events['Muon_pt'].compute() >= 20) & (np.abs(events['Muon_eta'].compute()) <= 2.3) & (events['Muon_passSingleMuon'].compute() | events['Muon_passMuTau'].compute())
    return Muon_mask

def evt_sel_MuTau(events, par, is_gen = False):
    # Selection of event passing condition of MuTau with PNet HLT path + mask of objects passing those conditions

    L1_Mu_mask = L1_Mu18er2p1_selection(events) 
    L1_Tau_mask = L1_Tau24er2p1_OR_Tau26er2p1_selection(events)
    
    Tau_mask = Jet_selection_Tau(events, par, apply_PNET_WP = True)
    
    Muon_mask = Muon_selection(events)
    # at least 1 L1tau/ recoJet should pass
    MuTau_evt_mask = (ak.sum(L1_Mu_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Muon_mask, axis=-1) >= 1)
    
    L1Taus = get_L1Taus(events)
    L1Taus_Tau = L1Taus[L1_Tau_mask]
    
    L1Muons = get_L1Muons(events)
    L1Muons_Muon = L1Muons[L1_Mu_mask]
    
    Jets = get_Jets(events)
    Jets_Tau = Jets[Tau_mask]
    
    Muons = get_Muons(events)
    Selected_Muons = Muons[Muon_mask]

    #overlap_mask1 = matching_dR_Min0p5(L1Muons_Muon, L1Taus_Tau)
    #overlap_mask2 = matching_dR_Min0p5(Selected_Muons, Jets_Tau)

    if is_gen:
        # if MC data, at least 1 GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenMuon_mask = GenMuon_selection(events)
        MuTau_evt_mask = MuTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= 1) & (ak.sum(GenMuon_mask, axis=-1) >= 1)

    # matching
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_Tau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_Tau, Jets_Tau, GenTaus_Tau)
        # at least 1 GenTau should match L1Tau/Taus
        evt_Taumask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 1)
        
        GenMuons = get_GenMuons(events)
        GenMuons_Muon  = GenMuons[GenMuon_mask]
        matchingGenMuons_mask = matching_GenMuons(L1Muons_Muon, Selected_Muons, GenMuons_Muon)
        #overlap_mask3 = matching_dR_Min0p5(GenMuons_Muon,GenTaus_Tau)
        #overlap_mask = overlap_mask1 & overlap_mask2 & overlap_mask3
        # at least 1 GenTau should match L1Tau/Taus
        evt_Muonmask_matching = (ak.sum(matchingGenMuons_mask, axis=-1) >= 1)
    else:
        matchingJets_mask = matching_L1Taus_obj(L1Taus_Tau, Jets_Tau)
        # at least 1 Tau should match L1Tau
        evt_Taumask_matching = (ak.sum(matchingJets_mask, axis=-1) >= 1)
        
        matchingMuons_mask = matching_L1Muons_obj(L1Muons_Muon, Selected_Muons)
        evt_Muonmask_matching = (ak.sum(matchingMuons_mask, axis=-1) >= 1)
        #overlap_mask = overlap_mask1 & overlap_mask2 

    #MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching & overlap_mask
    MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching 
    if is_gen: 
        return MuTau_evt_mask, matchingGentaus_mask, matchingGenMuons_mask
    else:
        return MuTau_evt_mask, matchingJets_mask, matchingMuons_mask
    
# ------------------------------ functions for DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1 ---------------------------------------------------
def compute_DeepTau_WP_MuTau(tau_pt):
    # return DeepTau WP for LooseDeepTauPFTauHPS27_eta2p1
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
    # return mask for Tau passing selection for LooseDeepTauPFTauHPS27_eta2p1
    tau_pt = events['Tau_pt'].compute()
    Tau_mask = (tau_pt >= 30) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
    if apply_DeepTau_WP:
        Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= compute_DeepTau_WP_MuTau(tau_pt))
    return Tau_mask

def evt_sel_HLT_IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1(events, is_gen = False):
    # Selection of event passing condition of IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1 + mask of objects passing those conditions

    L1_Mu_mask = L1_Mu18er2p1_selection(events) 
    L1_Tau_mask = L1_Tau24er2p1_OR_Tau26er2p1_selection(events)
    
    Tau_mask = Tau_selection_Tau(events)
    
    Muon_mask = Muon_selection(events)
    # at least 1 L1tau/ recoJet should pass
    MuTau_evt_mask = (ak.sum(L1_Mu_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Muon_mask, axis=-1) >= 1)
    
    L1Taus = get_L1Taus(events)
    L1Taus_Tau = L1Taus[L1_Tau_mask]
    
    L1Muons = get_L1Muons(events)
    L1Muons_Muon = L1Muons[L1_Mu_mask]
    
    Taus = get_Taus(events)
    Taus_Tau = Taus[Tau_mask]
    
    Muons = get_Muons(events)
    Selected_Muons = Muons[Muon_mask]

    #overlap_mask1 = matching_dR_Min0p5(L1Muons_Muon, L1Taus_Tau)
    #overlap_mask2 = matching_dR_Min0p5(Selected_Muons, Taus_Tau)
    

    if is_gen:
        # if MC data, at least 1 GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenMuon_mask = GenMuon_selection(events)
        MuTau_evt_mask = MuTau_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= 1) & (ak.sum(GenMuon_mask, axis=-1) >= 1)

    # matching
    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_Tau  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_Tau, Taus_Tau, GenTaus_Tau)
        # at least 1 GenTau should match L1Tau/Taus
        evt_Taumask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 1)
        
        GenMuons = get_GenMuons(events)
        GenMuons_Muon  = GenMuons[GenMuon_mask]
        matchingGenMuons_mask = matching_GenMuons(L1Muons_Muon, Selected_Muons, GenMuons_Muon)
        #overlap_mask3 = matching_dR_Min0p5(GenMuons_Muon,GenTaus_Tau)
        #overlap_mask = overlap_mask1 & overlap_mask2 & overlap_mask3
        # at least 1 GenTau should match L1Tau/Taus
        evt_Muonmask_matching = (ak.sum(matchingGenMuons_mask, axis=-1) >= 1)
    else:
        matchingTaus_mask = matching_L1Taus_obj(L1Taus_Tau, Taus_Tau)
        # at least 1 Tau should match L1Tau
        evt_Taumask_matching = (ak.sum(matchingTaus_mask, axis=-1) >= 1)
        
        matchingMuons_mask = matching_L1Muons_obj(L1Muons_Muon, Selected_Muons)
        evt_Muonmask_matching = (ak.sum(matchingMuons_mask, axis=-1) >= 1)
        #overlap_mask = overlap_mask1 & overlap_mask2 

    #MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching & overlap_mask
    MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching 
    if is_gen: 
        return MuTau_evt_mask, matchingGentaus_mask, matchingGenMuons_mask
    else:
        return MuTau_evt_mask, matchingTaus_mask, matchingMuons_mask

# ------------------------------ Common functions for Mutau path ---------------------------------------------------------------

def L1_Mu18er2p1_selection(events):
    L1_Mu18er2p1 = (events['L1Muon_pt'].compute() >= 18) & (events['L1Muon_eta'].compute() <= 2.131) & (events['L1Muon_eta'].compute() >= -2.131)
    return L1_Mu18er2p1

def L1_Tau24er2p1_OR_Tau26er2p1_selection(events):
    L1_Tau24er2p1_OR_Tau26er2p1 = ((events['L1Tau_pt'].compute() >=24) | (events['L1Tau_pt'].compute() >= 26)) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131)
    return L1_Tau24er2p1_OR_Tau26er2p1

def Denominator_Selection_MuTau(GenLepton):
    # return mask for event passing minimal Gen requirements for MuTau HLT 
    mask_GenTau = (GenLepton['kind'] == 5)
    mask_GenMuon = (GenLepton['kind'] == 4)
    ev_mask = (ak.sum(mask_GenTau, axis=-1) == 1) &  (ak.sum(mask_GenMuon, axis=-1) == 1) # 1 Gen tau and one Gen Muon should pass this requirements
    return ev_mask

class MuTauDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)
    
    # ------------------------------ functions to Compute Rate ---------------------------------------------------------------

    def get_Nnum_Nden_MuTauDeepNet(self):
        print(f'Computing rate for HLT_IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        MuTau_evt_mask, matchingTaus_mask, matchingMuons_mask = evt_sel_HLT_IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1(events, is_gen = False)
        N_num = len(events[MuTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_MuTauPNet(self, par):
        print(f'Computing Rate for MuTau path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")
        MuTau_evt_mask, matchingJets_mask, matchingMuons_mask = evt_sel_MuTau(events, par, is_gen = False)
        N_num = len(events[MuTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num
    
    def get_Nnum_Nden_MuTauDeepNet_nPV(self, nPV):
        print(f'Computing rate for HLT_IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1 for nPV = {nPV}:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        events = events[events['nPFPrimaryVertex'].compute() == nPV]
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        MuTau_evt_mask, matchingTaus_mask, matchingMuons_mask = evt_sel_HLT_IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1(events, is_gen = False)
        N_num = len(events[MuTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_MuTauPNet_nPV(self, par, nPV):
        print(f'Computing Rate for MuTau path with param: {par} and for nPV = {nPV}:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        events = events[events['nPFPrimaryVertex'].compute() == nPV]
        N_den = len(events)
        print(f"Number of events in denominator: {N_den}")

        MuTau_evt_mask, matchingJets_mask, matchingMuons_mask = evt_sel_MuTau(events, par, is_gen = False)
        N_num = len(events[MuTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

# ------------------------------ functions for ComputeEfficiency ---------------------------------------------------------------

    def save_Event_Nden_eff_MuTau(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for diTau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask = Denominator_Selection_MuTau(GenLepton)
        print(f"Number of events with exactly 1 hadronic Tau and one Muon (kind=TauDecayedToHadrons and TauDecayedTo Muon): {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def produceRoot_MuTauDeepNet(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]
    
        GenMuon_mask = GenMuon_selection(events)
        GenMuons = get_GenMuons(events)
        Muon_Den  = GenMuons[GenMuon_mask]

        mask_den_selection = ( ak.num(Tau_Den['pt']) >=1 ) & ( ak.num(Muon_Den['pt']) >=1 )
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        MuTau_evt_mask, matchingGentaus_mask, matchingMuons_mask = evt_sel_HLT_IsoMu20_eta2p1_PNetTauhPFJetPt30_Loose_eta2p3_CrossL1(events, is_gen = True)
        Tau_Num = (Tau_Den[matchingGentaus_mask])[MuTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num = events[MuTau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return

    def produceRoot_MuTauPNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]
    
        GenMuon_mask = GenMuon_selection(events)
        GenMuons = get_GenMuons(events)
        Muon_Den  = GenMuons[GenMuon_mask]

        mask_den_selection = ( ak.num(Tau_Den['pt']) >=1 ) & ( ak.num(Muon_Den['pt']) >=1 )
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]
        
        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        MuTau_evt_mask, matchingGentaus_mask, matchingMuons_mask = evt_sel_MuTau(events, par, is_gen = True)

        Tau_Num = (Tau_Den[matchingGentaus_mask])[MuTau_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")

        events_Num = events[MuTau_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, out_file)
        return
    
    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_MuTauPNet(self, par):

        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()

        L1_Mu_mask = L1_Mu18er2p1_selection(events) 
        L1_Tau_mask = L1_Tau24er2p1_OR_Tau26er2p1_selection(events)
        Tau_mask = Jet_selection_Tau(events, par, apply_PNET_WP = False)
        GenTau_mask = hGenTau_selection(events)
        GenMuon_mask = GenMuon_selection(events)

        Muon_mask = Muon_selection(events)
        
        # at least 1 L1tau/ recoJet should pass
        MuTau_evt_mask = (ak.sum(L1_Mu_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Muon_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)
        
        L1Taus = get_L1Taus(events)
        L1Taus_Tau = L1Taus[L1_Tau_mask]
        
        L1Muons = get_L1Muons(events)
        L1Muons_Muon = L1Muons[L1_Mu_mask]
        
        Jets = get_Jets(events)
        Jets_Tau = Jets[Tau_mask]

        Muons = get_Muons(events)
        Selected_Muons = Muons[Muon_mask]

        GenTaus = get_GenTaus(events)
        GenTaus_Sel = GenTaus[GenTau_mask]
        
        GenMuons = get_GenMuons(events)
        GenMuons_Muon  = GenMuons[GenMuon_mask]
        
        #matching
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Jets_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1)  

        matchingGenMuons_mask = matching_GenMuons(L1Muons_Muon, Selected_Muons, GenMuons_Muon)
        evt_Muonmask_matching = (ak.sum(matchingGenMuons_mask, axis=-1) >= 1)
        
        MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching

        N_den = len(events[MuTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        Tau_mask = Jet_selection_Tau(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        MuTau_evt_mask = (ak.sum(L1_Mu_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Muon_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        #matching
        Jets_Tau = Jets[Tau_mask]
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Jets_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1) 
     
        MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching

        N_num = len(events[MuTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def ComputeEffAlgo_MuTauDeepNet(self):


        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()

        L1_Mu_mask = L1_Mu18er2p1_selection(events) 
        L1_Tau_mask = L1_Tau24er2p1_OR_Tau26er2p1_selection(events)
        Tau_mask = Tau_selection_Tau(events, apply_DeepTau_WP = False)
        GenTau_mask = hGenTau_selection(events)
        GenMuon_mask = GenMuon_selection(events)

        Muon_mask = Muon_selection(events)
        
        # at least 1 L1tau/ recoJet should pass
        MuTau_evt_mask = (ak.sum(L1_Mu_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Muon_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)
        
        L1Taus = get_L1Taus(events)
        L1Taus_Tau = L1Taus[L1_Tau_mask]
        
        L1Muons = get_L1Muons(events)
        L1Muons_Muon = L1Muons[L1_Mu_mask]
        
        Taus = get_Taus(events)
        Taus_Tau = Taus[Tau_mask]

        Muons = get_Muons(events)
        Selected_Muons = Muons[Muon_mask]

        GenTaus = get_GenTaus(events)
        GenTaus_Sel = GenTaus[GenTau_mask]
        
        GenMuons = get_GenMuons(events)
        GenMuons_Muon  = GenMuons[GenMuon_mask]
        
        #matching
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Taus_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1)  

        matchingGenMuons_mask = matching_GenMuons(L1Muons_Muon, Selected_Muons, GenMuons_Muon)
        evt_Muonmask_matching = (ak.sum(matchingGenMuons_mask, axis=-1) >= 1)
        
        MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching

        N_den = len(events[MuTau_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with PNET WP
        Tau_mask = Tau_selection_Tau(events, apply_DeepTau_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        MuTau_evt_mask = (ak.sum(L1_Mu_mask, axis=-1) >= 1) & (ak.sum(L1_Tau_mask, axis=-1) >= 1) & (ak.sum(Tau_mask, axis=-1) >= 1) & (ak.sum(Muon_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        #matching
        Taus_Tau = Taus[Tau_mask]
        matching_GenTaus_mask_Tau = matching_Gentaus(L1Taus_Tau, Taus_Tau, GenTaus_Sel)
        evt_Taumask_matching = (ak.sum(matching_GenTaus_mask_Tau, axis=-1) >= 1) 
     
        MuTau_evt_mask = MuTau_evt_mask & evt_Taumask_matching & evt_Muonmask_matching

        N_num = len(events[MuTau_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num
