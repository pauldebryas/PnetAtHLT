import json
import os
import pandas as pd

#p = "/eos/home-j/jleonhol/www/PNetAtHLT/2024_v1/PNetAtHLT/Optimisation/result/"

#p = "/eos/home-j/jleonhol/www/PNetAtHLT/2024_v1/PNetAtHLT/OptimisationDiTau/result_new/"

p = "/afs/cern.ch/work/s/skeshri/TauHLT/Braden/Forked/TauTriggerDev/Optimisation/result/"

#p = "/eos/home-j/jleonhol/www/PNetAtHLT/2024_v1/PNetAtHLT/OptimisationDiTauSingleTau/result/"

# p = "/eos/home-j/jleonhol/www/PNetAtHLT/2024_v1/PNetAtHLT/OptimisationDiTauSingleTauDenDouble/result/"

l = []
for f in os.listdir(p):
    with open(p + f) as f:
        l.append(json.load(f))

df = pd.DataFrame(l)

#df.to_pickle("results_ditaujet.pickle")
df.to_pickle("results_mutau.pickle")
#df.to_pickle("results_ditau_singletau.pickle")
# df.to_pickle("results_ditau_singletau_doubleden.pickle")
