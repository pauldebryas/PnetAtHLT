import json
from helpers import compute_ratio_witherr, load_cfg_file
import os

# load config file info
config = load_cfg_file()
number_of_ephemeral_folder = int(config['DATA']['number_of_ephemeral_folder'])
L1A_physics = float(config['RUNINFO']['L1A_physics'])
RefRun = int(config['RUNINFO']['ref_run'])
LumiSectionsRange = [int(config['RUNINFO']['LumiSectionsRange_low']), int(config['RUNINFO']['LumiSectionsRange_up'])]
HLT_name = config['HLT']['HLTname']
HLT_rate = config['HLT']['HLT_rate']
runArea = config['RUNINFO']['Area']

PNet_mode = config['OPT']['PNet_mode']
if PNet_mode == 'false':
    PNetMode = False
    path_result = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}', HLT_name, HLT_name)
else:
    PNetMode = True
    PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]
    path_result = os.path.join(config['DATA']['result_rate'], f'result_{RefRun}', HLT_name, f'PNetTresh_{config["OPT"]["PNet_t1"]}_{config["OPT"]["PNet_t2"]}_{config["OPT"]["PNet_t3"]}')

# compute sum of all Nevents in all the files
N_den = 0
N_num = 0
for i in range(number_of_ephemeral_folder):
    f = open(os.path.join(path_result, f'EphemeralHLTPhysics{i}_Run{runArea}.json'))
    data = json.load(f)
    for file in data.keys():
        N_den += data[file]['N_den']
        N_num += data[file]['N_num']
    f.close()

# compute rate here
eff, err_low, err_up = compute_ratio_witherr(N_num, N_den)

print(f"Total number of events belonging to run {RefRun} and in LumiSections range {LumiSectionsRange}: {N_den}")

if PNetMode:
    print(f"... and passing {HLT_name} conditions without DeepTau, and using PNet with param {PNetparam}: {N_num}")
else:
    print(f"... and passing {HLT_name} conditions: {N_num}")
    print(f"For comparison, {HLT_name} rate in cms oms is {HLT_rate}")

print('Computed rate: ')
print(f"rate : {eff*L1A_physics}")
print(f"rate_up : {err_low*L1A_physics}")
print(f"rate_down : {err_up*L1A_physics}")
