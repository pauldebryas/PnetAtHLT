#!/usr/bin/env bash

action() {
    # determine the directory of this file
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

    #export some paths
    export PYTHONPATH="$this_dir:$PYTHONPATH"
    export LAW_HOME="$this_dir/.law"
    export LAW_CONFIG_FILE="$this_dir/law.cfg"
    export RUN_PATH="$this_dir"

    #setup private conda installation for env activation
    PRIVATE_CONDA_INSTALL=/eos/user/s/skeshri/miniconda3
    __conda_setup="$($PRIVATE_CONDA_INSTALL/bin/conda shell.${SHELL##*/} hook)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$PRIVATE_CONDA_INSTALL/etc/profile.d/conda.sh" ]; then
            . "$PRIVATE_CONDA_INSTALL/etc/profile.d/conda.sh"
        else
            export PATH="$PRIVATE_CONDA_INSTALL/bin:$PATH"
        fi
    fi
    unset __conda_setup
    
    conda activate PnetEnv
    #conda activate /afs/cern.ch/work/s/skeshri/.conda/pkgs

    source "$( law completion )" ""
}
action
