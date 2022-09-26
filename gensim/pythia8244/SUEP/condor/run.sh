#!/bin/bash

cd JOBS

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh
./suep_main HM PHIM TEMP decay_darkphoton_hadronic_DPM.cmnd suep_gen_hHM_phiPHIM_dpDPM_dtTEMP_SEED.file SEED

echo "======= gen step done ======="

rm gensim/suep_gensim_hHM_phiPHIM_dpDPM_dtTEMP_SEED.root
./DelphesHepMC2 delphes_card_CMS_PileUp_ML.tcl gensim/suep_gensim_hHM_phiPHIM_dpDPM_dtTEMP_SEED.root suep_gen_hHM_phiPHIM_dpDPM_dtTEMP_SEED.file

echo "======= sim step done ======="
