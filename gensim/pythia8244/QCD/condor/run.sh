#!/bin/bash

cd JOBS

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh
./qcd_main qcd_gen_SEED.file SEED

echo "======= gen step done ======="

rm gensim/qcd_gensim_SEED.root
./DelphesHepMC2 delphes_card_CMS_PileUp_ML.tcl gensim/qcd_gensim_SEED.root qcd_gen_SEED.file

echo "======= sim step done ======="
