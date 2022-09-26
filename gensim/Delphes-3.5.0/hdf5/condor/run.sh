#!/bin/bash

cd JOBS

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh
python3 generate-configuration-file.py SRC -i NAME.root -s data/NAME.json

python3 hdf5-generator.py JOBS -c data/NAME.json -s output/NAME -n 10 -v
#python3 hdf5-generator_HCALbins.py JOBS -c data/NAME.json -s output/NAME -n 10 -v

rm data/NAME.json
