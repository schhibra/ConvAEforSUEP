source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc8-opt/setup.sh

./configure --with-lhapdf6=/cvmfs/sft.cern.ch/lcg/releases/MCGenerators/lhapdf/6.3.0-3c8e0/x86_64-centos7-gcc8-opt/ --with-fastjet3=/cvmfs/sft.cern.ch/lcg/releases/fastjet/3.3.4-0d9d5/x86_64-centos7-gcc8-opt/ --with-hepmc2=/cvmfs/cms.cern.ch/slc7_amd64_gcc820/external/hepmc/2.06.07/

make -j 8

cd SUEP/
make suep_main
cd -

cd QCD
make qcd_main
