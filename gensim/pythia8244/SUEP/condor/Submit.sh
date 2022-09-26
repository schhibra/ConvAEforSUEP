#!/bin/bash

workdir="suep_v1"
jobDir="/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_$workdir"

if [ ! -d $jobDir ]; then
    mkdir -p $jobDir;
fi

if [ ! -d $jobDir/gensim ]; then
    mkdir -p $jobDir/gensim;
fi

cp ../suep_main $jobDir
cp ../decay_cards/decay_darkphoton_hadronic.cmnd $jobDir/decay_darkphoton_hadronic_0p7.cmnd
cp ../../../Delphes-3.5.0/DelphesHepMC2 $jobDir
cp ../../../Delphes-3.5.0/MinBias.pileup $jobDir
cp ../../../Delphes-3.5.0/*.pcm $jobDir
cp ../../../Delphes-3.5.0/cards/delphes_card_CMS_PileUp_ML.tcl $jobDir

if [ ! -d dir_$workdir ]; then
    mkdir -p dir_$workdir;
fi

if [ ! -d dir_$workdir/out ]; then
    mkdir -p dir_$workdir/out;
fi

if [ ! -d dir_$workdir/err ]; then
    mkdir -p dir_$workdir/err;
fi

if [ ! -d dir_$workdir/log ]; then
    mkdir -p dir_$workdir/log;
fi

homedir=$PWD

for m_H in 125; do ### 400 700 1000; do

    echo "m_H" $m_H

    m_phi=2
    temp=2
    m_dp=0p7

    seed=200
    i=0   
    while [ $i -lt 10 ]; do
        
	((i = i + 1))
	((seed = seed + 1))
	echo $seed
	
	cat run.sh | sed "s?JOBS?${jobDir}?g" | sed "s?HM?${m_H}?g" | sed "s?PHIM?${m_phi}?g" | sed "s?TEMP?${temp}?g" | sed "s?DPM?${m_dp}?g" | sed "s?SEED?${seed}?g" > dir_$workdir/run_${m_H}_${m_phi}_${temp}_${m_dp}_${seed}.sh
	
	cat condor.sub | sed "s?run.sh?run_${m_H}_${m_phi}_${temp}_${m_dp}_${seed}.sh?g" > dir_$workdir/condor_${m_H}_${m_phi}_${temp}_${m_dp}_${seed}.sub
	
	cd dir_$workdir
	condor_submit condor_${m_H}_${m_phi}_${temp}_${m_dp}_${seed}.sub
	cd -
	
    done
 done
