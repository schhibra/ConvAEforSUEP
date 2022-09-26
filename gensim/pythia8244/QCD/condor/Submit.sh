#!/bin/bash

workdir="qcd_v1"
jobDir="/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_$workdir"

if [ ! -d $jobDir ]; then
    mkdir -p $jobDir;
fi

if [ ! -d $jobDir/gensim ]; then
    mkdir -p $jobDir/gensim;
fi

cp ../qcd_main $jobDir
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

seed=100
i=0

while [ $i -lt 50 ]; do
        
    ((i = i + 1))
    ((seed = seed + 1))
    echo $seed

    cat run.sh | sed "s?JOBS?${jobDir}?g" | sed "s?SEED?${seed}?g" > dir_$workdir/run_${seed}.sh
    
    cat condor.sub | sed "s?run.sh?run_${seed}.sh?g" > dir_$workdir/condor_${seed}.sub

    cd dir_$workdir
    condor_submit condor_${seed}.sub
    cd -

done
