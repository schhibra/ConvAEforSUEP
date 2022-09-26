#!/bin/bash

#workdir="qcd_v1"
workdir="suep_v1"
jobDir="/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_$workdir/gensim"
srcDir="/eos/cms/store/cmst3/user/schhibra/ML_02042022/jobs_$workdir"

if [ ! -d $jobDir/data ]; then
    mkdir -p $jobDir/data;
fi

if [ ! -d $jobDir/output ]; then
    mkdir -p $jobDir/output;
fi

cp ../scripts/generate-configuration-file.py $jobDir
cp ../hdf5-generator.py $jobDir
#cp ../hdf5-generator_HCALbins.py $jobDir

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

rm datasets.txt
ls $jobDir/*h125*_2*.root | awk -F "/" '{print $11}' > datasets.txt

for dataset in `less datasets.txt`;do

    echo $dataset
    name=`echo $dataset | awk -F "." '{print $1}'`

    cat run.sh | sed "s?JOBS?${jobDir}?g" | sed "s?SRC?${srcDir}?g" | sed "s?INPUT?${dataset}?g" | sed "s?NAME?${name}?g" > dir_$workdir/run_${name}.sh

    cat condor.sub | sed "s?run.sh?run_${name}.sh?g" > dir_$workdir/condor_${name}.sub

    cd dir_$workdir
    condor_submit condor_${name}.sub
    cd -

done
