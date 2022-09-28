## Convolutional AEs for anomaly (non-QCD events) detection

### Structure of this repository:
   * `pythia8244_with_sueps.tgz`: pythia setup for QCD and SUEP events generation
   * `gensim/pythia8244`: QCD and SUEP events generation and DELPHES simulation (DELPHES setup is in gensim/Delphes-3.5.0)
   * `gensim/Delphes-3.5.0`: HDF5 generator (provides input files)     
   * `models`: different AE models
   * `scripts`: different loss functions plus plotting scripts
   * `notebooks`: different notebooks for testing and plotting

### Important links:
   * Model architecture visulaisation: http://alexlenail.me/NN-SVG/AlexNet.html
   * Python and HDF5: https://twiki.cern.ch/twiki/pub/Sandbox/JaredDavidLittleSandbox/PythonandHDF5.pdf
   * Nice paper (AEs for semivisible jet detection): https://arxiv.org/pdf/2112.02864.pdf
