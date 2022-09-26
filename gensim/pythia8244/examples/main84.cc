// main41.cc is a part of the PYTHIA event generator.
// Copyright (C) 2019 Torbjorn Sjostrand.
// PYTHIA is licenced under the GNU GPL v2 or later, see COPYING for details.
// Please respect the MCnet Guidelines, see GUIDELINES for details.

// Author: Mikhail Kirsanov, Mikhail.Kirsanov@cern.ch, based on main01.cc.
// This program illustrates how HepMC can be interfaced to Pythia8.
// It studies the charged multiplicity distribution at the LHC.
// HepMC events are output to the hepmcout41.dat file.

// WARNING: typically one needs 25 MB/100 events at the LHC.
// Therefore large event samples may be impractical.

#include "Pythia8/Pythia.h"
#ifndef HEPMC2
#include "Pythia8Plugins/HepMC3.h"
#else
#include "Pythia8Plugins/HepMC2.h"
#endif
#include <string>
#include <vector>
#include <sstream>

using namespace Pythia8;

std::string toString(int i){
  std::stringstream ss;
  ss << i;
  return ss.str();
}


int main(int argc, char *argv[]) {

  if (argc<3) {
    std::cout<<"Usage: ./main43 Mode Seed"<<std::endl;
    std::cout<<"  Mode = PileUp, TTBar, WJets, VBFHbb, DiJet"<<std::endl;
    return false;
  }
  std::string m_mode = argv[1];
  int seed = atoi(argv[2]);
  //cout << seed << endl;
  std::string m_OutputName = "results/hepmcout_result.data";
  std::vector<std::string> vecPythiaCommands;

  string pdfSet = "LHAPDF6:NNPDF30_nnlo_as_0118";
 
  //vecPythiaCommands.push_back("PDF:pSet = " + pdfSet);
  vecPythiaCommands.push_back("Beams:eCM = 14000.");
  vecPythiaCommands.push_back("Tune:pp = 5");
  vecPythiaCommands.push_back("Random:setSeed = on");
  vecPythiaCommands.push_back("Random:seed = "+toString(seed));
  if (m_mode=="PileUp") {
    m_OutputName = "results/hepmcout_SoftQCD_"+toString(seed)+".data";
    vecPythiaCommands.push_back("SoftQCD:all = on");
  }
  if (m_mode=="DiJet") {
    m_OutputName = "results/hepmcout_DiJet_"+toString(seed)+".data";
    vecPythiaCommands.push_back("HardQCD:all = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 100");
  }
  if (m_mode=="WJets") {
    m_OutputName = "results/hepmcout_WJets_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakSingleBoson:ffbar2W = on");
    vecPythiaCommands.push_back("24:onMode = off");
    vecPythiaCommands.push_back("24:onIfAny = 11 13");
    vecPythiaCommands.push_back("24:onIfAny = -11 -13");
  }
  if (m_mode=="WJets_HighPT") {
    m_OutputName = "results/hepmcout_WJets_HighPT_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakBosonAndParton:qqbar2Wg = on");
    vecPythiaCommands.push_back("WeakBosonAndParton:qg2Wq = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 100");
    vecPythiaCommands.push_back("24:onMode = off");
    vecPythiaCommands.push_back("24:onIfAny = 11 13");
    vecPythiaCommands.push_back("24:onIfAny = -11 -13");
  }
  if (m_mode=="ZJets_HighPT") {
    m_OutputName = "results/hepmcout_ZJets_HighPT_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakZ0:gmZmode = 2");
    vecPythiaCommands.push_back("WeakBosonAndParton:qqbar2gmZg = on");
    vecPythiaCommands.push_back("WeakBosonAndParton:qg2gmZq = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 20");
    vecPythiaCommands.push_back("23:onMode = off");
    vecPythiaCommands.push_back("23:onIfAny = 11 13");
    vecPythiaCommands.push_back("-23:onIfAny = -11 -13");
  }
  if (m_mode=="ZvvJets_HighPT") {
    m_OutputName = "results/hepmcout_ZvvJets_HighPT_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakZ0:gmZmode = 2");
    vecPythiaCommands.push_back("WeakBosonAndParton:qqbar2gmZg = on");
    vecPythiaCommands.push_back("WeakBosonAndParton:qg2gmZq = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 20");
    vecPythiaCommands.push_back("23:onMode = off");
    vecPythiaCommands.push_back("23:onIfAny = 12 14 16");
    vecPythiaCommands.push_back("-23:onIfAny = -12 -14 16");
  }
  if (m_mode=="ZvvJets_MediumHighPT") {
    m_OutputName = "results/hepmcout_ZvvJets_MediumHighPT_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakZ0:gmZmode = 2");
    vecPythiaCommands.push_back("WeakBosonAndParton:qqbar2gmZg = on");
    vecPythiaCommands.push_back("WeakBosonAndParton:qg2gmZq = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 100");
    vecPythiaCommands.push_back("23:onMode = off");
    vecPythiaCommands.push_back("23:onIfAny = 12 14 16");
    vecPythiaCommands.push_back("-23:onIfAny = -12 -14 16");
  }
  if (m_mode=="ZvvJets_VeryHighPT") {
    m_OutputName = "results/hepmcout_ZvvJets_VeryHighPT_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakZ0:gmZmode = 2");
    vecPythiaCommands.push_back("WeakBosonAndParton:qqbar2gmZg = on");
    vecPythiaCommands.push_back("WeakBosonAndParton:qg2gmZq = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 150");
    vecPythiaCommands.push_back("23:onMode = off");
    vecPythiaCommands.push_back("23:onIfAny = 12 14 16");
    vecPythiaCommands.push_back("-23:onIfAny = -12 -14 16");
  }
  if (m_mode=="ZvvJets_LowPT") {
    m_OutputName = "results/hepmcout_ZvvJets_LowPT_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakZ0:gmZmode = 2");
    vecPythiaCommands.push_back("WeakBosonAndParton:qqbar2gmZg = on");
    vecPythiaCommands.push_back("WeakBosonAndParton:qg2gmZq = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 10");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMax = 30");
    vecPythiaCommands.push_back("23:onMode = off");
    vecPythiaCommands.push_back("23:onIfAny = 12 14 16");
    vecPythiaCommands.push_back("-23:onIfAny = -12 -14 16");
  }
  if (m_mode=="WJets_Max") {
    m_OutputName = "/data/t3home000/bmaier/hepmc/results/hepmcout_WJets_"+toString(seed)+".data";
    vecPythiaCommands.push_back("WeakSingleBoson:ffbar2W = on");
    vecPythiaCommands.push_back("24:onMode = off");
    vecPythiaCommands.push_back("24:onIfAny = 11 13");
    vecPythiaCommands.push_back("24:onIfAny = -11 -13");
  }
  if (m_mode=="TTBar") {
    m_OutputName = "results/hepmcout_TTBar_"+toString(seed)+".data";
    vecPythiaCommands.push_back("Top:all = on");
    vecPythiaCommands.push_back("24:onMode = off");
    vecPythiaCommands.push_back("24:onIfAny = 11 13");
    vecPythiaCommands.push_back("-24:onIfAny = -11 -13");
  }
  if (m_mode=="VBFHcc") {
    m_OutputName = "results/hepmcout_VBFHcc_"+toString(seed)+".data";
    vecPythiaCommands.push_back("HiggsSM:ff2Hff(t:WW) = on");
    vecPythiaCommands.push_back("25:onMode = off");
    vecPythiaCommands.push_back("25:onIfAny = 4 -4");
  }
  if (m_mode=="ggHbb") {
    m_OutputName = "results/hepmcout_ggHbb_"+toString(seed)+".data";
    vecPythiaCommands.push_back("HiggsSM:gg2Hg(l:t) = on");
    vecPythiaCommands.push_back("PhaseSpace:pTHatMin = 250");
    vecPythiaCommands.push_back("25:onMode = off");
    vecPythiaCommands.push_back("25:onIfAny = 5 -5");
  }

  // Interface for conversion from Pythia8::Event to HepMC event.
  HepMC3::Pythia8ToHepMC3 ToHepMC;

  // Specify file where HepMC events will be stored.
  HepMC3::WriterAscii ascii_io(m_OutputName.c_str());

  // Generator. Process selection. LHC initialization. Histogram.
  Pythia pythia;
  for (unsigned int n=0; n<vecPythiaCommands.size(); n++) {
    cout << vecPythiaCommands[n] << endl;
    pythia.readString(vecPythiaCommands[n]);
  }
  pythia.init();
  Hist mult("charged multiplicity", 100, -0.5, 799.5);

  // Begin event loop. Generate event. Skip if error.
  std::cout<<" Event loop start"<<std::endl;
  int counter = 0;
  std::cout<<counter<<std::endl;
  for (int iEvent = 0; iEvent < 100; ++iEvent) {
    counter+=1;
    if (counter%10000==0)
      std::cout<<counter<<std::endl;
    if (!pythia.next()) continue;
    // Find number of all final charged particles and fill histogram.
    int nCharged = 0;
    for (int i = 0; i < pythia.event.size(); ++i)
      if (pythia.event[i].isFinal() && pythia.event[i].isCharged())
	++nCharged;
    mult.fill( nCharged );

    // Construct new empty HepMC event and fill it.
    // Units will be as chosen for HepMC build; but can be changed
    // by arguments, e.g. GenEvt( HepMC::Units::GEV, HepMC::Units::MM)
    HepMC3::GenEvent hepmcevt;
    ToHepMC.fill_next_event( pythia, &hepmcevt );

    // Write the HepMC event to file. Done with it.
    ascii_io.write_event(hepmcevt);
    //ascii_io << hepmcevt;
    //delete hepmcevt;
    // End of event loop. Statistics. Histogram.
  }

  pythia.stat();
  cout << mult;

  // Done.
  return 0;
}

