// This is an example main pythia script to generate to generate a dark sector shower in a strongly coupled, quasi-conformal
// hidden valley, often referred to as "soft unclustered energy patterns (SUEP)" or "softbomb" events.
// The code for the dark shower itself is in suep_shower.cxx

// The algorithm relies on arXiv:1305.5226. See arXiv:1612.00850 for a description of the model. 
// Please cite both papers when using this code.

// Written by Simon Knapen on 12/22/2019

// pythia headers
#include <iostream>
#include <math.h>
#include <boost/math/tools/roots.hpp>
#include <boost/bind/bind.hpp>
//#include <boost/bind.hpp>
#include<string>
#include<stdio.h>
#include<stdlib.h>

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC2.h"

#include "DecayToSUEP.h"

using namespace Pythia8;

template <typename T> string tostr(const T& t) { 
   ostringstream os; 
   os<<t; 
   return os.str(); 
} 

int main(int argc, char *argv[]) {
     
   // read model parameters from the command line
  if(!(argc==7)){
    std::cout << "I need the following arguments: M m T decaycard outputfilename randomseed\n";
    std::cout << "with\n";
    std::cout << "M: mass of heavy scalar\n";
    std::cout << "m: mass of dark mesons\n";
    std::cout << "T: Temperature parameter\n";
    std::cout << "decaycard: filename of the decay card\n";
    std::cout << "outputfilename: filename where events will be written\n";
    std::cout << "randomseed: an integer, specifying the random seed\n";
    return 0;
  }
     

  // model parameters and settings
  float mh, mX,T;
  string seed, filename, cardfilename;    
  mh=atof(argv[1]);
  mX=atof(argv[2]);
  T=atof(argv[3]);
  cardfilename=tostr(argv[4]);
  filename=tostr(argv[5]);
  seed=tostr(argv[6]);    
  
  // number of events    
  int Nevents=10000;    
    
  // Interface for conversion from Pythia8::Event to HepMC event.
  HepMC::Pythia8ToHepMC ToHepMC;

  // Specify file where HepMC events will be stored.
  HepMC::IO_GenEvent ascii_io(filename, std::ios::out);

  // pythia object
  Pythia pythia;
  
  //Settings for the Pythia object
  pythia.readString("Beams:eCM = 13600.");
  pythia.readString("HiggsSM:all = off"); //All SM major Higgs production modes
  pythia.readString("HiggsSM:gg2H = on"); //only gluon fussion SM Higgs production mode
  pythia.readString("25:m0 = "+tostr(mh)); // set the mass of "Higgs" scalar
  pythia.readString("25:0:onMode=1");
  pythia.readString("25:1:onMode=0");
  pythia.readString("25:2:onMode=0");
  pythia.readString("25:3:onMode=0");
  pythia.readString("25:4:onMode=0");
  pythia.readString("25:5:onMode=0");
  pythia.readString("25:6:onMode=0");
  pythia.readString("25:7:onMode=0");
  pythia.readString("25:8:onMode=0");
  pythia.readString("25:9:onMode=0");
  pythia.readString("25:10:onMode=0");
  pythia.readString("25:11:onMode=0");
  pythia.readString("25:12:onMode=0");
  pythia.readString("25:13:onMode=0");
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = "+seed); 
  pythia.readString("Next:numberShowEvent = 0");
  
  // for debugging / testing purposes only
  pythia.readString("PartonLevel:ISR = on");
  
  // define the dark meson
  pythia.readString("999999:all = GeneralResonance void 0 0 0 "+tostr(mX)+" 0.001 0.0 0.0 0.0");
  // this card had the dark photon branching ratios
  pythia.readFile(cardfilename);
  pythia.readString("Check:event = off");
  pythia.readString("Next:numberShowEvent = 0");
  DecayToSUEP *suep_hook = new DecayToSUEP();
  suep_hook->m_pdgId = 25;
  suep_hook->m_mass = mh;
  suep_hook->m_darkMesonMass = mX;
  suep_hook->m_darkTemperature = T;
  pythia.setUserHooksPtr(suep_hook);

  pythia.init();
  pythia.settings.listChanged();
   
  // Shortcuts
  Event& event = pythia.event;
  
  // Begin event loop. Generate event. Skip if error.
  int iEvent=0;
  while (iEvent < Nevents) {
    // Run the event generation

    if (!pythia.next()) {
      cout << " Event generation aborted prematurely, owing to error!\n";
      break;
    }
    
    // Print out a few events
    if (iEvent<1){  
        event.list();
    }
    
    // Construct new empty HepMC event and fill it.
    // Units will be as chosen for HepMC build; but can be changed
    // by arguments, e.g. GenEvt( HepMC::Units::GEV, HepMC::Units::MM)
    HepMC::GenEvent* hepmcevt = new HepMC::GenEvent(HepMC::Units::GEV, HepMC::Units::MM);
    ToHepMC.fill_next_event( pythia, hepmcevt );

    // Write the HepMC event to file. Done with it.
    ascii_io << hepmcevt;
    delete hepmcevt;
    
    iEvent++;

  // End of event loop.
  }
  // print the cross sections etc    
  pythia.stat();

  delete suep_hook;

  // Done.
  return 0;
}
