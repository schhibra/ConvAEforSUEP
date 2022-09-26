import argparse
import h5py
import numpy as np
import os
import simplejson as json
import uproot3 as uproot
import warnings
import matplotlib.pyplot as plt

from tqdm import tqdm

import ROOT

class IsReadableDir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(
                    '{0} is not a valid path'.format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                    '{0} is not a readable directory'.format(prospective_dir))

class PhysicsConstants():

    def __init__(self, example_file):

        # Define jet constants
        self.delta_r = .4
        self.delphes = uproot.open(example_file)['Delphes']
        self.min_eta = -2.52
        self.max_eta = 2.52
        self.min_pt = {'q': 30., 'q': 30.}
        self.settings = {'g': {'id': 0, 'pid': [21]       },
                         'q': {'id': 1, 'pid': [1,2,3,4,5]}}
        
    def get_edges_ecal(self, x, sample_events=1000):

        all_edges = np.array([], dtype=np.float16)
        edge_arr = self.delphes['ecalTower']['ecalTower.Edges[4]'].array()

        for i in range(sample_events):
            all_edges = np.append(all_edges, edge_arr[i][:, [x, x+1]])
            all_edges = np.unique(all_edges)
        
        if x == 0:
            all_edges = all_edges[(all_edges > self.min_eta) &
                                  (all_edges < self.max_eta)]

        return all_edges

    def get_edges_hcal(self, x, sample_events=1000):

        all_edges = np.array([], dtype=np.float16)
        edge_arr = self.delphes['hcalTower']['hcalTower.Edges[4]'].array()

        for i in range(sample_events):
            all_edges = np.append(all_edges, edge_arr[i][:, [x, x+1]])
            all_edges = np.unique(all_edges)
        
        if x == 0:
            all_edges = all_edges[(all_edges > -2.54) &
                                  (all_edges < +2.54)]

        return all_edges


class HDF5Generator:

    def __init__(self, hdf5_dataset_path, hdf5_dataset_size, files_details,
                 verbose=True):

        self.constants = PhysicsConstants(list(files_details[0])[0])
        self.edges_eta = self.constants.get_edges_ecal(0)
        self.edges_phi = self.constants.get_edges_ecal(2)

        self.edges_eta_hcal = self.constants.get_edges_hcal(0)
        self.edges_phi_hcal = self.constants.get_edges_hcal(2)

        #print(self.edges_eta)
        #print(self.edges_eta_hcal)

        #print(self.edges_phi)
        #print(self.edges_phi_hcal)

        self.hdf5_dataset_path = hdf5_dataset_path
        self.hdf5_dataset_size = hdf5_dataset_size
        self.files_details = files_details

        self.verbose = verbose

    def create_hdf5_dataset(self, progress_bar):

        # Create the HDF5 file.
        hdf5_dataset = h5py.File(self.hdf5_dataset_path, 'w')

        hdf5_ImageTrk_PUcorr = hdf5_dataset.create_dataset(
                name='ImageTrk_PUcorr',
                shape=(self.hdf5_dataset_size, 18, 18),
                maxshape=(None),
                compression="gzip",
                dtype=np.float16)

        hdf5_ImageECAL = hdf5_dataset.create_dataset(
                name='ImageECAL',
                shape=(self.hdf5_dataset_size, 18, 18),
                maxshape=(None),
                compression="gzip",
                dtype=np.float16)

        hdf5_ImageHCAL = hdf5_dataset.create_dataset(
                name='ImageHCAL',
                shape=(self.hdf5_dataset_size, 18, 18),
                maxshape=(None),
                compression="gzip",
                dtype=np.float16)

        i = 0

        for file_details in self.files_details:
            file_path = next(iter(file_details.keys()))

            events = file_details[file_path]

            file = uproot.open(file_path)

            ScalarHT            = file['Delphes']['ScalarHT']
            ScalarHT_HT         = ScalarHT['ScalarHT.HT'].array()

            Track_PUcorr               = file['Delphes']['Track_PUcorr']
            Track_Eta_PUcorr_full      = Track_PUcorr['Track_PUcorr.EtaOuter'].array()
            Track_Phi_PUcorr_full      = Track_PUcorr['Track_PUcorr.PhiOuter'].array()
            Track_PT_PUcorr_full       = Track_PUcorr['Track_PUcorr.PT'].array()

            ECALTower           = file['Delphes']['ecalTower']
            ECALTower_Eta_full  = ECALTower['ecalTower.Eta'].array()
            ECALTower_Phi_full  = ECALTower['ecalTower.Phi'].array()
            ECALTower_ET_full   = ECALTower['ecalTower.ET'].array()

            HCALTower           = file['Delphes']['hcalTower']
            HCALTower_Eta_full  = HCALTower['hcalTower.Eta'].array()
            HCALTower_Phi_full  = HCALTower['hcalTower.Phi'].array()
            HCALTower_ET_full   = HCALTower['hcalTower.ET'].array()

            for event_number in np.arange(events[0], events[1], dtype=int):

                if self.verbose:
                    progress_bar.update(1)

                if (ScalarHT_HT[event_number][0] <= 500): continue
                #print(ScalarHT_HT[event_number][0])
 
                # Get Track PUcorr
                e = Track_Eta_PUcorr_full[event_number]
                p = Track_Phi_PUcorr_full[event_number]
                v = Track_PT_PUcorr_full[event_number]
                mask = ((e > self.edges_eta[0]) & (e < self.edges_eta[-1]))
                e, p, v = e[mask], p[mask], v[mask]
                h, _, _= np.histogram2d(e, p, bins=[18, 18],
                                        range=[[self.edges_eta[0], self.edges_eta[-1]],[self.edges_phi[0], self.edges_phi[-1]]],
                                        weights=v)
                hdf5_ImageTrk_PUcorr[i] = h

                # Get ECAL Tower
                e = ECALTower_Eta_full[event_number]
                p = ECALTower_Phi_full[event_number]
                v = ECALTower_ET_full[event_number]
                mask = ((e > self.edges_eta[0]) & (e < self.edges_eta[-1]))
                e, p, v = e[mask], p[mask], v[mask]
                h, _, _= np.histogram2d(e, p, bins=[18, 18], 
                                        range=[[self.edges_eta[0], self.edges_eta[-1]],[self.edges_phi[0], self.edges_phi[-1]]], 
                                        weights=v)
                hdf5_ImageECAL[i] = h

                # Get HCAL Tower
                e = HCALTower_Eta_full[event_number]
                p = HCALTower_Phi_full[event_number]
                v = HCALTower_ET_full[event_number]

                #print(self.edges_eta[0], self.edges_eta[-1])
                #print(self.edges_eta_hcal[0], self.edges_eta_hcal[-1])
                mask = ((e > self.edges_eta_hcal[0]) & (e < self.edges_eta_hcal[-1]))
                e, p, v = e[mask], p[mask], v[mask]
                for j in range (0, e.size): 
                    if (e[j] < self.edges_eta[0]): e[j] = -2.5
                    if (e[j] > self.edges_eta[-1]): e[j] = +2.5 
                h, _, _= np.histogram2d(e, p, bins=[18, 18], 
                                        range=[[self.edges_eta[0], self.edges_eta[-1]],[self.edges_phi[0], self.edges_phi[-1]]], 
                                        weights=v)
                hdf5_ImageHCAL[i] = h

                i=i+1

        hdf5_ImageTrk_PUcorr.resize((i, hdf5_ImageTrk_PUcorr.shape[1], hdf5_ImageTrk_PUcorr.shape[2]))
        hdf5_ImageECAL.resize((i, hdf5_ImageECAL.shape[1], hdf5_ImageECAL.shape[2]))
        hdf5_ImageHCAL.resize((i, hdf5_ImageHCAL.shape[1], hdf5_ImageHCAL.shape[2]))
        hdf5_dataset.close()

class Utils():

    def parse_config(self, folder, nofiles, config_path):

        # Laod configuration
        with open(config_path, 'r') as f:
            config = json.loads(f.read())

        # Total number of events
        total = config[folder]['events']
        files_list = list(config[folder]['files'])
        files_details, files_batch = [], []
        gtotal, fid, event_min_next = 0, 0, 0
        batch_id = 1
        batch_size = total / float(nofiles)
        jtype = folder.split('/')[-1]

        while gtotal < total:

            file = files_list[fid]

            # Set FROM and TO indexes
            event_min = event_min_next
            event_max = config[folder]['files'][file]

            # Fix nominal target of events
            gtotal_target = gtotal + event_max - event_min

            # Save filenames with indexes
            # Fraction of the file
            if batch_id*batch_size <= gtotal_target:
                max_in_this_batch = int(batch_id*batch_size)
                event_max = event_max - (gtotal_target - max_in_this_batch)
                event_min_next = event_max

                # Prevent saving files with no events
                if event_max != event_min:
                    files_batch.append({file: (event_min, event_max)})

                # Push to file details
                files_details.append(files_batch)
                files_batch = []
                batch_id = batch_id + 1
            # Otherwise: full file
            else:
                files_batch.append({file: (event_min, event_max)})
                event_min_next = 0
                fid += 1

            gtotal = gtotal + event_max - event_min

        return files_details, batch_size, gtotal, jtype


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Convert root file data to h5')
    parser.add_argument('src_folder', type=str, help='Folder to convert')
    parser.add_argument('-n', '--number-of-files', type=int, default=10,
                        help='Target number of output files', dest='nfiles')

    #parser.add_argument('-o', '--save-path', type=str, action=IsReadableDir,
    #                    default='.', help='Output directory', dest='save_dir')

    parser.add_argument('-s', '--save-file', type=str,
                        default='.', help='Output file', dest='save_file')

    parser.add_argument('-c', '--config', type=str, 
                        default='.', help='Configuration file path', dest='config')

    parser.add_argument('-v', '--verbose', action="store_true",
                        help='Output verbosity')
    args = parser.parse_args()

    utils = Utils()

    files_details, batch_size, total_events, jtype = utils.parse_config(args.src_folder, args.nfiles, args.config)

    pb = None
    if args.verbose:
        pb = tqdm(total=total_events, desc=('Processing %s' % jtype))

    for index, file_dict in enumerate(files_details):

        dataset_size = int((index+1)*batch_size)-int((index)*batch_size)
        generator = HDF5Generator(
            #hdf5_dataset_path='{0}/{1}_{2}.h5'.format(args.save_dir, jtype, index),
            hdf5_dataset_path='{0}_{1}.h5'.format(args.save_file, index),
            #hdf5_dataset_path=args.save_file,
            hdf5_dataset_size=dataset_size,
            files_details=file_dict,
            verbose=args.verbose)
        generator.create_hdf5_dataset(pb)

    if args.verbose:
        pb.close()
