import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset



class DynamicDataset(Dataset):
    '''Date: 23/10/2025, by Jason Jiang

    This dynamic dataset is a upgrade version from SeT-v1, upgraded by Gemini-2.5-Pro to speed up.
    The main difference is this version maintain the data in CPU, not GPU.
    The process of senting the data from CPU to GPU is implemented in training function.

    Date: 3/11/2025 by Jason Jiang

    Adding arguments indices (optinal): array, which is used to calculate train and test dataset.
    
    Date: 6/11/2025 by Jason Jiang
    
    Adding normalization support with optional DataNormalizer
    
    Date: 21/11/2025 by Jason Jiang
    
    Adding FFT pre-calculation for target responses to optimize training speed

    Date: 04/12/2025 by Copilot

    Cloned for grouped splitting logic.
    '''
    
    def __init__(self, 
                 gm_file_path: str, 
                 building_files_dir: str,
                 gm_indices: np.ndarray = None,
                 normalizer=None):
        
        self.gm_file_path = gm_file_path
        self.building_files_paths = [
            os.path.join(building_files_dir, f)
            for f in os.listdir(building_files_dir) if f.endswith('.h5')
        ]
        
        self.gm_file = None
        self.building_files = None 

        with h5py.File(self.gm_file_path, 'r') as f:
            # check the index, use 3474 or splitting one. gm_indices is optional arguments
            # if it is none, then generate a [0,1,2,3...,3474] instead
            if gm_indices is None:
                self.num_gm_samples_total = f['Acc_GMs'].shape[0]
                self.gm_indices = np.arange(self.num_gm_samples_total)
            else:
                self.gm_indices = gm_indices

        self.num_gm_samples = len(self.gm_indices)
        self.num_building_conditions = len(self.building_files_paths)
        self.total_samples = self.num_gm_samples * self.num_building_conditions

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):

        # Lazy loading is more efficent
        if self.gm_file is None:
            self._open_files()

        try:    # try to load
            return self._read_data(idx)
        except OSError as e:
            print(f"Worker {os.getpid()} encountered OSError: {e}. Re-opening files and retrying.")
            # if load fail, then try again
            self._close_files()  # close the file
            self._open_files()   # re-open the file

            return self._read_data(idx)
        
    def _open_files(self):
        self.gm_file = h5py.File(self.gm_file_path, 'r')
        self.building_files = {
            path: h5py.File(path, 'r') for path in self.building_files_paths
        }

    def _close_files(self):
        if self.gm_file:
            self.gm_file.close()
        if self.building_files:
            for f in self.building_files.values():
                f.close()
        self.gm_file = None
        self.building_files = None
    
    # original _getitem_ function
    def _read_data(self, idx):

        # calculate the index in subset (train or validation)
        local_gm_index_in_subset = idx // self.num_building_conditions
        building_index = idx % self.num_building_conditions

        # mapping the subset to the global (e.g. sub[2] -> global[120])
        global_gm_index = self.gm_indices[local_gm_index_in_subset]

        gm_data = self.gm_file['Acc_GMs'][global_gm_index]
        gm_data = torch.from_numpy(gm_data).float().unsqueeze(-1)

        bldg_path = self.building_files_paths[building_index]
        bldg_file_handle = self.building_files[bldg_path]

        acc_floor_response = bldg_file_handle['Acc_Floor_Response'][global_gm_index]
        blg_damage_state = bldg_file_handle['Blg_Damage_State'][global_gm_index][0]

        blg_attributes = {}
        for key, dataset in bldg_file_handle["Blg_Attributes"].items():
            data = dataset[:]
            if dataset.dtype.kind in ['i', 'u']:
                tensor = torch.from_numpy(data).to(torch.int64)
            elif dataset.dtype.kind == 'f':
                tensor = torch.from_numpy(data).to(torch.float32)
            else:
                raise TypeError(f'Unknown data type: {data.dtype}')
            # save in cpu
            blg_attributes[key] = tensor

        acc_floor_response = torch.from_numpy(acc_floor_response).float().unsqueeze(-1)
        blg_damage_state = torch.tensor([blg_damage_state], dtype=torch.long)

        return gm_data, blg_attributes, acc_floor_response, blg_damage_state
