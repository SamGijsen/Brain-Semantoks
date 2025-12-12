import os
import numpy as np
import torch
import h5py
import random
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
import logging
import math
import pandas as pd
import time
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FMRIDataset(Dataset):
    """
    Aggregates timeseries data from one or more HDF5 files for a single modality.
    Handles modality-specific sampling by pre-calculating valid sample definitions.
    Provides one sample per unique subject per epoch call.
    Can optionally load labels associated with each epoch/subject if available in HDF5.
    """
    def __init__(
        self,
        modality_config,
        mode='train',  # 'train' or 'test' or 'probe_train' etc. to select subject ID files
        probe_label_names=None, # List of label names (e.g., ['labels/age_reg']) to load if mode starts with 'probe_'
        use_augmentation=True, # Flag to control augmentation (e.g., random cropping)
        augment_level=0, # 0: no augmentation, 1: light, 2: medium, 3: heavy
        crop_starts='random_mismatch', # 'regular_*' for evenly spaced, 'random_*' or 'conditional_*' for random
        masking_ratio=0.0, # ratio of patches to mask
        masking_type="random",
        temporal_resampling_tr=None,  # Target TR for temporal resampling (e.g. 0.735 for BrainLM)
        **kwargs,  # Accept but ignore legacy params (current_epoch, total_epochs, etc.)
    ):
        super().__init__()
        self.modality_config = modality_config
        self.mode = mode
        self.modality_name = modality_config['name'].lower()
        self.target_signal_length = modality_config['target_signal_length']
        self.number_of_crops = modality_config['number_of_crops']
        self.probe_label_names = probe_label_names if mode.startswith('probe') else None
        self.use_augmentation = use_augmentation
        self.augment_level = augment_level

        # Set augmentation parameters based on augment_level
        if self.use_augmentation and np.sum(self.augment_level) > 0:
            self._set_augment_params()
        else:
            self.spatial_masking_ratio = 0.0
            self.temporal_masking_ratio = 0.0
            self.noise_std = 0.0

        self.crop_starts = crop_starts
        self.patch_length = modality_config['patch_size']

        # Explicit atlas configuration
        self.max_spatial = modality_config.get('max_spatial', False)
        self.min_spatial = modality_config.get('min_spatial', False)
        
        # Build atlas names and network counts from explicit config
        self.atlas_names = []
        self.atlas_rois = []
        self.network_counts = []
        
        for atlas_type in ['schaefer', 'tian', 'buckner']:
            atlas_name = modality_config.get(f'{atlas_type}_atlas')
            if atlas_name is not None:  # Atlas is used
                self.atlas_names.append(atlas_name)
                self.atlas_rois.append(modality_config[f'{atlas_type}_rois'])
                self.network_counts.append(modality_config[f'{atlas_type}_networks'])
        
        self.total_rois = sum(self.atlas_rois)
        
        # Create atlas_combo for backward compatibility (keep the for loop structure)
        self.atlas_combo = [self.atlas_names]  # Single combo of all atlases
        
        # Determine final number of networks based on spatial modes
        if self.max_spatial:
            self.tot_num_networks = self.total_rois  # Each ROI = 1 network
        elif self.min_spatial:
            self.tot_num_networks = 1  # All ROIs = 1 network
        else:
            self.tot_num_networks = sum(self.network_counts)  # Standard mode

        # Temporal resampling parameters
        self.temporal_resampling_tr = temporal_resampling_tr
        self.source_tr = modality_config.get('source_tr', 2.0)  # Default TR of source data
        # Calculate number of patches (no overlap)
        self.num_patches = self.target_signal_length // modality_config['patch_size']

        if self.target_signal_length % self.patch_length != 0:
            self.padding = self.patch_length - (self.target_signal_length % self.patch_length)
        else:
            self.padding = 0

        self.masking_ratio = masking_ratio
        if masking_type == "network":
            self.masker = network_masker
        elif masking_type == "temporal":
            self.masker = temporal_masker
        elif masking_type == "slice":
            self.masker = slice_masker
        
        # Canonical network masking configuration
        self.canonical_network_masks = modality_config.get('canonical_network_masks', False)
        if self.canonical_network_masks:
            # Initialize canonical network tracking
            self.canonical_network_counts = self._get_canonical_networks()
            self.tot_canonical_networks = sum(self.canonical_network_counts)
            self.functional_to_actual_mapping = self._compute_functional_to_actual_mapping()

        self.file_handles = []
        self.dataset_configs = []
        self.file_subject_id_arrays = []
        # Stores mapping: subject_id -> list of valid sample definitions
        # fMRI sample definition: ('fmri', ds_idx, epoch_idx_in_file)
        self.subject_sample_pool = {}
        self.available_labels_in_files = {} # ds_idx -> set of available label keys in that HDF5

        # Build network mappings from explicit configuration
        try:
            if self.max_spatial:
                # Max spatial: each ROI = own network [0,1,2,...,total_rois-1]
                self.netmaps = {','.join(self.atlas_names): np.arange(self.total_rois, dtype=np.int64)}
            elif self.min_spatial:
                # Min spatial: all ROIs → network 0
                self.netmaps = {','.join(self.atlas_names): np.zeros(self.total_rois, dtype=np.int64)}
            else:
                # Standard: load from network mapping file
                all_maps = np.load(self.modality_config['network_map_path'])
                combo_map = []
                for atlas_name, network_count in zip(self.atlas_names, self.network_counts):
                    if network_count > 1:
                        map_key = f"network_map_{atlas_name}_{network_count}n"
                    else:
                        map_key = f"network_map_{atlas_name}"
                    combo_map.append(all_maps[map_key])
                concatenated_map = np.concatenate(combo_map, axis=0)
                self.netmaps = {','.join(self.atlas_names): concatenated_map}
        except:
            assert self.number_of_local_crops == 0

        # --- Aggregate data across all specified datasets for the modality ---
        for ds_idx, dataset_info in enumerate(self.modality_config['datasets']):
            self._process_dataset(ds_idx, dataset_info)

        if not self.file_handles:
             raise ValueError(f"No valid HDF5 datasets could be loaded for modality '{self.modality_name}' in mode '{self.mode}'.")

        # --- Final list of unique subjects that have valid samples ---
        self.subject_ids = sorted([sid for sid, samples in self.subject_sample_pool.items() if samples])
        self.num_samples = len(self.subject_ids) # Length is number of unique subjects

        if self.num_samples == 0:
            raise ValueError(f"No subjects with valid samples found for modality '{self.modality_name}' in mode '{self.mode}' after filtering.")

        self.arr_mean = 0.0
        self.arr_std = 1.0

    def __del__(self):
        if hasattr(self, 'file_handles'):
            for f in self.file_handles:
                f.close()

    def __len__(self):
        """Returns the number of unique subjects available."""
        return self.num_samples

    def _process_dataset(self, ds_idx, dataset_info):
        """
        Helper method to process a single dataset configuration, load its data,
        and populate the subject sample pool.
        """
        data_path = dataset_info['data_path']
        raw_signal_length = dataset_info['raw_signal_length']

        try:
            f = h5py.File(data_path, "r")
            self.file_handles.append(f)
            self.dataset_configs.append(dataset_info)
        except (IOError, FileNotFoundError) as e:
            logging.error(f"Could not open or read HDF5 file {data_path}: {e}")
            return

        # --- Check for available labels in this HDF5 file ---
        self.available_labels_in_files[ds_idx] = set()
        if self.probe_label_names:
            for label_name in self.probe_label_names:
                if label_name in f:
                    self.available_labels_in_files[ds_idx].add(label_name)
                else:
                    logging.warning(f"    Label '{label_name}' not found in HDF5 file {data_path}.")


        # --- Subject ID loading ---
        file_subject_ids_bytes = f["long_subject_id"][:]
        file_subject_ids = [s.decode('utf-8') for s in file_subject_ids_bytes]
        self.file_subject_id_arrays.append(file_subject_ids_bytes)
        unique_file_subjects = set(file_subject_ids)

        # --- Determine Target Subjects for this file ---
        subject_id_key = f"{self.mode}_subject_ids_path"
        subject_id_path = dataset_info.get(subject_id_key)

        # Check if direct subject IDs are provided (for automatic splitting)
        if '_target_subject_ids' in dataset_info:
            target_subject_ids = set(dataset_info['_target_subject_ids'])
            logging.info(f"    Using provided target subject IDs for '{self.mode}' mode: {len(target_subject_ids)} subjects")
        elif subject_id_path and os.path.exists(subject_id_path):
            target_subject_ids = set(np.load(subject_id_path, allow_pickle=True))
            logging.info(f"    Loaded {len(target_subject_ids)} target subject IDs for '{self.mode}' mode from {subject_id_path}")
        elif subject_id_path:
            logging.warning(f"    Subject ID path for '{self.mode}' not found: {subject_id_path}. Using all {len(unique_file_subjects)} subjects from HDF5 file.")
            target_subject_ids = unique_file_subjects
        else:
            logging.info(f"    '{subject_id_key}' not specified. Using all {len(unique_file_subjects)} unique subjects from HDF5 file for '{self.mode}' mode.")
            target_subject_ids = unique_file_subjects

        # --- Map subject IDs to their epoch indices within this file ---
        subject_epochs_in_file = {}
        for epoch_idx, subj_id in enumerate(file_subject_ids):
            if subj_id in target_subject_ids:
                if subj_id not in subject_epochs_in_file:
                    subject_epochs_in_file[subj_id] = []
                subject_epochs_in_file[subj_id].append(epoch_idx)

        # --- Pre-calculate and Store Valid Sample Definitions ---
        num_valid_samples_added = 0

        for subj_id, epoch_indices in subject_epochs_in_file.items():
            if subj_id not in self.subject_sample_pool:
                self.subject_sample_pool[subj_id] = []
            for epoch_idx in epoch_indices:
                self.subject_sample_pool[subj_id].append( ('fmri', ds_idx, epoch_idx) )
                num_valid_samples_added += 1

    def _get_crop_start_indices(self, max_start: int) -> np.ndarray:
        """
        Calculate crop start indices based on crop_starts mode.

        Args:
            max_start: Maximum valid start position (full_length - target_signal_length)

        Returns:
            Array of start indices for each crop
        """
        if max_start <= 0:
            return np.zeros(self.number_of_crops, dtype=int)

        if self.crop_starts.startswith('regular'):
            # Evenly spaced crops
            return np.linspace(0, max_start, self.number_of_crops).astype(int)
        else:
            # Random crops (handles 'random_*' and 'conditional_*')
            return np.random.randint(0, max_start + 1, self.number_of_crops)

    def __getitem__(self, index, sudo_aug=None):
        """
        Returns a sample dictionary containing signal crops and optionally labels
        for the subject corresponding to the given index.
        """
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"Index {index} out of bounds for dataset with size {self.num_samples}")

        # Get subject and sample info
        subject_id = self.subject_ids[index]
        sample_info = random.choice(self.subject_sample_pool.get(subject_id))
        ds_idx = sample_info[1]
        epoch_idx_in_file = sample_info[-1]
        file_handle = self.file_handles[ds_idx]

        # Determine signal length (from metadata or array shape)
        if 'ts_len' in file_handle:
            original_full_length = file_handle["ts_len"][epoch_idx_in_file]
        else:
            original_full_length = file_handle["timeseries/" + self.atlas_names[0]].shape[-1]

        # Calculate full_length after potential temporal resampling
        if self.temporal_resampling_tr is not None and self.temporal_resampling_tr != self.source_tr:
            resample_factor = self.source_tr / self.temporal_resampling_tr
            full_length = int(original_full_length * resample_factor)
        else:
            full_length = original_full_length

        # Load and concatenate timeseries from all atlases
        epoch_data = np.concatenate(
            [file_handle["timeseries/" + an][epoch_idx_in_file, :, :].astype(np.float32)
             for an in self.atlas_names], axis=0
        )  # [C, T]
        epoch_data = self.zscore_roiwise(epoch_data)

        # Apply temporal resampling if needed
        if self.temporal_resampling_tr is not None:
            resampled_list = [self._resample_temporal(epoch_data[i]) for i in range(epoch_data.shape[0])]
            epoch_data = np.stack(resampled_list, axis=0)

        # Truncate at first NaN if present
        nan_indices = np.where(np.isnan(epoch_data[0]))[0]
        if len(nan_indices) > 0:
            epoch_data = epoch_data[:, :nan_indices[0]]
            full_length = nan_indices[0]

        # Handle short signals by padding at end
        if full_length < self.target_signal_length:
            pad_needed = self.target_signal_length - full_length
            epoch_data = np.pad(epoch_data, ((0, 0), (0, pad_needed)), mode='constant', constant_values=0)
            start_idx = np.zeros(self.number_of_crops, dtype=int)
        else:
            # Apply patch alignment padding if needed
            if self.padding > 0:
                epoch_data = np.pad(epoch_data, ((0, 0), (0, self.padding)), mode='constant', constant_values=0)
            max_start = full_length - self.target_signal_length
            start_idx = self._get_crop_start_indices(max_start)

        # Extract crops
        crops = []
        for i in range(self.number_of_crops):
            crop_data = epoch_data[:, start_idx[i]:start_idx[i] + self.target_signal_length].astype(np.float32)
            if self.use_augmentation and np.sum(self.augment_level) > 0:
                crop_data = self.augment(crop_data)
            crops.append(crop_data)

        # Load labels if in probe mode
        labels_dict = {}
        if self.probe_label_names:
            available_labels = self.available_labels_in_files.get(ds_idx, set())
            for label_name in self.probe_label_names:
                if label_name in available_labels:
                    label_value = file_handle[label_name][epoch_idx_in_file]
                    if isinstance(label_value, bytes):
                        label_value = label_value.decode('utf-8')
                    labels_dict[label_name] = label_value
                else:
                    labels_dict[label_name] = np.nan

        return_dict = {
            "signal": [[torch.from_numpy(crop) for crop in crops]],
            "id": subject_id,
            "atlas_names": [[','.join(self.atlas_names) for _ in range(self.number_of_crops)]],
            "mask": self._generate_masks(crops),
            "view_starts": torch.tensor(start_idx),
        }
        if self.probe_label_names:
            return_dict["labels"] = labels_dict

        return return_dict
    
    def zscore_roiwise(self, signal):
        """Signal is C, T"""
        return (signal - np.nanmean(signal, axis=1, keepdims=True)) / (np.nanstd(signal, axis=1, keepdims=True) + 1e-8)

    def sample_by_id(self, target_id):
        """
        Samples a single data instance for the given subject ID using the pre-calculated pool.
        Args:
            target_id (str): The subject ID to sample.
        Returns:
            dict: A sample dictionary {'signal': tensor, 'id': str}.
        Raises:
            ValueError: If the target_id is not found or has no valid samples.
        """
        if target_id not in self.subject_sample_pool or not self.subject_sample_pool[target_id]:
            raise ValueError(f"No valid samples found for subject ID: {target_id} in mode '{self.mode}' for modality '{self.modality_name}'")

        # Find the index corresponding to the target_id in our sorted list
        try:
            # Note: This index isn't strictly needed if we just sample from the pool,
            # but calling __getitem__ ensures consistent logic.
            index = self.subject_ids.index(target_id)
        except ValueError:
             raise ValueError(f"Subject ID {target_id} found in sample pool but not in the final subject list. Data inconsistency?")

        # Use the main __getitem__ logic to get the sample
        # This ensures random selection from the pool if called multiple times for the same ID
        return self[index]
    
    def _set_augment_params(self):
        # Augmentation parameter lookup tables
        aug_params = {
            'spatial': [0.0, [0.0, 0.1], [0.1, 0.3], [0.25, 0.5]],
            'temporal': [0.0, [0.0, 0.3], [0.2, 0.4], [0.3, 0.6]],
            'noise': [0.0, 0.1, 0.2, 0.4],
            'scale': [1.0, [0.8, 1.2], [0.6, 1.4], [0.4, 1.6]]
        }
        
        # Handle list input [spatial, temporal, noise, scale] or single int
        if isinstance(self.augment_level, (list, tuple)):
            if len(self.augment_level) != 4:
                raise ValueError(f"augment_level list must have 4 elements [spatial, temporal, noise, scale], got {len(self.augment_level)}")
            levels = self.augment_level
        else:
            # Single int applies same level to all augmentation types
            levels = [self.augment_level] * 4
        
        # Validate and set parameters
        if not all(0 <= lvl <= 3 for lvl in levels):
            raise ValueError(f"All augment levels must be 0-3, got {levels}")
        
        self.spatial_masking_ratio = aug_params['spatial'][levels[0]]
        self.temporal_masking_ratio = aug_params['temporal'][levels[1]]
        self.noise_std = aug_params['noise'][levels[2]]
        self.scale_range = aug_params['scale'][levels[3]]
    
    def augment(self, signal): # signal is [C, T]
        # Sample augmentation parameters from ranges if needed

        if isinstance(self.spatial_masking_ratio, list):
            spatial_ratio = np.random.uniform(self.spatial_masking_ratio[0], self.spatial_masking_ratio[1])
        else:
            spatial_ratio = self.spatial_masking_ratio
            
        if isinstance(self.temporal_masking_ratio, list):
            temporal_ratio = np.random.uniform(self.temporal_masking_ratio[0], self.temporal_masking_ratio[1])
        else:
            temporal_ratio = self.temporal_masking_ratio
            
        if isinstance(self.scale_range, list):
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale_factor = self.scale_range
            
        num_channels = signal.shape[0]
        
        # Zero out some channels (spatial masking)
        if spatial_ratio > 0:
            num_rois_to_zero = max(1, int(num_channels * spatial_ratio))
            zero_indices = np.random.choice(num_channels, size=num_rois_to_zero, replace=False)
            signal[zero_indices, :] = 0.0

        # Zero out some timepoints (contiguous block - temporal masking)
        if temporal_ratio > 0:
            num_timepoints = signal.shape[1]
            num_timepoints_to_zero = max(1, int(num_timepoints * temporal_ratio))
            # Random starting position for the temporal mask
            max_start = num_timepoints - num_timepoints_to_zero
            if max_start > 0:
                start_idx = np.random.randint(0, max_start + 1)
                signal[:, start_idx:start_idx + num_timepoints_to_zero] = 0.0
            else:
                # If mask is larger than signal, zero everything
                signal[:, :] = 0.0
                
        # Scale amplitude (only if scale_factor != 1.0)
        if scale_factor != 1.0:
            signal *= scale_factor
                
        # Apply random noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, signal.shape)
            signal += noise
        
        return signal
    
    def _get_canonical_networks(self):
        """
        Returns the canonical (functional) network counts for each atlas.
        Hardcoded: Schaefer→7, Tian→1, Buckner→1
        """
        canonical_counts = []
        for atlas_name in self.atlas_names:
            if 'schaefer' in atlas_name.lower():
                canonical_counts.append(7)  # 7 canonical networks for Schaefer
            elif 'tian' in atlas_name.lower():
                canonical_counts.append(1)  # Treat Tian as single functional unit
            elif 'buckner' in atlas_name.lower():
                canonical_counts.append(1)  # Treat Buckner as single functional unit
            else:
                # Default to actual network count if atlas type unknown
                idx = self.atlas_names.index(atlas_name)
                canonical_counts.append(self.network_counts[idx])
        return canonical_counts
    
    def _compute_functional_to_actual_mapping(self):
        """
        Creates mapping from functional (canonical) network indices to actual network indices
        using network mapping files to maintain neuroanatomical correspondence.
        """
        mapping = {}
        functional_idx = 0
        actual_idx_offset = 0
        
        # Load network maps
        all_maps = np.load(self.modality_config['network_map_path'])
        
        for atlas_idx, (atlas_name, canonical_count, actual_count) in enumerate(
            zip(self.atlas_names, self.canonical_network_counts, self.network_counts)):
            
            if canonical_count == actual_count:
                # No mapping needed - canonical equals actual
                for i in range(canonical_count):
                    mapping[functional_idx + i] = [actual_idx_offset + i]
                functional_idx += canonical_count
            else:
                # Load canonical and actual network maps
                canonical_map_key = f"network_map_{atlas_name}_{canonical_count}n"
                actual_map_key = f"network_map_{atlas_name}_{actual_count}n"
                
                canonical_map = all_maps[canonical_map_key]  # [n_rois] -> canonical_network_id
                actual_map = all_maps[actual_map_key]        # [n_rois] -> actual_network_id
                
                # Get unique network values and create value-to-index mappings
                canonical_unique_values = np.unique(canonical_map)
                actual_unique_values = np.unique(actual_map)
                
                # Create value-to-index mappings
                canonical_val_to_idx = {val: idx for idx, val in enumerate(canonical_unique_values)}
                actual_val_to_idx = {val: idx for idx, val in enumerate(actual_unique_values)}
                
                # Create mapping based on ROI correspondence
                for canonical_idx in range(canonical_count):
                    canonical_value = canonical_unique_values[canonical_idx]
                    
                    # Find ROIs belonging to this canonical network
                    canonical_rois = np.where(canonical_map == canonical_value)[0]
                    
                    # Find which actual network values these ROIs map to
                    actual_vals_for_rois = actual_map[canonical_rois]
                    unique_actual_vals = np.unique(actual_vals_for_rois)
                    
                    # Convert actual values to indices, then offset for multi-atlas
                    actual_indices = [actual_val_to_idx[val] for val in unique_actual_vals]
                    mapping[functional_idx] = [actual_idx_offset + idx for idx in actual_indices]
                    functional_idx += 1
            
            actual_idx_offset += actual_count
        
        return mapping
    
    def _expand_functional_mask(self, functional_mask):
        """
        Expands a functional (canonical) level mask to actual network level.
        
        Args:
            functional_mask: [C_canonical, T] tensor where C_canonical is tot_canonical_networks
            
        Returns:
            actual_mask: [C_actual, T] tensor where C_actual is tot_num_networks
        """
        C_actual = self.tot_num_networks
        T = functional_mask.shape[1]
        actual_mask = torch.zeros(C_actual, T, dtype=functional_mask.dtype, device=functional_mask.device)
        
        for func_idx, actual_indices in self.functional_to_actual_mapping.items():
            # Copy the functional mask to all corresponding actual networks
            for actual_idx in actual_indices:
                actual_mask[actual_idx] = functional_mask[func_idx]
        
        return actual_mask
    
    def _generate_masks(self, atlas_crops):
        """        
        Args:
            atlas_crops: List of crops to determine the number of masks needed
            
        Returns:
            List of masks: [mask_for_view1, mask_for_view2]
        """
        if self.canonical_network_masks:
            # Generate one mask per crop at canonical level
            masks_canonical = [self.masker(self.tot_canonical_networks, self.num_patches, self.masking_ratio) 
                                for _ in atlas_crops]
            # Expand each mask to actual network level
            return [self._expand_functional_mask(mask) for mask in masks_canonical]
        else:
            return [self.masker(self.tot_num_networks, self.num_patches, self.masking_ratio) 
                    for _ in atlas_crops]
    
    def _resample_temporal(self, data):
        """
        Efficiently resample temporal dimension from source TR to target TR using vectorized operations.
        
        Args:
            data: numpy array of shape (C, T) where C=channels, T=timepoints
            
        Returns:
            Resampled data with new temporal dimension
        """
        if self.temporal_resampling_tr is None or self.temporal_resampling_tr == self.source_tr:
            return data
            
        # Calculate resampling factor
        resample_factor = self.source_tr / self.temporal_resampling_tr
        original_length = data.shape[-1]
        new_length = int(original_length * resample_factor)
        
        # Create time indices
        original_times = np.arange(original_length)
        new_times = np.arange(new_length) / resample_factor
        
        # Create interpolator for all channels at once
        # data.T makes it (T, C) so axis=0 interpolates along time
        interpolator = interp1d(original_times, data.T, kind='linear', 
                               axis=0, bounds_error=False, fill_value='extrapolate')
        
        # Interpolate all channels in one go and transpose back to (C, T)
        resampled_data = interpolator(new_times).T
        
        # print("old and new shape:", data.shape, resampled_data.shape)
        
        return resampled_data.astype(data.dtype)


def simple_custom_collate(batch):
    elem = batch[0]
    collated_batch = {}
    keys_to_list_collate = ["local_crops"] # Add other keys here if they also cause issues

    for key in elem:
        if key in keys_to_list_collate:
            # For specified keys, just gather them into a list.
            # Each element in this list will be the structure as returned by __getitem__ for one sample.
            collated_batch[key] = [sample[key] for sample in batch]
        else:
            # For all other keys, use the default collate behavior.
            # This will attempt to stack tensors if possible.
            try:
                collated_batch[key] = default_collate([sample[key] for sample in batch])
            except RuntimeError as e:
                # If default_collate fails for other keys, you might need to handle them explicitly too.
                # For now, we'll let it raise an error to identify problematic keys.
                print(f"Warning: default_collate failed for key '{key}': {e}. Collecting as list instead.")
                collated_batch[key] = [sample[key] for sample in batch]
                
    return collated_batch

def network_masker(C, T, spatial_masking_range=(1/8, 4/8)):
    mask = torch.zeros(C, T)

    # sample networks
    perm = torch.randperm(C)
    min_networks = max(1, int(C * spatial_masking_range[0]))  # Ensure at least 1 network
    max_networks = min(C, int(C * spatial_masking_range[1]))  # Ensure at most C-1 networks
    num_idx = torch.randint(min_networks, max_networks, (1,)).item()
    network_idx = perm[:num_idx]
    mask[network_idx, :] = 1
    return mask.bool()

def temporal_masker(C, T, spatial_masking_range=(1/8, 4/8)):
    mask = torch.zeros(C, T)
    masking_ratio = random.uniform(spatial_masking_range[0], spatial_masking_range[1])

    # Ensure at least 1 timepoint is masked
    mask_length = max(1, math.ceil(masking_ratio * T))
    mask_length = min(mask_length, T) - 1  # Ensure not larger than T - 1
    
    # starting timepoint
    max_start = max(0, T - mask_length)
    start = random.randint(0, max_start)
    end = start + mask_length
    mask[:, start:end] = 1

    return mask.bool()

def slice_masker(C, T, masking_range: tuple[float, float] = (0.125, 0.5)):
    mask_type = random.choice(['temporal', 'network'])
    if mask_type == 'temporal':
        return temporal_masker(C, T, masking_range)
    else:
        return network_masker(C, T, masking_range)

