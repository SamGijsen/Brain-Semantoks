# linear_probe.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import random
import time
import logging
from collections import defaultdict
import math
import pandas as pd

from data.dataset import FMRIDataset, simple_custom_collate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LinearProbeHead(nn.Module):
    """Simple Linear Layer for Probing."""
    def __init__(self, feature_dim, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm1d(feature_dim)
        self.layer = nn.Linear(feature_dim, num_classes)
        nn.init.trunc_normal_(self.layer.weight, std=2e-5)

    def forward(self, x):
        return self.layer(self.bn(x))

def _extract_features_and_labels(model, dataloader, feature_types, device, num_crops=1, network_mask=None):
    """
    Extracts features and labels from a dataloader for specified feature types.

    Args:
        model: The model to extract features from
        dataloader: DataLoader providing batches
        feature_types: List of feature types to extract
        device: Device to run computations on
        num_crops: Number of crops to use from batch['signal']
        network_mask (Optional[int]): If provided, mask all networks EXCEPT this one (0-indexed)

    Returns:
        dict: {
            'features': {feature_type: Tensor(N, D)},
            'labels': {label_name: array},
            'ids': list[subject_id]
        }
    """
    model.eval()
    all_features = {ft: [] for ft in feature_types}
    all_labels = defaultdict(list)
    all_ids = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            subject_ids = batch['id']
            batch_labels = batch.get('labels', {})  # Get labels dict if present

            # Use first atlas (aggregated atlas with all sub-atlases concatenated)
            atlas_idx = 0
            for crop_idx in range(num_crops):
                # Always use aggregated atlas mode (single atlas containing all sub-atlases)
                view = [batch['signal'][j][crop_idx].to(device) for j in range(len(batch['signal']))]

                # Extend IDs and labels for each crop (duplicating them)
                all_ids.extend(subject_ids)
                for label_name, label_values in batch_labels.items():
                    if isinstance(label_values, torch.Tensor):
                        all_labels[label_name].extend(label_values.cpu().numpy())
                    else:
                        all_labels[label_name].extend(label_values)

                # Extract features for each requested type
                for ft in feature_types:
                    features = model.extract_features(view, atlas_idx, use_teacher=True, feature_type=ft, network_mask=network_mask)
                    all_features[ft].append(features.cpu())

    # Concatenate features for each type
    final_features = {}
    if all_ids:  # Only concatenate if data was processed
        for ft in feature_types:
            if all_features[ft]:
                final_features[ft] = torch.cat(all_features[ft], dim=0)
            else:
                final_features[ft] = torch.empty(0)  # Handle empty dataloader case

    # Convert label lists to numpy arrays for easier handling
    final_labels = {name: np.array(vals) for name, vals in all_labels.items()}

    return {
        'features': final_features,
        'labels': final_labels,
        'ids': all_ids
    }


def _train_eval_probe_head(head, optimizer, criterion, train_features, train_labels, eval_features, eval_labels, epochs, batch_size, device):
    """Trains and evaluates a single linear probe head configuration."""
    head.train()
    train_dataset = TensorDataset(train_features, torch.from_numpy(train_labels))
    
    if torch.isnan(train_features).any() or torch.isinf(train_features).any():
        logging.warning("Train features contain NaNs or infs. Exiting.")
        return -1.0, None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    iter_count = 0
    train_loss_sum = 0.0
    # epochs = math.ceil(iterations / len(train_loader))

    logging.debug(f"Starting head training for {epochs} epochs.")

    for epoch in range(epochs):
        for batch_features, batch_labels in train_loader:

            if not isinstance(criterion, nn.CrossEntropyLoss):
                batch_labels = batch_labels.unsqueeze(-1)

            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = head(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            iter_count += 1

    avg_train_loss = train_loss_sum / iter_count if iter_count > 0 else 0
    logging.debug(f"Head training finished. Avg Loss: {avg_train_loss:.4f}")

    # Evaluation
    head.eval()
    eval_dataset = TensorDataset(eval_features, torch.from_numpy(eval_labels))
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in eval_loader:
            batch_features = batch_features.to(device)
            outputs = head(batch_features)
            all_outputs.append(outputs.cpu())
            all_labels.append(batch_labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Determine if it's classification or regression based on label name or criterion
    is_classification = isinstance(criterion, nn.CrossEntropyLoss)
    metrics = calculate_metrics(all_outputs, all_labels, is_classification)

    logging.debug(f"Head evaluation metrics: {metrics}")
    return metrics.get('balanced_accuracy' if is_classification else 'mae', -1.0), head

def calculate_metrics(outputs, labels, is_classification):
    """
    Calculate appropriate metrics based on whether the task is classification or regression.
    
    Args:
        outputs (torch.Tensor): Model predictions.
        labels (torch.Tensor): Ground truth labels.
        is_classification (bool): Whether the task is classification (True) or regression (False).
    
    Returns:
        dict: Dictionary of metric names and their values.
    """
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, mean_absolute_error, 
                                 r2_score, accuracy_score, f1_score, precision_score, recall_score)
    import numpy as np
    import scipy.stats as stats

    metrics = {}
    if is_classification:
        # For classification, outputs are logits
        probabilities = torch.softmax(outputs, dim=1).numpy()
        predictions = torch.argmax(outputs, dim=1).numpy()
        labels_np = labels.numpy()
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(labels_np, predictions) * 100
        
        # Balanced Accuracy
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels_np, predictions) * 100
        
        # F1, Precision, Recall
        if probabilities.shape[1] == 2:  # Binary classification
            metrics['f1'] = f1_score(labels_np, predictions, average='binary') * 100
            metrics['precision'] = precision_score(labels_np, predictions, average='binary') * 100
            metrics['recall'] = recall_score(labels_np, predictions, average='binary') * 100
            metrics['auroc'] = roc_auc_score(labels_np, probabilities[:, 1]) * 100
        else:  # Multi-class
            metrics['f1_macro'] = f1_score(labels_np, predictions, average='macro') * 100
            metrics['precision_macro'] = precision_score(labels_np, predictions, average='macro') * 100
            metrics['recall_macro'] = recall_score(labels_np, predictions, average='macro') * 100
            metrics['f1_weighted'] = f1_score(labels_np, predictions, average='weighted') * 100
            metrics['precision_weighted'] = precision_score(labels_np, predictions, average='weighted') * 100
            metrics['recall_weighted'] = recall_score(labels_np, predictions, average='weighted') * 100
            try:
                metrics['auroc'] = roc_auc_score(labels_np, probabilities, multi_class='ovr') * 100
            except ValueError:
                # AUROC might fail for some multi-class cases
                metrics['auroc'] = -1.0
    else:
        # For regression, outputs are raw values
        predictions = outputs.squeeze().numpy()
        labels_np = labels.numpy()
        metrics['mae'] = mean_absolute_error(labels_np, predictions)
        metrics['r2'] = r2_score(labels_np, predictions)
        metrics['pearson_r'] = stats.pearsonr(labels_np, predictions)[0]

    return metrics

def run_linear_probe(ssl_model, probe_config, main_data_config, device, config=None):
    """
    Runs the linear probing evaluation pipeline.

    Args:
        ssl_model (nn.Module): The pre-trained SSL model (student or teacher backbone selected).
        probe_config (dict): Configuration dictionary for linear probing.
        main_data_config (dict): Main data configuration (used for dataset params).
        device (torch.device): Device to run computations on.
        config (dict, optional): Full configuration dictionary to access training seed.

    Returns:
        dict: Results dictionary mapping dataset_name -> label_name to evaluation metrics.
              e.g., {'ukbb': {'labels/sex_cat': {'best_val_acc': 85.2, 'test_acc': 84.9, 'best_lr': 0.01, 'best_feature_type': 'cls_avg'}}}
    """
    start_time = time.time()
    logging.info("--- Starting Linear Probing Evaluation ---")

    # Get training seed from config, fallback to 0 if not available
    training_seed = config.get('training', {}).get('seed', 0) if config else 0
    
    # Get number of repetitions
    n_repetitions = probe_config.get('n_repetitions', 1)
    logging.info(f"Number of repetitions: {n_repetitions}")

    feature_types = probe_config['feature_types']
    learning_rates = probe_config['learning_rates']
    probe_batch_size = probe_config['probe_batch_size']
    num_workers = probe_config['probe_num_workers']
    test_time_crops = probe_config.get('test_time_crops', 1)
    probe_epochs = probe_config.get('probe_epochs', 50)

    # Build atlas names from explicit config (always use aggregated atlas)
    atlas_names = []
    for atlas_type in ['schaefer', 'tian', 'buckner']:
        atlas_name = main_data_config.get(f'{atlas_type}_atlas')
        if atlas_name is not None:  # Atlas is used
            atlas_names.append(atlas_name)

    logging.info(f"Using aggregated atlas: {atlas_names}")

    feature_extractor = ssl_model
    feature_extractor.eval() 

    all_results = {}

    for probe_ds_info in probe_config.get('probe_datasets', []):
        ds_name = probe_ds_info['name']
        logging.info(f"--- Processing Dataset: {ds_name} ---")
        
        # Ensure the probe dataset entry in the config is self-contained
        required_keys = ['data_path', 'raw_signal_length', 'label_names', 'n_class']
        if not all(key in probe_ds_info for key in required_keys):
            logging.warning(f"Skipping probe dataset '{ds_name}' because required keys {required_keys} are missing in the config.")
            continue
        
        # Get dataset-specific parameters
        dataset_label_names = probe_ds_info['label_names']
        dataset_n_class = probe_ds_info['n_class']
        
        # Create mapping from label names to number of classes
        if len(dataset_label_names) != len(dataset_n_class):
            logging.warning(f"Skipping probe dataset '{ds_name}' because label_names and n_class have different lengths.")
            continue
            
        label_to_nclass = dict(zip(dataset_label_names, dataset_n_class))
        
        max_train_subjects = probe_ds_info.get('max_train_subjects_per_label', None)
        
        # Check if we need to split and determine number of repetitions to run
        split_keys = ['probe_train_subject_ids_path', 'probe_val_subject_ids_path', 'probe_test_subject_ids_path']
        provided_splits = [key for key in split_keys if key in probe_ds_info and probe_ds_info[key] is not None]
        
        if len(provided_splits) == 3:
            # All splits provided, ignore repetitions
            n_reps_to_run = 1
            logging.info(f"All splits provided for {ds_name}, ignoring n_repetitions")
        else:
            # Need splitting, use repetitions
            n_reps_to_run = n_repetitions
            logging.info(f"Splitting needed for {ds_name}, running {n_reps_to_run} repetitions")
        
        # Determine all subjects that will be needed across repetitions
        all_needed_subjects = set()
        
        if len(provided_splits) == 3:
            # All splits provided - load all subject files
            for split_key in split_keys:
                if split_key in probe_ds_info and probe_ds_info[split_key] is not None:
                    subjects = set(np.load(probe_ds_info[split_key], allow_pickle=True))
                    all_needed_subjects.update(subjects)
        else:
            # Splits will be generated from provided train subjects
            train_subjects = set(np.load(probe_ds_info['probe_train_subject_ids_path'], allow_pickle=True))
            all_needed_subjects.update(train_subjects)
            
            # If test subjects are also provided separately, include them
            if 'probe_test_subject_ids_path' in probe_ds_info and probe_ds_info['probe_test_subject_ids_path'] is not None:
                test_subjects = set(np.load(probe_ds_info['probe_test_subject_ids_path'], allow_pickle=True))
                all_needed_subjects.update(test_subjects)
        
        logging.info(f"Extracting features once for {len(all_needed_subjects)} subjects with {test_time_crops} crops each")
        
        # Create dataloader for all needed subjects with test_time_crops
        modality_config = {
            'name': main_data_config['name'],
            'target_signal_length': main_data_config['target_signal_length'],
            'number_of_crops': test_time_crops,  # Use test_time_crops for all subjects
            'patch_size': main_data_config['patch_size'],
            'channels': main_data_config.get('channels', 'all'),
            'min_crop_distance': 0,
            'max_crop_distance': 0,
            'network_map_path': main_data_config['network_map_path'],
            # Add explicit atlas specification
            'schaefer_atlas': main_data_config.get('schaefer_atlas'),
            'schaefer_rois': main_data_config.get('schaefer_rois'),
            'schaefer_networks': main_data_config.get('schaefer_networks'),
            'tian_atlas': main_data_config.get('tian_atlas'),
            'tian_rois': main_data_config.get('tian_rois'),
            'tian_networks': main_data_config.get('tian_networks'),
            'buckner_atlas': main_data_config.get('buckner_atlas'),
            'buckner_rois': main_data_config.get('buckner_rois'),
            'buckner_networks': main_data_config.get('buckner_networks'),
            'max_spatial': main_data_config.get('max_spatial', False),
            'min_spatial': main_data_config.get('min_spatial', False),
            'datasets': [{
                'name': ds_name,
                'data_path': probe_ds_info['data_path'],
                'raw_signal_length': probe_ds_info['raw_signal_length'],
                '_target_subject_ids': list(all_needed_subjects)
            }]
        }

        # Extract temporal resampling parameter from main data config
        temporal_resampling_tr = main_data_config.get('temporal_resampling_tr', None)

        dataset = FMRIDataset(
            modality_config=modality_config,
            mode='probe_train',  
            probe_label_names=dataset_label_names,
            use_augmentation=False,
            augment_level=[0, 0, 0, 0],  # No augmentation for linear probing [spatial, temporal, noise, scale]
            crop_starts='regular_align',
            temporal_resampling_tr=temporal_resampling_tr,
        )
        
        all_subjects_dataloader = DataLoader(
            dataset, 
            batch_size=int(probe_batch_size),  # Using configured batch size
            shuffle=False, 
            num_workers=num_workers,
            drop_last=False,
            # collate_fn=simple_custom_collate,
        )

        # Extract features once for all subjects (always using aggregated atlas mode)
        all_extracted_data = _extract_features_and_labels(
            feature_extractor,
            all_subjects_dataloader,
            feature_types,
            device,
            num_crops=test_time_crops,
        )
        
        # Create mapping from subject_id to indices in extracted data
        subject_to_indices = defaultdict(list)
        for idx, subject_id in enumerate(all_extracted_data['ids']):
            subject_to_indices[subject_id].append(idx)
        
        logging.info(f"Extracted features for {len(set(all_extracted_data['ids']))} unique subjects with {len(all_extracted_data['ids'])} total samples")
        
        # Store results for all repetitions
        dataset_repetition_results = []
        
        for rep in range(n_reps_to_run):
            if n_reps_to_run > 1:
                logging.info(f"--- Running repetition {rep + 1}/{n_reps_to_run} for {ds_name} ---")
            
            # Use different seed for each repetition
            rep_seed = training_seed + rep
            
            # Make a copy of probe_ds_info for this repetition to avoid modifying the original
            current_probe_ds_info = probe_ds_info.copy()
            
            if len(provided_splits) < 3:  # Need to perform automatic splitting
                from sklearn.model_selection import train_test_split
                import h5py
                
                stratify_vars = current_probe_ds_info.get('stratify', [])
                split_ratio = current_probe_ds_info.get('split_ratio', [0.6, 0.2, 0.2] if len(provided_splits) == 1 else [0.75, 0.25])
                logging.info(f"Splitting needed for {ds_name} (rep {rep + 1}). Provided splits: {len(provided_splits)}, using ratios: {split_ratio}")
                
                # Always start with the provided train subjects as the base for splitting
                provided_train_subjects = list(np.load(current_probe_ds_info['probe_train_subject_ids_path'], allow_pickle=True))
                
                # Load stratification variables if specified
                subject_stratify = None
                if stratify_vars:
                    with h5py.File(current_probe_ds_info['data_path'], 'r') as f:
                        all_subject_ids = [s.decode('utf-8') for s in f["long_subject_id"][:]]
                        
                        stratify_arrays = []
                        for var in stratify_vars:
                            if var in f:
                                stratify_arrays.append(f[var][:])
                            else:
                                logging.warning(f"Stratification variable '{var}' not found in {current_probe_ds_info['data_path']}")
                        
                        if stratify_arrays:
                            stratify_data = np.column_stack(stratify_arrays)
                            stratify_data = np.array(['_'.join(map(str, row)) for row in stratify_data])
                            
                            subject_to_stratify = {}
                            for i, subj in enumerate(all_subject_ids):
                                if subj in provided_train_subjects:
                                    subject_to_stratify[subj] = stratify_data[i]
                            
                            subject_stratify = [subject_to_stratify.get(subj) for subj in provided_train_subjects]

                    # Filter out subjects with invalid stratification labels (e.g., 'nan', None)
                    if subject_stratify:
                        valid_pairs = [
                            (subj, strat) for subj, strat in zip(provided_train_subjects, subject_stratify)
                            if strat is not None and strat != 'nan'
                        ]
                        if len(valid_pairs) < len(provided_train_subjects):
                            logging.info(f"Removed {len(provided_train_subjects) - len(valid_pairs)} subjects due to invalid stratification values.")
                        
                        if valid_pairs:
                            provided_train_subjects, subject_stratify = map(list, zip(*valid_pairs))
                        else: # All subjects were filtered out
                            provided_train_subjects = []
                            subject_stratify = None
                
                if len(provided_splits) == 1 and 'probe_train_subject_ids_path' in provided_splits:
                    # Case 1: Only training provided, split provided train subjects into train/val/test
                    train_ratio, val_ratio, test_ratio = split_ratio
                    
                    # First split: separate test set from provided train subjects
                    temp_subjects, test_subjects = train_test_split(
                        provided_train_subjects, test_size=test_ratio, random_state=rep_seed,
                        stratify=subject_stratify
                    )
                    
                    # Second split: separate train and val from remaining
                    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
                    if subject_stratify is not None:
                        temp_stratify = [subject_to_stratify[subj] for subj in temp_subjects]
                    else:
                        temp_stratify = None
                        
                    train_subjects, val_subjects = train_test_split(
                        temp_subjects, test_size=val_size_adjusted, random_state=rep_seed,
                        stratify=temp_stratify
                    )
                    
                    current_probe_ds_info['_generated_train_subjects'] = train_subjects
                    current_probe_ds_info['_generated_val_subjects'] = val_subjects
                    current_probe_ds_info['_generated_test_subjects'] = test_subjects
                    
                elif len(provided_splits) == 2 and 'probe_test_subject_ids_path' in provided_splits:
                    # Case 2: Train and test provided, split provided train subjects into train/val
                    train_ratio, val_ratio = split_ratio
                    
                    # Split provided training subjects into train/val
                    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
                    train_subjects, val_subjects = train_test_split(
                        provided_train_subjects, test_size=val_size_adjusted, random_state=rep_seed,
                        stratify=subject_stratify
                    )
                    
                    # Update current_probe_ds_info with generated splits
                    current_probe_ds_info['_generated_train_subjects'] = train_subjects
                    current_probe_ds_info['_generated_val_subjects'] = val_subjects
                    
                logging.info(f"Generated splits (rep {rep + 1}) - Train: {len(current_probe_ds_info.get('_generated_train_subjects', []))}, "
                            f"Val: {len(current_probe_ds_info.get('_generated_val_subjects', []))}, "
                            f"Test: {len(current_probe_ds_info.get('_generated_test_subjects', []))}")
        
            # Assign pre-extracted features to splits
            extracted_data = {}
            
            for split in ['probe_train', 'probe_val', 'probe_test']:
                # Get subject IDs for this split
                subject_id_path_key = f"{split}_subject_ids_path"
                generated_subjects_key = f"_generated_{split.replace('probe_', '')}_subjects"
                
                target_subject_ids = None
                
                # Prioritize generated subjects over file paths
                if generated_subjects_key in current_probe_ds_info:
                    target_subject_ids = set(current_probe_ds_info[generated_subjects_key])
                elif subject_id_path_key in current_probe_ds_info and current_probe_ds_info[subject_id_path_key] is not None:
                    target_subject_ids = set(np.load(current_probe_ds_info[subject_id_path_key], allow_pickle=True))
                else:
                    logging.info(f"'{split}' not available for dataset '{ds_name}'. Skipping this split.")
                    extracted_data[split] = {'features': {}, 'labels': {}, 'ids': []}
                    continue

                # Get indices for subjects in this split
                split_indices = []
                split_ids = []
                for subject_id in target_subject_ids:
                    if subject_id in subject_to_indices:

                        indices = subject_to_indices[subject_id] # all crops
                        
                        split_indices.extend(indices)
                        split_ids.extend([subject_id] * len(indices))

                # Extract features and labels for this split
                split_features = {}
                for ft in feature_types:
                    if split_indices:
                        split_features[ft] = all_extracted_data['features'][ft][split_indices]
                    else:
                        split_features[ft] = torch.empty(0)
                
                split_labels = {}
                for label_name in dataset_label_names:
                    if split_indices:
                        split_labels[label_name] = all_extracted_data['labels'][label_name][split_indices]
                    else:
                        split_labels[label_name] = np.array([])

                extracted_data[split] = {
                    'features': split_features,
                    'labels': split_labels,
                    'ids': split_ids
                }

            # Process each label in this dataset
            repetition_label_results = {}
                
            for label_name in dataset_label_names:
                logging.info(f"--- Probing Dataset '{ds_name}' for Label: {label_name} (rep {rep + 1}) ---")
                
                n_class = label_to_nclass[label_name]
                is_classification = n_class > 1
                
                label_results = {}
                best_val_acc = -float('inf') if is_classification else float('inf')
                best_config = {'lr': None, 'feature_type': None}
                best_head = None  # Store the best performing head
                
                # Prepare Data for this Label (Filter NaNs, Subsample)
                current_label_data = {}
                valid_subjects_exist = {'probe_train': False, 'probe_val': False, 'probe_test': False}

                for split in ['probe_train', 'probe_val', 'probe_test']:
                    split_data = extracted_data[split]
                    
                    labels = split_data['labels'].get(label_name)
                    ids = split_data['ids']  # Store IDs for later use in test-time averaging

                    # Identify valid (non-NaN) indices for this label
                    if labels.dtype == object or labels.dtype.kind == 'U':
                        try:
                            numeric_labels = pd.to_numeric(labels, errors='coerce')
                            valid_indices = np.where(~np.isnan(numeric_labels))[0]
                        except Exception:
                            logging.warning(f"Could not reliably convert labels for {label_name} in {split} to numeric for NaN check. Assuming all are valid.")
                            valid_indices = np.arange(len(labels))
                    else:
                        valid_indices = np.where(~np.isnan(labels))[0]

                    logging.info(f"  Split {split}: Found {len(valid_indices)} non-NaN samples for {label_name} (out of {len(labels)}).")

                    # Filter labels
                    filtered_labels = labels[valid_indices]
                    filtered_ids = [ids[i] for i in valid_indices]
                    
                    # Filter features for all types
                    filtered_features = {}
                    for ft in feature_types:
                        if ft in split_data['features']:
                            filtered_features[ft] = split_data['features'][ft][valid_indices]
                        else:
                            filtered_features[ft] = torch.empty(0)

                    # Apply training set subsampling
                    current_indices = np.arange(len(valid_indices))
                    if split == 'probe_train' and max_train_subjects is not None and len(valid_indices) > max_train_subjects:

                        random.seed(rep_seed)
                        if is_classification:
                            # For classification, balance classes
                            unique_classes = np.unique(filtered_labels)
                            samples_per_class = max_train_subjects // len(unique_classes)
                            if samples_per_class == 0:
                                samples_per_class = 1  # Ensure at least one sample per class if possible

                            selected_indices = []
                            for cls in unique_classes:
                                cls_indices = np.where(filtered_labels == cls)[0]
                                if len(cls_indices) > 0:
                                    if len(cls_indices) > samples_per_class:
                                        cls_selected = np.array(random.sample(list(cls_indices), samples_per_class))
                                    else:
                                        cls_selected = cls_indices
                                    selected_indices.extend(cls_selected.tolist())
                            current_indices = np.array(selected_indices)
                            if len(current_indices) > max_train_subjects:
                                current_indices = current_indices[:max_train_subjects]
                        else:
                            # For non-categorical, random subsample
                            current_indices = np.array(random.sample(list(current_indices), max_train_subjects))
                        
                        # Apply the subsampling to labels, features, and IDs
                        filtered_labels = filtered_labels[current_indices]
                        filtered_ids = [filtered_ids[i] for i in current_indices]
                        for ft in feature_types:
                            if ft in filtered_features:
                                filtered_features[ft] = filtered_features[ft][current_indices]
                        logging.info(f"  Training set subsampled to {len(current_indices)}.")

                    current_label_data[split] = {
                        'features': filtered_features,
                        'labels': filtered_labels,
                        'indices': current_indices,
                        'ids': filtered_ids
                    }
                    if len(current_indices) > 0:
                        valid_subjects_exist[split] = True

                if current_label_data is None:
                    continue

                # Determine number of classes dynamically for categorical labels
                if is_classification:
                    num_classes = n_class  # Use the configured number of classes
                else:
                    num_classes = 1
                
                criterion = nn.CrossEntropyLoss() if is_classification else nn.L1Loss()

                # Grid Search
                for lr in learning_rates:
                    for ft in feature_types:
                        logging.debug(f"    Testing LR={lr}, FeatureType={ft}")

                        # Get features for this type
                        train_feat = current_label_data['probe_train']['features'].get(ft)
                        val_feat = current_label_data['probe_val']['features'].get(ft)
                        train_lab = current_label_data['probe_train']['labels']
                        val_lab = current_label_data['probe_val']['labels']
                        if is_classification:
                            train_lab = train_lab.astype(np.int64)
                            val_lab = val_lab.astype(np.int64)
                        else:
                            train_lab = train_lab.astype(np.float32)
                            val_lab = val_lab.astype(np.float32)

                        feature_dim = train_feat.shape[1]
                        head = LinearProbeHead(feature_dim, num_classes).to(device)
                        optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9)
                                                                                
                        # Train and evaluate on validation set
                        adaptive_batch_size = min(probe_batch_size, max(1, len(train_feat) // 8))
                        val_metric, trained_head = _train_eval_probe_head(
                            head, optimizer, criterion,
                            train_feat, train_lab,
                            val_feat, val_lab,
                            probe_epochs, adaptive_batch_size, device
                        )
                        
                        metric_name = "Acc" if is_classification else "MAE"
                        # logging.info(f"    LR={lr}, FeatureType={ft} -> Val {metric_name}: {val_metric:.2f}" + ("%" if is_classification else ""))

                        # For classification, higher is better; for regression, lower is better
                        is_better = (val_metric > best_val_acc) if is_classification else (val_metric < best_val_acc)
                        
                        if is_better:
                            best_val_acc = val_metric
                            best_config['lr'] = lr
                            best_config['feature_type'] = ft
                            best_head = trained_head  # Save the best performing head

                # Final Test
                metric_name = "Acc" if is_classification else "MAE"
                logging.info(f"  Best validation config for {label_name}: LR={best_config['lr']}, FeatureType={best_config['feature_type']} (Val {metric_name}: {best_val_acc:.2f}" + ("%" if is_classification else "") + ")")
                test_metrics = {}

                best_ft = best_config['feature_type']
                best_lr = best_config['lr']

                # Get test data for the best feature type
                test_feat = current_label_data['probe_test']['features'].get(best_ft)
                test_lab = current_label_data['probe_test']['labels']
                test_ids = current_label_data['probe_test']['ids']
                
                # Convert labels to appropriate type
                if is_classification:
                    test_lab_typed = test_lab.astype(np.int64)
                else:
                    test_lab_typed = test_lab.astype(np.float32)

                # Evaluate on test set using the best head from validation
                best_head.eval()
                
                # Check if we have test data
                if len(test_feat) > 0:
                    with torch.no_grad():
                        test_outputs = best_head(test_feat.to(device)).cpu()
                        
                        # Sort by subject ID to ensure consistent ordering for TTA logic
                        if test_time_crops > 1 and len(test_ids) > 0:
                            sort_df = pd.DataFrame({'subject_id': test_ids, 'index': range(len(test_ids))})
                            sort_df = sort_df.sort_values(by='subject_id', kind='mergesort') # Use stable sort
                            sorted_indices = sort_df['index'].values
                            
                            test_ids = [test_ids[i] for i in sorted_indices]
                            test_lab_typed = test_lab_typed[sorted_indices]
                            test_outputs = test_outputs[sorted_indices]

                        # Check if we need to average predictions across crops
                        is_tta_active = test_time_crops > 1 and len(test_ids) > 0
                        
                        if is_tta_active:
                            # Average predictions by subject ID
                            if is_classification:
                                # For classification, average logits
                                df_outputs = pd.DataFrame(test_outputs.numpy())
                            else:
                                # For regression, average scalar outputs
                                df_outputs = pd.DataFrame({'output': test_outputs.squeeze().numpy()})
                                
                            # Add subject IDs
                            df_outputs['subject_id'] = test_ids
                            
                            # Group by subject ID and average
                            if is_classification:
                                averaged_outputs = df_outputs.groupby('subject_id').mean().values
                                averaged_outputs = torch.tensor(averaged_outputs)
                            else:
                                averaged_outputs = df_outputs.groupby('subject_id')['output'].mean().values
                                averaged_outputs = torch.tensor(averaged_outputs).unsqueeze(1)
                            
                            # Get unique labels per subject (all crops of same subject have same label)
                            df_labels = pd.DataFrame({'label': test_lab_typed, 'subject_id': test_ids})
                            unique_labels = df_labels.groupby('subject_id')['label'].first().values
                            unique_labels = torch.tensor(unique_labels, dtype=torch.int64 if is_classification else torch.float32)
                            
                            # Calculate metrics on subject-level predictions (ensembled)
                            test_metrics = calculate_metrics(averaged_outputs, unique_labels, is_classification)
                            test_primary_metric = test_metrics.get('balanced_accuracy' if is_classification else 'mae', -1.0)
                            
                            # Add prefix to ensembled metrics for clarity
                            test_metrics = {f'ensembled_{k}': v for k, v in test_metrics.items()}
                            
                            # Additional metrics: Random crop per subject
                            random.seed(rep_seed)  # Use repetition seed for reproducibility
                            df_all = pd.DataFrame({
                                'subject_id': test_ids,
                                'label': test_lab_typed
                            })
                            if is_classification:
                                for i in range(test_outputs.shape[1]):
                                    df_all[f'output_{i}'] = test_outputs[:, i].numpy()
                            else:
                                df_all['output'] = test_outputs.squeeze().numpy()
                            
                            # Sample one random crop per subject
                            random_crop_data = df_all.groupby('subject_id').apply(lambda x: x.sample(1, random_state=rep_seed)).reset_index(drop=True)
                            
                            if is_classification:
                                random_crop_outputs = torch.tensor(random_crop_data[[f'output_{i}' for i in range(test_outputs.shape[1])]].values)
                            else:
                                random_crop_outputs = torch.tensor(random_crop_data['output'].values).unsqueeze(1)
                            random_crop_labels = torch.tensor(random_crop_data['label'].values, dtype=torch.int64 if is_classification else torch.float32)
                            
                            random_crop_metrics = calculate_metrics(random_crop_outputs, random_crop_labels, is_classification)
                            for k, v in random_crop_metrics.items():
                                test_metrics[f'random_crop_{k}'] = v
                            
                            # Additional metrics: Per-crop metrics then average
                            num_unique_subjects = len(df_labels['subject_id'].unique())
                            crop_metrics_list = []
                            
                            for crop_idx in range(test_time_crops):
                                # Extract data for this crop - data is organized as subject-first, then crops
                                # So for each subject, we need to get the crop_idx-th occurrence
                                crop_indices = []
                                for subj_idx in range(num_unique_subjects):
                                    # Each subject has test_time_crops consecutive entries
                                    base_idx = subj_idx * test_time_crops
                                    crop_indices.append(base_idx + crop_idx)
                                
                                crop_outputs = test_outputs[crop_indices]
                                crop_labels = torch.tensor(test_lab_typed[crop_indices], dtype=torch.int64 if is_classification else torch.float32)
                                
                                crop_metrics = calculate_metrics(crop_outputs, crop_labels, is_classification)
                                crop_metrics_list.append(crop_metrics)
                            
                            # Average metrics across crops
                            if crop_metrics_list:
                                avg_crop_metrics = {}
                                for metric_name in crop_metrics_list[0].keys():
                                    metric_values = [cm[metric_name] for cm in crop_metrics_list]
                                    avg_crop_metrics[f'avg_single_crop_{metric_name}'] = np.mean(metric_values)
                                    avg_crop_metrics[f'std_single_crop_{metric_name}'] = np.std(metric_values)
                                
                                test_metrics.update(avg_crop_metrics)
                        else:
                            # No TTA, calculate metrics directly
                            test_metrics = calculate_metrics(test_outputs, torch.from_numpy(test_lab_typed), is_classification)
                            test_primary_metric = test_metrics.get('balanced_accuracy' if is_classification else 'mae', -1.0)
                else:
                    test_primary_metric = -1.0
                    test_metrics = {}
                
                # Log key metrics for better visibility
                if is_classification:
                    key_metrics = {k: v for k, v in test_metrics.items() if any(m in k for m in ['accuracy', 'f1', 'precision', 'recall', 'auroc'])}
                    logging.info(f"  Final Test Metrics for {label_name}: {key_metrics}")
                else:
                    logging.info(f"  Final Test Metrics for {label_name}: {test_metrics}")

                # Store results for this label, converting to standard Python types
                repetition_label_results[label_name] = {
                    'best_val_primary_metric': float(best_val_acc) if best_val_acc > -1.0 else None,
                    'test_primary_metric': float(test_primary_metric) if test_primary_metric > -1.0 else None,
                    'test_metrics': {k: float(v) for k, v in test_metrics.items()},
                    'best_lr': float(best_config['lr']),
                    'best_feature_type': best_config['feature_type']
                }
            
            # Store results for this repetition
            dataset_repetition_results.append(repetition_label_results)
        
        # Process results across repetitions
        if ds_name not in all_results:
            all_results[ds_name] = {}
        
        for label_name in dataset_label_names:
            if n_reps_to_run == 1:
                # Single repetition, store results directly
                all_results[ds_name][label_name] = dataset_repetition_results[0][label_name]
            else:
                # Multiple repetitions, store all repetitions and compute summary
                repetition_data = []
                for rep_results in dataset_repetition_results:
                    if label_name in rep_results:
                        repetition_data.append(rep_results[label_name])
                
                # Compute summary statistics
                if repetition_data:
                    # Extract test metrics across repetitions
                    test_primary_metrics = [r['test_primary_metric'] for r in repetition_data if r['test_primary_metric'] is not None]
                    val_primary_metrics = [r['best_val_primary_metric'] for r in repetition_data if r['best_val_primary_metric'] is not None]
                    
                    summary = {}
                    if test_primary_metrics:
                        summary['mean_test_primary_metric'] = float(np.mean(test_primary_metrics))
                        summary['std_test_primary_metric'] = float(np.std(test_primary_metrics))
                    if val_primary_metrics:
                        summary['mean_val_primary_metric'] = float(np.mean(val_primary_metrics))
                        summary['std_val_primary_metric'] = float(np.std(val_primary_metrics))
                    
                    all_results[ds_name][label_name] = {
                        'repetitions': repetition_data,
                        'summary': summary
                    }

    end_time = time.time()
    logging.info(f"--- Linear Probing Finished ({end_time - start_time:.2f}s) ---")
    # logging.info(f"Results: {all_results}")
    return all_results
