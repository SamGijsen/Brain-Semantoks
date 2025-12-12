from collections import defaultdict


def get_simdino_lr_decay_rate(name, lr_decay_rate=1.0, num_transformer_layers=12):
    """
    Calculate lr decay rate for different SimDINO model components.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_transformer_layers (int): number of transformer blocks in InterNetworkTransformer.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_transformer_layers + 1  # Default to final layer
    
    # Remove any encoder prefix (e.g., "student_encoder." or "teacher_encoder.")
    # This handles both full model (with prefix) and encoder-only (without prefix) cases
    clean_name = name
    if name.startswith("student_encoder.") or name.startswith("teacher_encoder."):
        clean_name = name.split(".", 1)[1]  # Remove first component
    
    # Embeddings and tokens (layer 0)
    if any(token in clean_name for token in [
        "cls_token", "mask_embedding", "fuse_token", 
        "pos_embedding", "atlas_embeddings", "network_type_embeddings",
        "network_learnable_pos_embedding", "atlas_positional_embeddings"
    ]):
        layer_id = 0
    
    # CNN components (early layers - layer 1)
    elif "intra_network_cnns" in clean_name:
        layer_id = 1
    
    # Gradient projection and fusion components (layer 1)
    elif any(component in clean_name for component in [
        "gradient_projection", "atlas_fusion", "linear_refinement", 
        "cat_mlp", "shared_atlas_fusion_layer", "network_atlas_fusion_modules"
    ]):
        layer_id = 1
    
    # Transformer blocks (layers 2 to num_transformer_layers+1)
    elif "transformer_blocks." in clean_name:
        try:
            block_idx = int(clean_name.split("transformer_blocks.")[1].split(".")[0])
            layer_id = block_idx + 2  # Offset by 2 (0=embeddings, 1=cnns)
        except (ValueError, IndexError):
            layer_id = 2  # Default transformer layer
    
    # Projection heads (final layers)
    elif any(head in clean_name for head in ["student_predictor", "teacher_predictor", "student_mask_predictor", "teacher_mask_predictor"]):
        layer_id = num_transformer_layers + 1
    
    # Final norm and head (final layer)
    elif clean_name.startswith("norm") or clean_name.startswith("head"):
        layer_id = num_transformer_layers + 1
    
    return lr_decay_rate ** (num_transformer_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0):
    """
    Create parameter groups with different learning rates for layerwise decay.
    """
    # Determine number of transformer layers
    num_transformer_layers = 0
    
    # Try full model structure first (pretraining case)
    if hasattr(model, "student_encoder") and hasattr(model.student_encoder, "inter_network_transformer"):
        inter_tf = model.student_encoder.inter_network_transformer
        if hasattr(inter_tf, "transformer_blocks"):
            num_transformer_layers = len(inter_tf.transformer_blocks)
    elif hasattr(model, "module") and hasattr(model.module, "student_encoder"):
        # Handle DataParallel case
        inter_tf = model.module.student_encoder.inter_network_transformer
        if hasattr(inter_tf, "transformer_blocks"):
            num_transformer_layers = len(inter_tf.transformer_blocks)
    # Try encoder-only structure (finetuning case)
    elif hasattr(model, "inter_network_transformer") and hasattr(model.inter_network_transformer, "transformer_blocks"):
        num_transformer_layers = len(model.inter_network_transformer.transformer_blocks)
    elif hasattr(model, "module") and hasattr(model.module, "inter_network_transformer"):
        # Handle DataParallel encoder case
        num_transformer_layers = len(model.module.inter_network_transformer.transformer_blocks)
    
    all_param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Remove 'module.' prefix from DataParallel
        clean_name = name.replace("module.", "")
        
        decay_rate = get_simdino_lr_decay_rate(
            clean_name, lr_decay_rate, num_transformer_layers
        )
        
        d = {
            "params": param, 
            "lr_multiplier": decay_rate, 
            "wd_multiplier": 1.0, 
            "name": clean_name
        }
        
        # No weight decay for biases and normalization layers
        if clean_name.endswith(".bias") or "norm" in clean_name or "gamma" in clean_name:
            d.update({"wd_multiplier": 0.0})
        
        all_param_groups.append(d)
    
    return all_param_groups


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier")):
    """
    Fuse parameter groups with the same lr_multiplier and wd_multiplier.
    """
    fused_params_groups = defaultdict(lambda: {"params": []})
    
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"
        
        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])
    
    return list(fused_params_groups.values()) 