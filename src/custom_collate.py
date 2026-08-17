import torch

def train_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    context_length=None,
    device="cpu"
):
    """Collate function for training dataloader. Prepares input and target tensors for the model.   
    Args:
        batch: List of sequences (lists of token IDs) in the batch.
        pad_token_id: Token ID used for padding shorter sequences.
        ignore_index: Token ID used in targets to indicate positions that should be ignored in loss computation (e.g., padding tokens).
        context_length: Maximum length of the input sequence (context) for the model. If None, no truncation is applied and the longest sequence in the batch is used.
        device: Target device for the output tensors.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Prepared input and target tensors.
    """
    if context_length is not None:
        batch_max_length = context_length + 1  # +1 for the target item
    else:
        batch_max_length = max(len(item) for item in batch)  # No truncation, use max length in batch
        

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item = new_item[-batch_max_length:] # Keep only the last context_length items if specified
       
        # Pad sequences to max_length
        input_padded = (
            #new_item + [pad_token_id] *
            #    (batch_max_length - len(new_item))
            [pad_token_id] * (batch_max_length - len(new_item)) + new_item
        )

        target_padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )


        inputs = torch.tensor(input_padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(target_padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)

    return inputs_tensor, targets_tensor



def test_collate_fn(
    batch,
    pad_token_id=50256,
    context_length=5,
    dynamic_target_length=True,
    device="cpu"
):
    """
    Collate function for test dataloader. Prepares input and target tensors for the model.
    
    Args:
        batch: List of sequences (lists of token IDs) in the batch.
        pad_token_id: Token ID used for padding shorter sequences.
        context_length: Maximum length of the input sequence (context) for the model.
        dynamic_target_length: If True, target length is dynamic based on the longest sequence in the batch; if False, target length is fixed to 1 (predict only the last item).
        device: Target device for the output tensors.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Prepared input and target tensors.
    """
    
    assert context_length is not None, "context_length must be specified for test_collate_fn"

    if dynamic_target_length:
        # Find the longest sequence in the batch
        batch_max_length = max(len(item) for item in batch)
        target_length = max(batch_max_length - context_length, 1) # Ensure at least 1 token is predicted
    else:
        target_length = 1  # Always predict only the last item in the sequence
    
    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        #new_item = new_item[-batch_max_length:] # Keep only the last context_length items if specified

        
         # Truncate to context_length for inputs, and keep only the last target_length items for targets
        
        if dynamic_target_length:
            # If dynamic target length, keep the context_length items for inputs and the rest for targets
            if len(new_item) <= context_length:
                new_item_input = new_item[:-1]  # If the sequence is shorter than or equal to 
                                                # context_length, use all but the last item for inputs
            else:
                new_item_input = new_item[:context_length] # Keep only the first context_length items for inputs
            padded = (
                #new_item_input + [pad_token_id] *(context_length -len(new_item_input))
                [pad_token_id] *(context_length -len(new_item_input)) + new_item_input 
            )

        else:
            # Keep only the last n_input_items for inputs, and the lastes items for targets
            new_item_input = new_item[:-target_length][-context_length:] 
            # Pad sequences to context_length if needed (only for inputs, targets will be just the last item)
            padded = (
                #new_item_input + [pad_token_id] *
                #   (context_length - len(new_item_input))
                [pad_token_id] * (context_length - len(new_item_input)) + new_item_input
            )

            
        
        
        inputs = torch.tensor(padded)  # Inputs are the last n_input_items (padded if needed)
        
        if dynamic_target_length:
            if len(new_item) <= context_length:
                pre_targets = new_item[-1:]  # If the sequence is shorter than or equal to context_length, predict only the last item
            else:          
                pre_targets = new_item[context_length:] 
            
            target_padded = (
                pre_targets + [pad_token_id] *
                (target_length - len(pre_targets))
            )
            
            targets = torch.tensor(target_padded)  # Pad targets to target_length to ensure consistent shape across the batch
        else:
            targets = torch.tensor(new_item[-target_length:])  # No padding for targets, just the last item(s)
            

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    targets_tensor = torch.stack(targets_lst)

    return inputs_tensor, targets_tensor