import torch

def collate_fn(
    batch,
    pad_token_id=50256,
    context_length=None,
):
    """Collate function for training dataloader. Prepares input and target tensors for the model.   
    Args:
        batch: List of sequences (lists of token IDs) in the batch.
        pad_token_id: Token ID used for padding shorter sequences.
        context_length: Maximum length of the input sequence (context) for the model. If None, no truncation is applied and the longest sequence in the batch is used.
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

        # target_padded = (
        #     new_item + [pad_token_id] *
        #     (batch_max_length - len(new_item))
        # )


        inputs = torch.tensor(input_padded[:-1])  # Truncate the last token for inputs
        targets = new_item[-1]#torch.tensor(target_padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        # mask = targets == pad_token_id
        # indices = torch.nonzero(mask).squeeze()
        # if indices.numel() > 1:
        #     targets[indices[1:]] = ignore_index
        

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst)
    #targets_tensor = torch.stack(targets_lst)
    targets_tensor = torch.tensor(targets_lst)

    return inputs_tensor, targets_tensor



if __name__ == "__main__":

    PAD = -1
    batch = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [10, 20, 30, 40,],
            [7, 8, 9, 11, 12, 13, 14, 15],
        ]

    print("----------- CONTEXT_LEN 3 -----------\n")
    inputs, targets = collate_fn(batch, pad_token_id=PAD, context_length=3)
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)

    print("----------- CONTEXT_LEN 10 -----------\n")
    inputs, targets = collate_fn(batch, pad_token_id=PAD, context_length=10)
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)