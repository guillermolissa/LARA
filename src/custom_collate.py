from __future__ import annotations


import torch

IGNORE_INDEX = -100

def collate_fn(
    batch,
    pad_token_id: int = 0,
    right_pad: bool = True,
    context_length: int | None = None
):
    """Collate function for training dataloader. Prepares input and target tensors for the model.   
    Args:
        batch: List of sequences (lists of token IDs) in the batch.
        pad_token_id: Token ID used for padding shorter sequences.
        right_pad: If True, pad sequences on the right; otherwise, pad on the left.
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
        if right_pad:
            input_padded = (
                new_item[:-1] + [pad_token_id] *
                (batch_max_length - len(new_item[:-1]))
            )
        else:
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

def collate_next_item_fn(batch, pad_token_id: int = 0, context_length: int | None = None):
    """
    Args:
        batch:          list of token-id lists (one per session).
        pad_token_id:   id used to right-pad ``inputs``.
        context_length: max number of *input* tokens. The last
                        ``context_length + 1`` items of each session are kept
                        (the +1 leaves a target for the final context item).

    Returns:
        inputs:  (B, T) LongTensor, right-padded.
        targets: (B, T) LongTensor, ``inputs`` shifted left by one; pad /
                 non-predictable positions are ``IGNORE_INDEX``.
        lengths: (B,) LongTensor, real token count in each ``inputs`` row.
    """
    if context_length is not None:
        cap = context_length + 1
        batch = [seq[-cap:] for seq in batch]

    width = max((len(seq) for seq in batch), default=2)
    width = max(width - 1, 1)  # number of input positions

    inputs, targets, lengths = [], [], []
    for seq in batch:
        x = list(seq[:-1])
        y = list(seq[1:])
        n = len(x)
        pad = width - n
        inputs.append(x + [pad_token_id] * pad)
        targets.append(y + [IGNORE_INDEX] * pad)
        lengths.append(max(n, 1))

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(lengths, dtype=torch.long),
    )







if __name__ == "__main__":

    PAD = -1
    batch = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [10, 20, 30, 40,],
            [7, 8, 9, 11, 12, 13, 14, 15],
        ]

    print("----------- CONTEXT_LEN 3 RIGHT_PAD -----------\n")
    inputs, targets = collate_fn(batch, pad_token_id=PAD, right_pad=True, context_length=3)
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)

    print("\n----------- CONTEXT_LEN 10 RIGHT_PAD -----------\n")
    inputs, targets = collate_fn(batch, pad_token_id=PAD, right_pad=True, context_length=10)
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)


    print("----------- CONTEXT_LEN 3 LEFT_PAD -----------\n")
    inputs, targets = collate_fn(batch, pad_token_id=PAD, right_pad=False, context_length=3)
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)

    print("\n----------- CONTEXT_LEN 10 LEFT_PAD -----------\n")
    inputs, targets = collate_fn(batch, pad_token_id=PAD, right_pad=False, context_length=10)
    print("Inputs:")
    print(inputs)
    print("Targets:")
    print(targets)


    print("\n----------- NEXT ITEM CONTEXT_LEN 5 -----------\n")
    demo = [
            [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            [31, 32, 33, 34],
            [7, 8, 9],
        ]
    x, y, ln = collate_next_item_fn(demo, pad_token_id=PAD, context_length=5)
    print("inputs\n", x)
    print("targets\n", y)
    print("lengths", ln)
