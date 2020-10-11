import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

from codebase.tokenizerwrapper import TokenizerWrapper
from codebase.constants import SEG_ID_PAD, SEG_ID_CLS, SEG_ID_A

max_len = 64


def get_inputs(sentence, model_folder):
    """
    Gets the inputs
    :param sentence: the sentence to get the inputs for
    :return: input_ids, input_mask and segment_ids
    """
    tokenizer_wrapper = TokenizerWrapper(model_folder)
    tokenizer = tokenizer_wrapper.tokenizer

    # Tokenize sentence to list of token ids
    tokens_a = tokenizer.encode(sentence)

    # Trim text if too large
    if len(tokens_a) > max_len - 2:
        tokens_a = tokens_a[: max_len - 2]

    tokens = []
    segment_ids = []

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(SEG_ID_A)

    tokens.append(tokenizer_wrapper.SEP_ID)
    segment_ids.append(SEG_ID_A)

    tokens.append(tokenizer_wrapper.CLS_ID)
    segment_ids.append(SEG_ID_CLS)

    input_ids = tokens

    # Mask is 0 for real tokens, 0 for padding tokens
    input_mask = [0] * len(input_ids)

    # Zero-pad up to the sequence length at fornt
    if len(input_ids) < max_len:
        delta_len = max_len - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(segment_ids) == max_len

    return input_ids, input_mask, segment_ids


def get_dataloader(inputs, masks, segs, batch_num, tags=None):
    """
    Get a dataloader 
    :param batch_num: the batch num
    :param inputs: the inputs
    :param masks: the masks
    :param segs: the segs
    :param tags: the tags
    :return: a dataloader
    """
    # building tensors
    inputs = torch.tensor(inputs)
    masks = torch.tensor(masks)
    segs = torch.tensor(segs)

    # get dataset
    if tags:
        tags = torch.tensor(tags)
        data = TensorDataset(inputs, masks, segs, tags)
        dataloader = DataLoader(
            data, sampler=RandomSampler(data), batch_size=batch_num, drop_last=True
        )
    else:
        data = TensorDataset(inputs, masks, segs)
        dataloader = DataLoader(
            data, sampler=SequentialSampler(data), batch_size=batch_num
        )



    return dataloader