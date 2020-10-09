from codebase.tokenizerwrapper import TokenizerWrapper
from codebase.constants import SEG_ID_PAD, SEG_ID_CLS, SEG_ID_A

max_len = 64


def get_inputs(sentence):
    """
    Gets the inputs
    :param sentence: the sentence to get the inputs for
    :return: input_ids, input_mask and segment_ids
    """
    tokenizer_wrapper = TokenizerWrapper()
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