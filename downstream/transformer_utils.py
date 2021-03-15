import torch
import itertools

def prepare_inputs_for_transformer_encoder(raw_in, tokenizer):
    '''
    @ input:
    - raw_in: list of strings
    @ output:
    - bert_inputs: padded Tensor
    '''

    bert_inputs = []
    
    for seq in raw_in:    
        tok_seq = [tokenizer.cls_token] + tokenizer.tokenize(seq)
        bert_inputs.append(tok_seq)
        
    input_lens = [len(seq) for seq in bert_inputs]
    max_len = max(input_lens)
    bert_input_ids = [tokenizer.convert_tokens_to_ids(seq) + [tokenizer.pad_token_id] * (max_len - len(seq)) 
    for seq in bert_inputs]
    assert len(bert_input_ids[0]) == max_len
    bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.long, device=device)
    return bert_input_ids