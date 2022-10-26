import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from spring_amr.IO import read_raw_amr_data
import json

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['lm_labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'lm_labels': lm_labels}
    return x, y

class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        dep_file=None,
        dep_vocab_file=None,
        srl_file=None,
        srl_vocab_file=None,
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.device = device
        graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.all_dep_data = read_depparse_features_Allen(dep_file)
        self.all_subtoken_map, self.all_token_list = get_subtoken_map_Allen(dep_file, tokenizer)
        self.dep_tag2id = get_tag_vocab(dep_vocab_file)
        self.srl_tag2id = get_tag_vocab(srl_vocab_file)
        self.all_token2subtoken_map, self.srl_all_token_list = get_token2subtoken_map(srl_file, tokenizer)
        self.all_srl_data = read_srl_2_subtoken(srl_file,self.all_token2subtoken_map)
        self.all_srl_data = transform2srl_dict(self.all_srl_data, self.srl_tag2id)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.remove_longer_than = remove_longer_than
        for index, g in enumerate(graphs):
            l, e = self.tokenizer.linearize(g)
            try:
                self.tokenizer.batch_encode_sentences([g.metadata['snt']])
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')

            self.sentences.append(g.metadata['snt'])
            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        # ------dep
        sample['dep_tags'] = self.all_dep_data[idx]
        sample['subtoken_map'] = self.all_subtoken_map[idx]
        sample['token_list'] = self.all_token_list[idx]
        # ------srl
        sample['srl_tags'] = self.all_srl_data[idx]
        sample['token2subtoken_map'] = self.all_token2subtoken_map[idx]
        sample['srl_token_list'] = self.srl_all_token_list[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['linearized_graphs_ids'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        dep_tags = [s['dep_tags'] for s in samples]
        input_tokens = [s['token_list'] for s in samples]
        subtoken_map = [s['subtoken_map'] for s in samples]
        srl_input_tokens = [s['srl_token_list'] for s in samples]
        srl_token2subtoken_map = [s['token2subtoken_map'] for s in samples]
        srl_tags = [s['srl_tags'] for s in samples]
        # x, extra = self.tokenizer.batch_encode_sentences(x, device=device)
        x, extra = self.tokenizer.batch_encode_sentences_self(x, self.dep_tag2id, dep_tags, input_tokens, subtoken_map, srl_input_tokens, srl_token2subtoken_map,srl_tags, device=device)
        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)
        else:
            y = None
        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra
    
class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()



def read_json(input_tag_file):
    input_tag_data = []
    with open(input_tag_file, "r", encoding='utf-8') as reader:
        for line in reader:
            input_tag_data.append(json.loads(line))
    return input_tag_data

# ---------------------dep-------------------------
def read_depparse_features(data_path):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    dependencies = []

    for example in examples:
        depparse_features = example['dep_parse']
        dependency_features = []
        for sent_feature in depparse_features:
            temp_dependency = []
            for feature in sent_feature:
                word_id = feature['id']
                head_id = feature['head_id']
                deprel = feature['deprel']
                temp_dependency.append([deprel, head_id, word_id])

            dependency_features.append(temp_dependency)
        dependencies.append(dependency_features)

    return dependencies


def get_subtoken_map(data_path, tokenizer):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    all_subtoken_map = []
    all_token_list = []

    for example in examples:
        token_list = example['all_token_list']
        subtoken_map = []
        word_idx = -1

        for sent_tokens in token_list:

            for token in sent_tokens:
                word_idx += 1
                subtokens = tokenizer.tokenize(token, add_special_tokens=False)
                for sidx, subtoken in enumerate(subtokens):
                    subtoken_map.append(word_idx)

        all_subtoken_map.append(subtoken_map)
        all_token_list.append(token_list)

    return all_subtoken_map, all_token_list


def get_tag_vocab(data_path):
    tag2id = {}
    with open(data_path) as f:
        for line in f.readlines():
            tag, idx = line.strip().split(" ")
            tag2id[tag] = int(idx)
    return tag2id

# ------------------dep Allen------------------------
def read_depparse_features_Allen(data_path):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    dependencies = []

    for example in examples:
        depparse_features = example['dep_parse']
        dependency_features = []
        for sent_feature in depparse_features:
            word_id = sent_feature['id']
            head_id = sent_feature['head_id']
            deprel = sent_feature['deprel']
            dependency_features.append([deprel, head_id, word_id])
        dependencies.append(dependency_features)

    return dependencies


def get_subtoken_map_Allen(data_path, tokenizer):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    all_subtoken_map = []
    all_token_list = []

    for example in examples:
        token_list = example['all_token_list']
        subtoken_map = []
        word_idx = -1

        for token in token_list:
            word_idx += 1
            subtokens = tokenizer.tokenize(token, add_special_tokens=False)
            for sidx, subtoken in enumerate(subtokens):
                subtoken_map.append(word_idx)

        all_subtoken_map.append(subtoken_map)
        all_token_list.append(token_list)

    return all_subtoken_map, all_token_list


# --------------------srl-----------------------
def read_srl_2_subtoken(data_path, all_token2subtoken_map):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    all_srl_list = []

    for index, example in enumerate(examples):
        srl_features = example['srl']
        srl_features_subtokens = []
        token2subtoken_map = all_token2subtoken_map[index]
        for predicate, arg_start, arg_end, label in srl_features:
            pred_sub = token2subtoken_map[predicate][0]
            if arg_start == arg_end:
                arg_subs = token2subtoken_map[arg_start]
                arg_sub_start = arg_subs[0]
                arg_sub_end = arg_subs[-1]
                srl_features_subtokens.append([pred_sub, arg_sub_start, arg_sub_end, label])
                continue
            arg_sub_start = token2subtoken_map[arg_start][0]
            arg_sub_end = token2subtoken_map[arg_end][-1]
            srl_features_subtokens.append([pred_sub, arg_sub_start, arg_sub_end, label])
        all_srl_list.append(srl_features_subtokens)

    return all_srl_list

# def read_srl_features(self, data_path):
#     with open(data_path, "r") as f:
#         examples = [json.loads(jsonline) for jsonline in f.readlines()]
#
#     all_srl_list = []
#
#
#     for example in examples:
#         srl_features = example['srl']
#         srl_dict = {}
#         args = set()
#         for predicate, arg_start, arg_end, label in srl_features:
#             args.add((arg_start, arg_end))
#             label = self.srl_tag2id.get(label, 0)
#             try:
#                 srl_dict[predicate].append((arg_start, arg_end, label))
#             except KeyError:
#                 srl_dict[predicate] = [(arg_start, arg_end, label)]
#         all_srl_list.append(srl_dict)
#
#     return all_srl_list

def get_token2subtoken_map(data_path, tokenizer):
    with open(data_path, "r") as f:
        examples = [json.loads(jsonline) for jsonline in f.readlines()]

    all_token2subtoken_map = []
    all_token_list = []

    for example in examples:
        token_list = example['tokens']
        token2subtoken_map = []
        sub_word_idx = -1

        for sent_token in token_list:
            subtoken_list =[]
            subtokens = tokenizer.tokenize(sent_token, add_special_tokens=False)
            for subtoken in subtokens:
                sub_word_idx +=1
                subtoken_list.append(sub_word_idx)
            token2subtoken_map.append(subtoken_list)

        all_token2subtoken_map.append(token2subtoken_map)
        all_token_list.append(token_list)

    return all_token2subtoken_map, all_token_list


def transform2srl_dict(all_srl_data, srl_tag2id):
    all_srl_list = []

    for srl_features in all_srl_data:
        srl_dict = {}
        args = set()
        for predicate, arg_start, arg_end, label in srl_features:
            args.add((arg_start, arg_end))
            label = srl_tag2id.get(label, 0)
            try:
                srl_dict[predicate].append((arg_start, arg_end, label))
            except KeyError:
                srl_dict[predicate] = [(arg_start, arg_end, label)]
        all_srl_list.append(srl_dict)

    return all_srl_list





