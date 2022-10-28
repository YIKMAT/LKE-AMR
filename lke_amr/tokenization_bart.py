import copy
import sys
from pathlib import Path

import penman
import regex as re
import torch
from transformers import BartTokenizer

from lke_amr import ROOT, postprocessing
from lke_amr.linearization import AMRTokens, AMRLinearizer
from lke_amr.penman import encode
import dgl
import numpy as np


class AMRBartTokenizer(BartTokenizer):

    INIT = 'Ġ'

    ADDITIONAL = [
        AMRTokens.PNTR_N,
        AMRTokens.STOP_N,
        AMRTokens.LIT_START,
        AMRTokens.LIT_END,
        AMRTokens.BACKR_SRC_N,
        AMRTokens.BACKR_TRG_N,]

    def __init__(self, *args, use_pointer_tokens=False, collapse_name_ops=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.patterns = re.compile(
            r""" ?<[a-z]+:?\d*>| ?:[^\s]+|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.linearizer = AMRLinearizer(use_pointer_tokens=use_pointer_tokens, collapse_name_ops=collapse_name_ops)
        self.use_pointer_tokens = use_pointer_tokens
        self.collapse_name_ops = collapse_name_ops
        self.recategorizations = set()
        self.modified = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_path, pred_min=5, *args, **kwargs):
        inst = super().from_pretrained(pretrained_model_path, *args, **kwargs)
        inst.init_amr_vocabulary(pred_min=pred_min)
        return inst

    def init_amr_vocabulary(self, pred_min=5):
        for tok in [self.bos_token, self.eos_token, self.pad_token, '<mask>', '<unk>']:
            ntok = self.INIT + tok
            i = self.encoder[tok]
            self.decoder[i] = ntok
            del self.encoder[tok]
            self.encoder[ntok] = i

        tokens = []
        for line in Path(ROOT/'data/vocab/predicates.txt').read_text().strip().splitlines():
            tok, count = line.split()
            if int(count) >= pred_min:
                tokens.append(tok)
                
        for tok in Path(ROOT/'data/vocab/additions.txt').read_text().strip().splitlines():
            tokens.append(tok)

        for tok in Path(ROOT/'data/vocab/recategorizations.txt').read_text().strip().splitlines():
            if not tok.startswith('_'):
                self.recategorizations.add(tok)
            tokens.append(tok)

        if self.use_pointer_tokens:
            for cnt in range(512):
                tokens.append(f"<pointer:{cnt}>")

        tokens += self.ADDITIONAL
        tokens = [self.INIT + t if t[0] not in ('_', '-') else t for t in tokens]
        tokens = [t for t in tokens if t not in self.encoder]
        self.old_enc_size = old_enc_size = len(self.encoder)
        for i, t in enumerate(tokens, start= old_enc_size):
            self.encoder[t] = i

        self.encoder = {k: i for i, (k,v) in enumerate(sorted(self.encoder.items(), key=lambda x: x[1]))}
        self.decoder = {v: k for k, v in sorted(self.encoder.items(), key=lambda x: x[1])}
        self.modified = len(tokens)
        
        self.bos_token = self.INIT + '<s>'
        self.pad_token = self.INIT + '<pad>'
        self.eos_token = self.INIT + '</s>'
        self.unk_token = self.INIT + '<unk>'

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        if token_ids_1 is None:
            return output
        return output + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def _tokenize(self, text):
        """ Tokenize a string. Modified in order to handle sentences with recategorization pointers"""
        bpe_tokens = []
        for tok_span in text.lstrip().split(' '):
            tok_span = tok_span.strip()
            recats = tok_span.rsplit('_', 1)
            if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
                bpe_tokens.extend([self.INIT + recats[0], '_' + recats[1]])
            else:
                for token in re.findall(self.pat, ' ' + tok_span):
                    token = "".join(
                        self.byte_encoder[b] for b in token.encode("utf-8")
                    )  # Maps all our bytes to unicode strings, avoiding controle tokens of the BPE (spaces in our case)
                    bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))

        return bpe_tokens

    def _tok_bpe(self, token, add_space=True):
        # if add_space:
        #     token = ' ' + token.lstrip()
        tokk = []
        tok = token.strip()
        recats = tok.rsplit('_', 1)
        if len(recats) == 2 and recats[0] in self.recategorizations and ('_' + recats[1]) in self.encoder:
            tokk.extend([self.INIT + recats[0], '_' + recats[1]])
        else:
            for tok in self.patterns.findall(' ' + token):
                tok = "".join(
                    self.byte_encoder[b] for b in tok.encode("utf-8"))
                toks = self.bpe(tok).split(' ')
                tokk.extend(toks)
        return tokk

    def _get_nodes_and_backreferences(self, graph):
        lin = self.linearizer.linearize(graph)
        linearized_nodes, backreferences = lin.nodes, lin.backreferences
        return linearized_nodes, backreferences

    def tokenize_amr(self, graph):
        linearized_nodes, backreferences = self._get_nodes_and_backreferences(graph)

        bpe_tokens = []
        bpe_backreferences = []
        counter = 0
        
        for i, (backr, tokk) in enumerate(zip(backreferences, linearized_nodes)):
            is_in_enc = self.INIT + tokk in self.encoder
            is_rel = tokk.startswith(':') and len(tokk) > 1
            is_spc = tokk.startswith('<') and tokk.endswith('>')
            is_of  = tokk.startswith(':') and tokk.endswith('-of')
            is_frame = re.match(r'.+-\d\d', tokk) is not None

            if tokk.startswith('"') and tokk.endswith('"'):
                tokk = tokk[1:-1].replace('_', ' ')
                bpe_toks = [self.INIT + AMRTokens.LIT_START]
                bpe_toks += self._tok_bpe(tokk, add_space=True)
                bpe_toks.append(self.INIT + AMRTokens.LIT_END)

            elif (is_rel or is_spc or is_frame or is_of):
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                elif is_frame:
                    bpe_toks = self._tok_bpe(tokk[:-3], add_space=True) + [tokk[-3:]]
                elif is_of:
                    rel = tokk[:-3]
                    if self.INIT + rel in self.encoder:
                        bpe_toks = [self.INIT + rel, '-of']
                    else:
                        bpe_toks = [self.INIT + ':'] + self._tok_bpe(rel[1:], add_space=True) + ['-of']
                elif is_rel:
                    bpe_toks = [self.INIT + ':'] + self._tok_bpe(tokk[1:], add_space=True)
                else:
                    raise

            else:
                if is_in_enc:
                    bpe_toks = [self.INIT + tokk]
                else:
                    bpe_toks = self._tok_bpe(tokk, add_space=True)

            bpe_tokens.append(bpe_toks)

            if i == backr:
                bpe_backr = list(range(counter, counter + len(bpe_toks)))
                counter += len(bpe_toks)
                bpe_backreferences.append(bpe_backr)
            else:
                bpe_backreferences.append(bpe_backreferences[backr][0:1])
                counter += 1               
        bpe_tokens = [b for bb in bpe_tokens for b in bb]
        bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
        bpe_backreferences = [b for bb in bpe_backreferences for b in bb]
        return bpe_tokens, bpe_token_ids, bpe_backreferences

    def batch_encode_sentences(self, sentences, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}
        batch = super().batch_encode_plus(sentences, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra


    def batch_encode_sentences_self(self, sentences, dep_tag2id, dep_tags, input_tokens, subtoken_map,srl_input_tokens, srl_token2subtoken_map,srl_tags, device=torch.device('cpu')):
        batch_sentences = []
        batch_graph = []
        batch_arg_ids = []
        batch_arg_mask = []
        batch_predicates = []



        for sample_index, input_token in enumerate(input_tokens):

            srl_tokens = " ".join(srl_input_tokens[sample_index])
            subtokens_srl = self.tokenize(srl_tokens, add_special_tokens=False)
            dep_all_tokens_one_sample = " ".join(input_tokens[sample_index])
            subtokens_dep = self.tokenize(dep_all_tokens_one_sample, add_special_tokens=False)

            batch_sentences.append(dep_all_tokens_one_sample)

            # -----------------------------------graph-------------------------------------
            dep_tag_sample = dep_tags[sample_index]
            srl_tag_sample = srl_tags[sample_index]
            subtoken_map_sample = subtoken_map[sample_index]
            graph, arg_ids, arg_mask, predicates = self.create_graph(dep_tag_sample, subtoken_map_sample, dep_tag2id, srl_tag_sample)
            batch_graph.append(graph)
            batch_arg_ids.append(arg_ids)
            batch_arg_mask.append(arg_mask)
            batch_predicates.append(predicates)


        sentences = [s for s in sentences]
        extra = {'sentences': sentences,"dep_graphs":batch_graph, "arg_ids": batch_arg_ids,"arg_mask": batch_arg_mask,"predicates": batch_predicates}
        # dep_graphs = {"dep_graphs":batch_graph}
        batch = super().batch_encode_plus(batch_sentences, return_tensors='pt', pad_to_max_length=True)
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra

    def create_graph(self, dep_tag_sample, subtoken_map_sample, dep_tag2id, srl_dict):
        graph = dgl.graph([])
        graph.set_n_initializer(dgl.init.zero_initializer)
        subtoken_map_sample = torch.tensor(subtoken_map_sample)
        num_tokens = subtoken_map_sample.size()[0]
        token_range = torch.arange(0, num_tokens, dtype=torch.int64)

        # dependency parsing trees (token -> token)
        dependency = dep_tag_sample
        graph.add_nodes(num_tokens)
        graph.ndata['unit'] = torch.zeros(num_tokens)
        graph.ndata['dtype'] = torch.zeros(num_tokens)
        root_token_ids = None
        for deprel, head_id, word_id in dependency:
            token_ids = token_range[subtoken_map_sample == (word_id - 1)]
            if head_id == 0:
                head_token_ids = []
            else:
                head_token_ids = token_range[subtoken_map_sample == (head_id - 1)]
            dep_rel_label = torch.tensor([dep_tag2id.get(deprel, 0)])
            for token_id in token_ids:
                for head_token_id in head_token_ids:
                    graph.add_edges(token_id, head_token_id, data={'dep_link': dep_rel_label,
                                                                   'dtype': torch.tensor([0])})

            # self loop
            dep_rel_label = torch.tensor([dep_tag2id['cyclic']])
            for token_id1 in token_ids:
                for token_id2 in token_ids:
                    graph.add_edges(token_id1, token_id2, data={'dep_link': dep_rel_label,
                                                                'dtype': torch.tensor([0])})

            # link roots between two adjacent sentences
            # dep_rel_label = torch.tensor([dep_tag2id['<pad>']])
            # if root_token_ids is not None and head_id == 0:
            #     for root_token_id in root_token_ids:
            #         for token_id in token_ids:
            #             graph.add_edges(token_id, root_token_id, data={'dep_link': dep_rel_label,
            #                                                            'dtype': torch.tensor([0])})
            #             graph.add_edges(root_token_id, token_id, data={'dep_link': dep_rel_label,
            #                                                            'dtype': torch.tensor([0])})
            # if head_id == 0:
            #     root_token_ids = token_ids


        # predicate and argument nodes
        # argument -> predicate
        # word -> argument & predicate
        predicates = srl_dict.keys()
        num_predicates = len(predicates)
        graph.add_nodes(num_predicates)
        node_id_offset = num_tokens
        graph.ndata['unit'][node_id_offset:] = torch.ones(num_predicates) * 1
        graph.ndata['dtype'][node_id_offset:] = torch.ones(num_predicates) * 1
        predicate2nid = {predicate: i + node_id_offset for i, predicate in enumerate(predicates)}
        arguments = set()
        for _, args in srl_dict.items():
            for arg_start, arg_end, _ in args:
                arguments.add((arg_start, arg_end))
        num_arguments = len(arguments)
        graph.add_nodes(num_arguments)
        node_id_offset += num_predicates
        graph.ndata['unit'][node_id_offset:] = torch.ones(num_arguments) * 2
        graph.ndata['dtype'][node_id_offset:] = torch.ones(num_arguments) * 2
        arg2nid = {(arg_start, arg_end): i + node_id_offset for i, (arg_start, arg_end) in enumerate(arguments)}
        for predicate in predicates:
            predicatenid = predicate2nid[predicate]
            for arg_start, arg_end, label in srl_dict[predicate]:
                argnid = arg2nid[(arg_start, arg_end)]
                graph.add_edges(argnid, predicatenid, data={'srl_link': torch.tensor([label]),
                                                            'dtype': torch.tensor([1])})
                graph.add_edges(predicatenid, argnid, data={'srl_link': torch.tensor([label]),
                                                            'dtype': torch.tensor([1])})

        for arg_start, arg_end in arguments:
            argnid = arg2nid[(arg_start, arg_end)]
            for i in range(arg_start, arg_end + 1):
                graph.add_edges(i, argnid, data={'ta_link': torch.randint(10, (1,)),
                                                 'dtype': torch.tensor([2])})
                graph.add_edges(argnid, i, data={'ta_link': torch.randint(10, (1,)),
                                                 'dtype': torch.tensor([2])})

        arg_ids = []
        arg_mask = []
        for arg_start, arg_end in arguments:
            arg_ids.append(list(range(arg_start, arg_end + 1)))
        if len(arg_ids) != 0:
            max_arg_len = max([len(arg) for arg in arg_ids])
            for i in range(len(arg_ids)):
                arg_len = len(arg_ids[i])
                if arg_len < max_arg_len:
                    arg_ids[i] += [0] * (max_arg_len - arg_len)
                arg_mask.append([1] * arg_len + [0] * (max_arg_len - arg_len))

        arg_ids = torch.tensor(arg_ids, dtype=torch.int64)
        arg_mask = torch.tensor(arg_mask, dtype=torch.int64)
        predicates = torch.tensor(list(predicates), dtype=torch.int64)

        return graph, arg_ids, arg_mask, predicates



    def linearize(self, graph):
        shift = len(self.encoder)
        tokens, token_ids, backreferences = self.tokenize_amr(graph)
        extra = {'linearized_graphs': tokens, 'graphs': graph}
        token_uni_ids = \
            [idx if i == b else b + shift for i, (idx, b) in enumerate(zip(token_ids, backreferences))]
        if token_uni_ids[-1] != (self.INIT + AMRTokens.EOS_N):
            tokens.append(self.INIT + AMRTokens.EOS_N)
            token_ids.append(self.eos_token_id)
            token_uni_ids.append(self.eos_token_id)
            backreferences.append(len(backreferences))
        return token_uni_ids, extra
        
    def batch_encode_graphs(self, graphs, device=torch.device('cpu')):
        linearized, extras = zip(*[self.linearize(g) for g in graphs])
        return self.batch_encode_graphs_from_linearized(linearized, extras, device=device)

    def batch_encode_graphs_from_linearized(self, linearized, extras=None, device=torch.device('cpu')):
        if extras is not None:
            batch_extra = {'linearized_graphs': [], 'graphs': []}
            for extra in extras:
                batch_extra['graphs'].append(extra['graphs'])
                batch_extra['linearized_graphs'].append(extra['linearized_graphs'])
        else:
            batch_extra = {}
        maxlen = 0
        batch = []
        for token_uni_ids in linearized:
            maxlen = max(len(token_uni_ids), maxlen)
            batch.append(token_uni_ids)
        batch = [x + [self.pad_token_id] * (maxlen - len(x)) for x in batch]
        batch = torch.tensor(batch).to(device)
        batch = {'decoder_input_ids': batch[:, :-1], 'lm_labels': batch[:, 1:]}
        return batch, batch_extra

    def decode_amr(self, tokens, restore_name_ops=False):
        try:
            nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        if self.use_pointer_tokens:
            nodes, backreferences = postprocessing.restore_backreferences_from_pointers(nodes)
        try:
            graph_ = graph = postprocessing.build_graph(nodes, backreferences, restore_name_ops=restore_name_ops)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes, backreferences)

class PENMANBartTokenizer(AMRBartTokenizer):

    def __init__(self, *args, raw_graph=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.linearizer = None
        self.remove_pars = False
        self.raw_graph = raw_graph

    def _tokenize_encoded_graph(self, encoded):
        linearized = re.sub(r"(\".+?\")", r' \1 ', encoded)
        pieces = []
        for piece in linearized.split():
            if piece.startswith('"') and piece.endswith('"'):
                pieces.append(piece)
            else:
                piece = piece.replace('(', ' ( ')
                piece = piece.replace(')', ' ) ')
                piece = piece.replace(':', ' :')
                piece = piece.replace('/', ' / ')
                piece = piece.strip()
                pieces.append(piece)
        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()
        linearized_nodes = [AMRTokens.BOS_N] + linearized.split(' ')
        return linearized_nodes

    def tokenize_amr(self, graph):
        if self.raw_graph:
            graph_ = copy.deepcopy(graph)
            graph_.metadata = {}
            linearized = penman.encode(graph_)
            linearized = re.sub(r"\s+", ' ', linearized)
            bpe_tokens = [self.bos_token] + self._tokenize(linearized)[:1022]
            bpe_token_ids = [self.encoder.get(b, self.unk_token_id) for b in bpe_tokens]
            bpe_backreferences = list(range(len(bpe_token_ids)))
            return bpe_tokens, bpe_token_ids, bpe_backreferences
        else:
            return super().tokenize_amr(graph)

    def _get_nodes_and_backreferences(self, graph):
        graph_ = copy.deepcopy(graph)
        graph_.metadata = {}
        linearized = penman.encode(graph_)
        linearized_nodes = self._tokenize_encoded_graph(linearized)

        if self.use_pointer_tokens:
            remap = {}
            for i in range(1, len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes[i-1]
                if nxt == '/':
                    remap[lst] = f'<pointer:{len(remap)}>'
            i = 1
            linearized_nodes_ = [linearized_nodes[0]]
            while i < (len(linearized_nodes)):
                nxt = linearized_nodes[i]
                lst = linearized_nodes_[-1]
                if nxt in remap:
                    if lst == '(' and linearized_nodes[i+1] == '/':
                        nxt = remap[nxt]
                        i += 1
                    elif lst.startswith(':'):
                        nxt = remap[nxt]
                linearized_nodes_.append(nxt)
                i += 1
            linearized_nodes = linearized_nodes_
            if self.remove_pars:
                linearized_nodes = [n for n in linearized_nodes if n != '(']
        backreferences = list(range(len(linearized_nodes)))
        return linearized_nodes, backreferences

    def _classify(self, node):
        if not isinstance(node, str):
            return "CONST"
        elif node == 'i':
            return "I"
        elif re.match(r'^[a-z]\d*$', node) is not None:
            return "VAR"
        elif node[0].isdigit():
            return "CONST"
        elif node.startswith('"') and node.endswith('"'):
            return "CONST"
        elif node in ('+', '-'):
            return "CONST"
        elif node == ':mode':
            return 'MODE'
        elif node.startswith(':'):
            return "EDGE"
        elif node in ['/', '(', ')']:
            return node
        elif node[0].isalpha():
            for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\'):
                if char in node:
                    return "CONST"
            return "INST"
        else:
            return 'CONST'

    def _fix_and_make_graph(self, nodes):

        nodes_ = []
        for n in nodes:
            if isinstance(n, str):
                if n.startswith('<') and n.endswith('>') and (not n.startswith('<pointer:')):
                    pass
                else:
                    nodes_.append(n)
            else:
                nodes_.append(n)
        nodes = nodes_

        if self.use_pointer_tokens:

            i = 0
            nodes_ = []
            while i < len(nodes):
                nxt = nodes[i]
                pst = None
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    e = nxt.find('>')
                    if e != len(nxt) -1:
                        pst = nxt[e+1:]
                        nxt = nxt[:e+1]
                    nodes_.append(nxt)
                    if pst is not None:
                        nodes_.append(pst)
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

            i = 1
            nodes_ = [nodes[0]]
            while i < len(nodes):
                nxt = nodes[i]
                if isinstance(nxt, str) and nxt.startswith('<pointer:'):
                    nxt = 'z' + nxt[9:-1]
                    fol = nodes[i+1]
                    # is not expansion
                    if isinstance(fol, str) and (fol.startswith(':') or (fol == ')')):
                        nodes_.append(nxt)
                    else:
                        if self.remove_pars:
                            nodes_.append('(')
                        else:
                            if nodes_[-1] != '(':
                                nodes_.append('(')
                                #pass
                        nodes_.append(nxt)
                        nodes_.append('/')
                else:
                    nodes_.append(nxt)
                i += 1
            nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes) - 1):
            if nodes[i] == ':':
                nodes_.append(nodes[i] + nodes[i+1])
                i += 2
                last = False
            else:
                nodes_.append(nodes[i])
                i += 1
                last = True
        if last:
            nodes_.append(nodes[-1])
        nodes = nodes_

        i = 0
        nodes_ = []
        while i < (len(nodes)):
            if i < 2:
                nodes_.append(nodes[i])
                i += 1
            elif nodes_[-2] == '/' and nodes[i] == '/':
                i += 2
            else:
                nodes_.append(nodes[i])
                i += 1
        nodes = nodes_

        i = 0
        newvars = 0
        variables = set()
        remap = {}
        nodes_ = []
        while i < (len(nodes)):

            next = nodes[i]

            if next == '/':
                last = nodes_[-1]
                if last in variables:
                    last_remap = f"z{newvars+1000}"
                    newvars += 1
                    nodes_[-1] = last_remap
                    remap[last] = last_remap
                variables.add(last)
                nodes_.append(next)

            elif self._classify(next) == 'VAR' and next in remap and (i < len(nodes) - 1) and nodes[i+1] != '/':
                next = remap[next]
                nodes_.append(next)

            else:
                nodes_.append(next)

            i += 1

        nodes = nodes_
        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if nodes[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in nodes:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        nodes = pieces_ + [')'] * (open_cnt - closed_cnt)

        pieces = []
        for piece in nodes:
            if not pieces:
                pieces.append('(')
            else:
                piece = str(piece)
                if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                    piece = '"' + piece.replace('"', '') + '"'

                prev = self._classify(pieces[-1])
                next = self._classify(piece)

                if next == 'CONST':
                    quote = False
                    for char in (',', ':', '/', '(', ')', '.', '!', '?', '\\', '_', '='):
                        if char in piece:
                            quote = True
                            break
                    if quote:
                        piece = '"' + piece.strip('"') + '"'

                if  prev == '(':
                    if next in ('VAR', 'I'):
                        pieces.append(piece)
                elif prev == ')':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'VAR':
                    if next in ('/', 'EDGE', 'MODE', ')'):
                        pieces.append(piece)
                elif prev == '/':
                    if next in ('INST', 'I'):
                        pieces.append(piece)
                elif prev == 'INST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'I':
                    if next in ('/', ')', 'EDGE', 'MODE'):
                        pieces.append(piece)
                elif prev == 'EDGE':
                    if next in ('(', 'VAR', 'CONST', 'I'):
                        pieces.append(piece)
                    elif next == ')':
                        pieces[-1] = piece
                    elif next in ('EDGE', 'MODE'):
                        pieces[-1] = piece
                elif prev == 'MODE':
                    if next == 'INST':
                        pieces.append(piece)
                elif prev == 'CONST':
                    if next in (')', 'EDGE', 'MODE'):
                        pieces.append(piece)

        pieces_ = []
        open_cnt = 0
        closed_cnt = 0
        if pieces[0] != '(':
            pieces_.append('(')
            open_cnt += 1
        for p in pieces:
            if p == '(':
                open_cnt += 1
            elif p == ')':
                closed_cnt += 1
            pieces_.append(p)
            if open_cnt == closed_cnt:
                break
        pieces = pieces_ + [')'] * (open_cnt - closed_cnt)

        linearized = re.sub(r'\s+', ' ', ' '.join(pieces)).strip()

        """
        line = linearized
        # make sure parentheses match
        # copied from https://github.com/RikVN/AMR/blob/master/restoreAMR/restore_amr.py
        open_count = 0
        close_count = 0
        for i, c in enumerate(line):
            if c == '(':
                open_count += 1
            elif c == ')':
                close_count += 1
            if open_count == close_count and open_count > 0:
                line = line[:i].strip()
                break
        old_line = line
        while True:
            open_count = len(re.findall(r'\(', line))
            close_count = len(re.findall(r'\)', line))
            if open_count > close_count:
                line += ')' * (open_count - close_count)
            elif close_count > open_count:
                for i in range(close_count - open_count):
                    line = line.rstrip(')')
                    line = line.rstrip(' ')
            if old_line == line:
                break
            old_line = line
        """

        graph = penman.decode(linearized + ' ')
        triples = []
        newvars = 2000
        for triple in graph.triples:
            x, rel, y = triple
            if x is None:
                pass
            elif rel == ':instance' and y is None:
                triples.append(penman.Triple(x, rel, 'thing'))
            elif y is None:
                var = f'z{newvars}'
                newvars += 1
                triples.append(penman.Triple(x, rel, var))
                triples.append(penman.Triple(var, ':instance', 'thing'))
            else:
                triples.append(triple)
        graph = penman.Graph(triples)
        linearized = encode(graph)

        def fix_text(linearized=linearized):
            n = 0
            def _repl1(match):
                nonlocal n
                out = match.group(1) + match.group(2) + str(3000 + n) + ' / ' + match.group(2) + match.group(3)
                n += 1
                return out
            linearized = re.sub(r'(\(\s?)([a-z])([^\/:\)]+[:\)])', _repl1, linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            def _repl2(match):
                return match.group(1)
            linearized = re.sub(r'(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)', _repl2,
                                linearized,
                                flags=re.IGNORECASE | re.MULTILINE)

            # adds a ':' to args w/o it
            linearized = re.sub(r'([^:])(ARG)', r'\1 :\2', linearized)

            # removes edges with no node
            # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

            return linearized

        linearized = fix_text(linearized)

        g = penman.decode(linearized)
        return g

    def decode_amr(self, tokens, restore_name_ops=None):
        try:
            if self.raw_graph:
                nodes = self._tokenize_encoded_graph(self.decode(tokens))
                backreferences = list(range(len(nodes)))
            else:
                nodes, backreferences = postprocessing.decode_into_node_and_backreferences(tokens, self)
            nodes_ = nodes
        except Exception as e:
            print('Decoding failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph_ = graph = self._fix_and_make_graph(nodes)
            if self.collapse_name_ops:
                graph_ = graph = postprocessing._split_name_ops(graph)
        except Exception as e:
            print('Building failure:', file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(e, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (None, None)
        try:
            graph, status = postprocessing.connect_graph_if_not_connected(graph)
            if status == postprocessing.ParsedStatus.BACKOFF:
                print('Reconnection 1 failure:')
                print(nodes, file=sys.stderr)
                print(backreferences, file=sys.stderr)
                print(graph_, file=sys.stderr)
            return graph, status, (nodes_, backreferences)
        except Exception as e:
            print('Reconnction 2 failure:', file=sys.stderr)
            print(e, file=sys.stderr)
            print(nodes, file=sys.stderr)
            print(backreferences, file=sys.stderr)
            print(graph_, file=sys.stderr)
            return postprocessing.BACKOFF, postprocessing.ParsedStatus.BACKOFF, (nodes_, backreferences)
