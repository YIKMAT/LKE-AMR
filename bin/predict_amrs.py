from pathlib import Path

import penman
import torch

from spring_amr import ROOT
from spring_amr.evaluation import predict_amrs, compute_smatch
from spring_amr.penman import encode
from spring_amr.utils import instantiate_loader, instantiate_model_and_tokenizer

if __name__ == '__main__':

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description="Script to predict AMR graphs given sentences. LDC format as input.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--datasets', type=str, required=True, nargs='+',
        help="Required. One or more glob patterns to use to load amr files.")
    parser.add_argument('--checkpoint', type=str, required=True,
        help="Required. Checkpoint to restore.")
    parser.add_argument('--model', type=str, default='facebook/bart-large',
        help="Model config to use to load the model class.")
    parser.add_argument('--beam-size', type=int, default=1,
        help="Beam size.")
    parser.add_argument('--batch-size', type=int, default=1000,
        help="Batch size (as number of linearized graph tokens per batch).")
    parser.add_argument('--device', type=str, default='cuda',
        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--pred-path', type=Path, default=ROOT / 'data/tmp/inf-pred.txt',
        help="Where to write predictions.")
    parser.add_argument('--gold-path', type=Path, default=ROOT / 'data/tmp/inf-gold.txt',
        help="Where to write the gold file.")
    parser.add_argument('--use-recategorization', action='store_true',
        help="Predict using Zhang recategorization on top of our linearization (requires recategorized sentences in input).")
    parser.add_argument('--penman-linearization', action='store_true',
        help="Predict using PENMAN linearization instead of ours.")
    parser.add_argument('--use-pointer-tokens', action='store_true')
    parser.add_argument('--raw-graph', action='store_true')
    parser.add_argument('--restore-name-ops', action='store_true')
    parser.add_argument('--return-all', action='store_true')

    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer = instantiate_model_and_tokenizer(
        args.model,
        dropout=0.,
        attention_dropout=0.,
        penman_linearization=args.penman_linearization,
        use_pointer_tokens=args.use_pointer_tokens,
        raw_graph=args.raw_graph,
    )
    model.amr_mode = True
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to(device)

    #
    srl_vocab_file = "configs/srl_gold_vocab.txt"
    dep_vocab_file = "configs/dep_gold_vocab_Allen.txt"
    gold_path = args.gold_path
    pred_path = args.pred_path
    # test_dep_file = "data/AMR/OOD/dep/BIO(500)_Allen_dep.json"
    # test_dep_file = "data/AMR/OOD/dep/BIO(V3)_Allen_dep.json"
    # test_dep_file = "data/AMR/OOD/dep/TLP(V3)_Allen_dep.json"
    # test_dep_file = "data/AMR/OOD/dep/New3_Allen_dep.json"
    test_dep_file = "data/AMR/amr_2.0/amr2.0_test_dep_Allen.json"

    # test_srl_file = "data/AMR/OOD/srl/BIO(500)_Allen_srl.json"
    # test_srl_file = "data/AMR/OOD/srl/BIO(V3)_Allen_srl.json"
    # test_srl_file = "data/AMR/OOD/srl/TLP(V3)_Allen_srl.json"
    # test_srl_file = "data/AMR/OOD/srl/New3_Allen_srl.json"
    test_srl_file = "data/AMR/amr_2.0/amr2.0_test_srl.json"

    loader = instantiate_loader(
        args.datasets,
        tokenizer,
        batch_size=args.batch_size,
        evaluation=True, out=gold_path,
        use_recategorization=args.use_recategorization,
        dep_file=test_dep_file,
        dep_vocab_file=dep_vocab_file,
        srl_file=test_srl_file,
        srl_vocab_file=srl_vocab_file,
    )
    loader.device = device

    graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=args.beam_size,
        restore_name_ops=args.restore_name_ops,
        return_all=args.return_all,
    )
    if args.return_all:
        graphs = [g for gg in graphs for g in gg]

    pieces = [encode(g) for g in graphs]
    pred_path.write_text('\n\n'.join(pieces))
    #
    if not args.return_all:
        score = compute_smatch(gold_path, pred_path)
        print(f'Smatch: {score:.3f}')
