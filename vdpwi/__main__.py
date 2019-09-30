import argparse
import logging
import os
import pprint
import random

import numpy as np
import torch
import torch.optim as optim

from common.dataset import DatasetFactory
from common.evaluation import EvaluatorFactory
from common.train import TrainerFactory
from utils.serialization import load_checkpoint
from .model import VDPWIModel

import pickle
from .tree_loader import Tree

def evaluate_dataset(split_name, dataset_cls, model, embedding, loader, batch_size, device, run_label, result_loc, dataset):
    saved_model_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, loader, batch_size, device)
    if dataset == 'sts2014' and split_name == 'test':
        scores, metric_names = saved_model_evaluator.get_sts_test_scores()
    else:
        scores, metric_names = saved_model_evaluator.get_scores()
    logger.info('Evaluation metrics for {}'.format(split_name))
    logger.info('\t'.join([' '] + metric_names))
    logger.info('\t'.join([split_name] + list(map(str, scores))))

    file = open(os.path.join(result_loc, dataset+"_"+run_label), "a+")
    file.write('Evaluation metrics for {}'.format(split_name))
    file.write("\n")
    file.write('\t'.join([' '] + metric_names))
    file.write("\n")
    file.write('\t'.join([split_name] + list(map(str, scores))))
    file.write("\n")
    file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of VDPWI')
    parser.add_argument('--model_outfile', help='file to save final model')
    parser.add_argument('--dataset', help='dataset to use, one of [sick, msrvid, trecqa, wikiqa]', default='sick')
    parser.add_argument('--word-vectors-dir', help='word vectors directory', default='../data/embeddings/GloVe')
    parser.add_argument('--word-vectors-file', help='word vectors filename', default='glove.840B.300d.txt')
    parser.add_argument('--word-vectors-dim', type=int, default=300,
                        help='number of dimensions of word vectors (default: 300)')
    parser.add_argument('--skip-training', help='will load pre-trained model', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='GPU device, -1 for CPU (default: 0)')
    parser.add_argument('--sparse-features', action='store_true', default=False, help='use sparse features (default: false)')
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer to use: adam, sgd, or rmsprop (default: adam)')
    parser.add_argument('--lr', type=float, default=5E-4, help='learning rate (default: 0.001)')
    parser.add_argument('--lr-reduce-factor', type=float, default=0.3, help='learning rate reduce factor after plateau (default: 0.3)')
    parser.add_argument('--patience', type=float, default=2, help='learning rate patience after seeing plateau (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.1, help='momentum (default: 0.1)')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Adam epsilon (default: 1e-8)')
    parser.add_argument('--log-interval', type=int, default=10, help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--regularization', type=float, default=1E-5, help='Regularization for the optimizer (default: 0.00001)')
    parser.add_argument('--hidden-units', type=int, default=250, help='number of hidden units in the RNN')
    parser.add_argument('--seed', type=int, default=620, help='random seed (default: 1)')
    parser.add_argument('--tensorboard', action='store_true', default=False, help='use TensorBoard to visualize training (default: false)')
    parser.add_argument('--run-label', type=str, help='label to describe run')

    parser.add_argument('--clip-norm', type=float, default=50)
    parser.add_argument('--decay', type=float, default=0.95)
    parser.add_argument('--res-fmaps', type=int, default=32)
    parser.add_argument('--res-layers', type=int, default=16)
    parser.add_argument('--rnn-hidden-dim', type=int, default=250)
    parser.add_argument('--fileID', type=str, default='0')
    parser.add_argument('--treeLSTMScale', type=float, default=1.0)
    parser.add_argument("--resultLoc", type=str, default="result_file/")
    parser.add_argument("--treeFile", type=str, default="toks_tree_quora.pkl")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    torch.cuda.set_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != -1:
        torch.cuda.manual_seed(args.seed)

    # logging setup
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(pprint.pformat(vars(args)))

    dataset_cls, embedding, train_loader, test_loader, dev_loader \
        = DatasetFactory.get_dataset(args.dataset, args.word_vectors_dir, args.word_vectors_file, args.batch_size, args.device)

    model_config = {
        'rnn_hidden_dim': args.rnn_hidden_dim,
        'n_labels': dataset_cls.NUM_CLASSES,
        'device': device,
        'res_layers': args.res_layers,
        'res_fmaps': args.res_fmaps,
        'treeLSTMScale': args.treeLSTMScale
    }

    tree_file = pickle.load(open(args.treeFile, 'rb'))

    model = VDPWIModel(args.word_vectors_dim, model_config, tree_file)
    model.to(device)
    embedding = embedding.to(device)

    optimizer = None
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.regularization, eps=args.epsilon)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.regularization)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, momentum=args.momentum, alpha=args.decay, 
            weight_decay=args.regularization)
    else:
        raise ValueError('optimizer not recognized: it should be one of adam, sgd, or rmsprop')

    train_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, train_loader, args.batch_size, args.device)
    test_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, test_loader, args.batch_size, args.device)
    dev_evaluator = EvaluatorFactory.get_evaluator(dataset_cls, model, embedding, dev_loader, args.batch_size, args.device)

    trainer_config = {
        'optimizer': optimizer,
        'batch_size': args.batch_size,
        'log_interval': args.log_interval,
        'model_outfile': args.model_outfile,
        'lr_reduce_factor': args.lr_reduce_factor,
        'patience': args.patience,
        'tensorboard': args.tensorboard,
        'run_label': args.run_label,
        'logger': logger,
        'clip_norm': args.clip_norm,
        'logOutput': args.resultLoc + args.fileID
    }
    
    trainer = TrainerFactory.get_trainer(args.dataset, model, embedding, train_loader, trainer_config, train_evaluator, test_evaluator, dev_evaluator)

    if not args.skip_training:
        total_params = 0
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        trainer.train(args.epochs)

    _, _, state_dict, _, _ = load_checkpoint(args.model_outfile)

    for k, tensor in state_dict.items():
        state_dict[k] = tensor.to(device)

    model.load_state_dict(state_dict)

    if args.skip_training:
        total_params = 0
        trainer = TrainerFactory.get_trainer(args.dataset, model, embedding, train_loader, trainer_config,
                                             train_evaluator, test_evaluator, dev_evaluator)
        for param in model.parameters():
            size = [s for s in param.size()]
            total_params += np.prod(size)
        logger.info('Total number of parameters: %s', total_params)
        trainer.train(args.epochs)

        _, _, state_dict, _, _ = load_checkpoint(args.model_outfile)

        for k, tensor in state_dict.items():
            state_dict[k] = tensor.to(device)

        model.load_state_dict(state_dict)

    if dev_loader:
        evaluate_dataset('dev', dataset_cls, model, embedding, dev_loader, args.batch_size, args.device, args.fileID, args.resultLoc, args.dataset)
    evaluate_dataset('test', dataset_cls, model, embedding, test_loader, args.batch_size, args.device, args.fileID, args.resultLoc, args.dataset)
