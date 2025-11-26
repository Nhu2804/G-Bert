from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import dill
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
# FIX NUMPY IMPORT ISSUE - THÊM ĐOẠN NÀY
import sys
try:
    # For numpy >= 1.25
    from numpy.core import numeric as _numeric
    sys.modules['numpy._core.numeric'] = _numeric
    sys.modules['numpy._core'] = sys.modules['numpy.core']
except ImportError:
    # For older numpy versions
    pass

# THÊM HÀM SAFE PICKLE LOAD
def safe_pickle_load(filepath):
    """Safely load pickle files with compatibility handling"""
    import pickle
    import pandas as pd
    
    try:
        # Thử load thông thường trước
        return pd.read_pickle(filepath)
    except Exception as e:
        print(f"pd.read_pickle failed for {filepath}, trying alternatives...")
        
        # Thử load với pickle trực tiếp
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            print(f"✅ Loaded {filepath} with direct pickle")
            return data
        except Exception as e2:
            print(f"Direct pickle failed: {e2}")
            
            # Thử với encoding latin1 (cho pickle cũ)
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f, encoding='latin1')
                print(f"✅ Loaded {filepath} with latin1 encoding")
                return data
            except Exception as e3:
                print(f"All methods failed: {e3}")
                raise e3
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from tensorboardX import SummaryWriter

from utils import metric_report, t2n, get_n_params
from config import BertConfig
from predictive_models import GBERT_Predict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.proc_voc = self.add_vocab(os.path.join(data_dir, 'proc-vocab.txt'))  # ĐỔI: rx_voc → proc_voc
        self.dx_voc = self.add_vocab(os.path.join(data_dir, 'dx-vocab.txt'))

        # code only in multi-visit data
        self.proc_voc_multi = Voc()  # ĐỔI: rx_voc_multi → proc_voc_multi
        self.dx_voc_multi = Voc()
        with open(os.path.join(data_dir, 'proc-vocab-multi.txt'), 'r') as fin:  # ĐỔI: rx-vocab-multi → proc-vocab-multi
            for code in fin:
                self.proc_voc_multi.add_sentence([code.rstrip('\n')])  # ĐỔI: rx_voc_multi → proc_voc_multi
        with open(os.path.join(data_dir, 'dx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.dx_voc_multi.add_sentence([code.rstrip('\n')])

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens


class EHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            records = {}
            for subject_id in data['SUBJECT_ID'].unique():
                item_df = data[data['SUBJECT_ID'] == subject_id]
                patient = []
                for _, row in item_df.iterrows():
                    admission = [list(row['ICD9_CODE']), list(row['PROC_CODE'])]  # ĐỔI: ATC4 → PROC_CODE
                    patient.append(admission)
                if len(patient) < 2:
                    continue
                records[subject_id] = patient
            return records

        self.records = transform_data(data_pd)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        cur_id = self.sample_counter
        self.sample_counter += 1
        subject_id = list(self.records.keys())[item]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (2*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_proc_tokens = []  # (adm-1, l)  # ĐỔI: output_rx_tokens → output_proc_tokens

        for idx, adm in enumerate(self.records[subject_id]):
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))
            # output_proc_tokens.append(list(adm[1]))  # ĐỔI: output_rx_tokens → output_proc_tokens

            if idx != 0:
                output_proc_tokens.append(list(adm[1]))  # ĐỔI: output_rx_tokens → output_proc_tokens
                output_dx_tokens.append(list(adm[0]))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_proc_labels = []  # (adm-1, proc_voc_size)  # ĐỔI: output_rx_labels → output_proc_labels

        dx_voc_size = len(self.tokenizer.dx_voc_multi.word2idx)
        proc_voc_size = len(self.tokenizer.proc_voc_multi.word2idx)  # ĐỔI: rx_voc_size → proc_voc_size
        for tokens in output_dx_tokens:
            tmp_labels = np.zeros(dx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.dx_voc_multi.word2idx[x], tokens))] = 1
            output_dx_labels.append(tmp_labels)

        for tokens in output_proc_tokens:  # ĐỔI: output_rx_tokens → output_proc_tokens
            tmp_labels = np.zeros(proc_voc_size)  # ĐỔI: rx_voc_size → proc_voc_size
            tmp_labels[list(
                map(lambda x: self.tokenizer.proc_voc_multi.word2idx[x], tokens))] = 1  # ĐỔI: rx_voc_multi → proc_voc_multi
            output_proc_labels.append(tmp_labels)  # ĐỔI: output_rx_labels → output_proc_labels

        if cur_id < 5:
            logger.info("*** Example ***")
            logger.info("subject_id: %s" % subject_id)
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        assert len(input_ids) == (self.seq_len *
                                  2 * len(self.records[subject_id]))
        assert len(output_dx_labels) == (len(self.records[subject_id]) - 1)
        # assert len(output_proc_labels) == len(self.records[subject_id])-1  # ĐỔI: output_rx_labels → output_proc_labels

        cur_tensors = (torch.tensor(input_ids).view(-1, self.seq_len),
                       torch.tensor(output_dx_labels, dtype=torch.float),
                       torch.tensor(output_proc_labels, dtype=torch.float))  # ĐỔI: output_rx_labels → output_proc_labels

        return cur_tensors


def load_dataset(args):
    data_dir = args.data_dir
    max_seq_len = args.max_seq_length

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data = safe_pickle_load(os.path.join(data_dir, 'data-multi-visit.pkl'))
    
    # DEBUG: Kiểm tra data
    print(f"DEBUG: data shape = {data.shape}")
    print(f"DEBUG: data columns = {data.columns.tolist()}")
    print(f"DEBUG: unique SUBJECT_IDs = {data['SUBJECT_ID'].nunique()}")

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'train-id.txt'),
                os.path.join(data_dir, 'eval-id.txt'),
                os.path.join(data_dir, 'test-id.txt')]

    def load_ids(data, file_name):
        """
        :param data: multi-visit data
        :param file_name:
        :return: raw data form
        """
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(int(line.rstrip('\n')))
        return data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)

    # DEBUG: Kiểm tra từng dataset
    datasets = []
    for i, file_name in enumerate(ids_file):
        subset_data = load_ids(data, file_name)
        dataset = EHRDataset(subset_data, tokenizer, max_seq_len)
        datasets.append(dataset)
        print(f"DEBUG: {['train', 'eval', 'test'][i]} - data shape: {subset_data.shape}, dataset size: {len(dataset)}")
        
        # DEBUG chi tiết cho train dataset
        if i == 0 and len(dataset) == 0:
            print("❌ TRAIN DATASET EMPTY - Investigating...")
            print(f"DEBUG: File {file_name} has {len(open(file_name).readlines())} lines")
            print(f"DEBUG: Filtered data has {subset_data.shape[0]} rows")
            print(f"DEBUG: Unique SUBJECT_IDs in filtered data: {subset_data['SUBJECT_ID'].nunique()}")
            
            # Kiểm tra transform_data
            test_records = {}
            for subject_id in subset_data['SUBJECT_ID'].unique()[:5]:  # Check first 5
                item_df = subset_data[subset_data['SUBJECT_ID'] == subject_id]
                patient = []
                for _, row in item_df.iterrows():
                    admission = [list(row['ICD9_CODE']), list(row['PROC_CODE'])]
                    patient.append(admission)
                test_records[subject_id] = patient
                print(f"DEBUG: subject {subject_id} has {len(patient)} visits")
            
    return tokenizer, tuple(datasets)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='GBert-predict', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-pretraining', type=str, required=False,
                        help="pretraining model")
    parser.add_argument("--train_file", default='data-multi-visit.pkl', type=str, required=False,
                        help="training data file.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain",
                        default=False,
                        action='store_true',
                        help="is use pretrain")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--therhold",
                        default=0.3,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=55,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=20.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available()
                          and not args.no_cuda else "cpu")

    if not args.do_train and not args.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    tokenizer, (train_dataset, eval_dataset, test_dataset) = load_dataset(args)
    # DEBUG CRITICAL: Kiểm tra dataset sizes
    print(f"🔍 CRITICAL DEBUG: train_dataset size = {len(train_dataset)}")
    print(f"🔍 CRITICAL DEBUG: eval_dataset size = {len(eval_dataset)}")
    print(f"🔍 CRITICAL DEBUG: test_dataset size = {len(test_dataset)}")

    if len(train_dataset) == 0:
        print("❌ ERROR: train_dataset is empty! Cannot create DataLoader.")
        print("❌ Training will fail. Check data files and filtering logic.")
        # Tạm thời skip training nếu dataset rỗng
        if args.do_train:
            print("🚫 Skipping training due to empty dataset")
            return
    train_dataloader = DataLoader(train_dataset,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=1)
    eval_dataloader = DataLoader(eval_dataset,
                                 sampler=SequentialSampler(eval_dataset),
                                 batch_size=1)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=SequentialSampler(test_dataset),
                                 batch_size=1)

    print('Loading Model: ' + args.model_name)
    if args.use_pretrain:
        logger.info("Use Pretraining model")
        model = GBERT_Predict.from_pretrained(
            args.pretrain_dir, tokenizer=tokenizer)
    else:
        config = BertConfig(
            vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
        config.graph = args.graph
        model = GBERT_Predict(config, tokenizer)
    logger.info('# of model parameters: ' + str(get_n_params(model)))

    model.to(device)

    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model it-self
    proc_output_model_file = os.path.join(  # ĐỔI: rx_output_model_file → proc_output_model_file
        args.output_dir, "pytorch_model.bin")

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    global_step = 0
    if args.do_train:
        writer = SummaryWriter(args.output_dir)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", 1)

        dx_acc_best, proc_acc_best = 0, 0  # ĐỔI: rx_acc_best → proc_acc_best
        acc_name = 'prauc'
        dx_history = {'prauc': []}
        proc_history = {'prauc': []}  # ĐỔI: rx_history → proc_history

        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            print('')
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            prog_iter = tqdm(train_dataloader, leave=False, desc='Training')
            model.train()
            for _, batch in enumerate(prog_iter):
                batch = tuple(t.to(device) for t in batch)
                input_ids, dx_labels, proc_labels = batch  # ĐỔI: rx_labels → proc_labels
                input_ids, dx_labels, proc_labels = input_ids.squeeze(
                    dim=0), dx_labels.squeeze(dim=0), proc_labels.squeeze(dim=0)  # ĐỔI: rx_labels → proc_labels
                loss, proc_logits = model(input_ids, dx_labels=dx_labels, proc_labels=proc_labels,  # ĐỔI: rx_labels → proc_labels, rx_logits → proc_logits
                                        epoch=global_step)
                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += 1
                nb_tr_steps += 1

                # Display loss
                prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

                optimizer.step()
                optimizer.zero_grad()

            writer.add_scalar('train/loss', tr_loss / nb_tr_steps, global_step)
            global_step += 1

            if args.do_eval:
                print('')
                logger.info("***** Running eval *****")
                model.eval()
                proc_y_preds = []  # ĐỔI: rx_y_preds → proc_y_preds
                proc_y_trues = []  # ĐỔI: rx_y_trues → proc_y_trues
                for eval_input in tqdm(eval_dataloader, desc="Evaluating"):
                    eval_input = tuple(t.to(device) for t in eval_input)
                    input_ids, dx_labels, proc_labels = eval_input  # ĐỔI: rx_labels → proc_labels
                    input_ids, dx_labels, proc_labels = input_ids.squeeze(
                    ), dx_labels.squeeze(), proc_labels.squeeze(dim=0)  # ĐỔI: rx_labels → proc_labels
                    with torch.no_grad():
                        loss, proc_logits = model(  # ĐỔI: rx_logits → proc_logits
                            input_ids, dx_labels=dx_labels, proc_labels=proc_labels)  # ĐỔI: rx_labels → proc_labels
                        proc_y_preds.append(t2n(torch.sigmoid(proc_logits)))  # ĐỔI: rx_logits → proc_logits
                        proc_y_trues.append(t2n(proc_labels))  # ĐỔI: rx_labels → proc_labels

                print('')
                proc_acc_container = metric_report(np.concatenate(proc_y_preds, axis=0), np.concatenate(proc_y_trues, axis=0),  # ĐỔI: rx_y_preds → proc_y_preds, rx_y_trues → proc_y_trues
                                                 args.therhold)
                for k, v in proc_acc_container.items():  # ĐỔI: rx_acc_container → proc_acc_container
                    writer.add_scalar(
                        'eval/{}'.format(k), v, global_step)

                if proc_acc_container[acc_name] > proc_acc_best:  # ĐỔI: rx_acc_container → proc_acc_container, rx_acc_best → proc_acc_best
                    proc_acc_best = proc_acc_container[acc_name]  # ĐỔI: rx_acc_best → proc_acc_best
                    # save model
                    torch.save(model_to_save.state_dict(),
                               proc_output_model_file)  # ĐỔI: rx_output_model_file → proc_output_model_file

        with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
            fout.write(model.config.to_json_string())

    if args.do_test:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", len(test_dataset))
        logger.info("  Batch size = %d", 1)

        def test(task=0):
            # Load a trained model that you have fine-tuned
            model_state_dict = torch.load(proc_output_model_file)  # ĐỔI: rx_output_model_file → proc_output_model_file
            model.load_state_dict(model_state_dict)
            model.to(device)

            model.eval()
            y_preds = []
            y_trues = []
            for test_input in tqdm(test_dataloader, desc="Testing"):
                test_input = tuple(t.to(device) for t in test_input)
                input_ids, dx_labels, proc_labels = test_input  # ĐỔI: rx_labels → proc_labels
                input_ids, dx_labels, proc_labels = input_ids.squeeze(
                ), dx_labels.squeeze(), proc_labels.squeeze(dim=0)  # ĐỔI: rx_labels → proc_labels
                with torch.no_grad():
                    loss, proc_logits = model(  # ĐỔI: rx_logits → proc_logits
                        input_ids, dx_labels=dx_labels, proc_labels=proc_labels)  # ĐỔI: rx_labels → proc_labels
                    y_preds.append(t2n(torch.sigmoid(proc_logits)))  # ĐỔI: rx_logits → proc_logits
                    y_trues.append(t2n(proc_labels))  # ĐỔI: rx_labels → proc_labels

            print('')
            acc_container = metric_report(np.concatenate(y_preds, axis=0), np.concatenate(y_trues, axis=0),
                                          args.therhold)

            # save report
            if args.do_train:
                for k, v in acc_container.items():
                    writer.add_scalar(
                        'test/{}'.format(k), v, 0)

            return acc_container

        test(task=0)


if __name__ == "__main__":
    main()