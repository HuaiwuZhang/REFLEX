import numpy as np
import random
import torch
import os
import math
import time
import models
from collections import defaultdict 
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
def sorted_pair(id1,id2):
	if id1<id2:
		return [id1,id2]
	return [id2,id1]

def check_files_exist(fList:list):
	for fName in fList:
		if not os.path.isfile(fName):
			print(f'error,input file {fName} does not exist')
			exit()
	return
#encode binary vecotr into single num
def encode_inter(vec,num=7):
	result = 0
	for i,v in enumerate(vec):
		result += v*pow(2,num-1-i)
	return result
#		print([1,1,0,0,0,1,0])
#		a = utils.encode_inter([1,1,0,0,0,1,0])
def decode_inter(value,num=7):
	result = []
	for i in range(num):
		tmp = pow(2,num-1-i)
		v = math.floor(value/tmp)
		#print(f'{value} {tmp} {v}')
		value -= v*tmp
		result.append(v)
	return np.array(result,dtype=float)
def sort_dir_by_value(dict): #note: decesending order
	keys = list(dict.keys())
	values = list(dict.values())
	sorted_value_index = np.argsort(values)[::-1]
	sorted_dict = {keys[i]: values[i] for i in sorted_value_index}
	return sorted_dict

class Metrictor_PPI:
	def __init__(self, pre_y, truth_y, is_binary=False):
		self.TP = 0
		self.FP = 0
		self.TN = 0
		self.FN = 0

		if is_binary:
			length = pre_y.shape[0]
			for i in range(length):
				if pre_y[i] == truth_y[i]:
					if truth_y[i] == 1:
						self.TP += 1
					else:
						self.TN += 1
				elif truth_y[i] == 1:
					self.FN += 1
				elif pre_y[i] == 1:
					self.FP += 1
			self.num = length

		else:
			N, C = pre_y.shape
			for i in range(N):
				for j in range(C):
					if pre_y[i][j] == truth_y[i][j]:
						if truth_y[i][j] == 1:
							self.TP += 1
						else:
							self.TN += 1
					elif truth_y[i][j] == 1:
						self.FN += 1
					elif truth_y[i][j] == 0:
						self.FP += 1
			self.num = N * C
	
	def append_result(self,train_loss=0.0,valid_loss=0.0):
		record = {}
		self.acc = (self.TP + self.TN) / (self.num + 1e-10)
		self.pre = self.TP / (self.TP + self.FP + 1e-10)
		self.recall = self.TP / (self.TP + self.FN + 1e-10)
		self.microF1 = 2 * self.pre * self.recall / (self.pre + self.recall + 1e-10)
		record['acc'] = self.acc
		record['pre'] = self.pre
		record['recall'] = self.recall
		record['microF1'] = self.microF1

		return record

import torch

def get_target_seqs_for_edges(ppi_data_obj, edge_ids, max_tgt_len, use_second_node=False):
    data = ppi_data_obj.data
    device = data.embed1.device
    prot_token_ids = ppi_data_obj.prot_token_ids  # List[Tensor]
    pad_value = ppi_data_obj.pad_value

    if not torch.is_tensor(edge_ids):
        edge_ids = torch.tensor(edge_ids, dtype=torch.long, device=device)
    else:
        edge_ids = edge_ids.to(device)

    node_idx = 1 if use_second_node else 0
    node_ids = data.edge2[node_idx, edge_ids]  # [B]

    seq_list = []
    for nid in node_ids.tolist():
        s = prot_token_ids[int(nid)].to(device)
        if s.size(0) > max_tgt_len:
            s = s[:max_tgt_len]
        seq_list.append(s)

    max_len = max(s.size(0) for s in seq_list)  # ≤ max_tgt_len

    batch = torch.full(
        (len(seq_list), max_len),
        pad_value,
        dtype=torch.long,
        device=device
    )
    for i, s in enumerate(seq_list):
        batch[i, :s.size(0)] = s

    return batch  # [B, T<=max_tgt_len]

def get_target_seq_embeddings_for_edges(ppi_data_obj, edge_ids, max_tgt_len, esm_model=None, d_esm=1280, use_second_node=False):
    data = ppi_data_obj.data
    device = data.embed1.device
    prot_token_ids = ppi_data_obj.prot_token_ids  # List[Tensor]
    pad_value = ppi_data_obj.pad_value
    
    if not hasattr(ppi_data_obj, 'alphabet'):
        raise AttributeError("PPIData missing alphabet attribute, cannot get ESM embeddings")
    
    if esm_model is None:
        raise ValueError("esm_model must be provided! Cannot use randomly initialized embeddings")
    
    alphabet = ppi_data_obj.alphabet
    
    if not torch.is_tensor(edge_ids):
        edge_ids = torch.tensor(edge_ids, dtype=torch.long, device=device)
    else:
        edge_ids = edge_ids.to(device)

    node_idx = 1 if use_second_node else 0
    node_ids = data.edge2[node_idx, edge_ids]  # [B]
    
    seq_list = []
    for nid in node_ids.tolist():
        s = prot_token_ids[int(nid)].to(device)
        if s.size(0) > max_tgt_len:
            s = s[:max_tgt_len]
        seq_list.append(s)
    
    max_len = max(s.size(0) for s in seq_list)
    
    batch_tokens = torch.full(
        (len(seq_list), max_len),
        pad_value,
        dtype=torch.long,
        device=device
    )
    for i, s in enumerate(seq_list):
        batch_tokens[i, :s.size(0)] = s
    
    esm_model = esm_model.to(device)
    esm_model.eval()
    
    with torch.no_grad():
        out = esm_model(batch_tokens, repr_layers=[esm_model.num_layers], return_contacts=False)
        token_embeddings = out["representations"][esm_model.num_layers]  # [B, T, d_esm]
    
    return token_embeddings  # [B, T, d_esm]

import os
import logging
import sys

def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    print('Log directory: %s', log_dir)
    return logger

from sklearn.metrics import confusion_matrix,mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import pearsonr
import numpy as np

def compute_reg_metrics(ypred, ytrue):
    mae = mean_absolute_error(ytrue, ypred)
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    pcc, _ = pearsonr(ytrue, ypred)

    return mae, rmse, r2, pcc

def count_positive_products(preds, labels):

    if not (isinstance(preds, np.ndarray) and isinstance(labels, np.ndarray)):
        raise TypeError("Not NumPy ndarray")

    if preds.shape != labels.shape:
        raise ValueError("Shape must be the same")

    product = preds * labels
    return np.count_nonzero(product > 0)

