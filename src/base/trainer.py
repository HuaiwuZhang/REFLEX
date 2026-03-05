import logging
import src.utils
import os
import time
from typing import Optional, List, Union
from collections import defaultdict
import numpy as np
from src.utilss.logging import get_logger
import torch
from torch import nn, Tensor
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim import Adam
from tqdm import tqdm
from src.utilss import metrics as mc
from src.FetterGrad import *
from src.utils import *
import math

import pandas as pd
import wandb
from src.sequence_comparator import SequenceComparator


class AdaptiveModelSaver:


    def __init__(self, f1_tolerance=0.005, min_lm_loss_delta=0.0):

        self.f1_tolerance = f1_tolerance
        self.min_lm_loss_delta = min_lm_loss_delta

        self.best_f1 = 0.0
        self.best_lm_loss = float('inf')

        self.last_saved_f1 = 0.0
        self.last_saved_lm_loss = float('inf')
        self.last_saved_epoch = -1

        self.save_count = 0

    def should_save(self, current_epoch, current_f1, current_lm_loss):

        should_save = current_f1 > self.best_f1

        if should_save:
            old_best_f1 = self.best_f1
            self.best_f1 = current_f1
            if current_lm_loss < self.best_lm_loss:
                self.best_lm_loss = current_lm_loss

            self.save_count += 1
            self.last_saved_epoch = current_epoch
            self.last_saved_f1 = current_f1
            self.last_saved_lm_loss = current_lm_loss

            save_reason = f"[IMPROVE] microF1 improved ({current_f1:.4f} > {old_best_f1:.4f})"
        else:
            save_reason = f"[WAIT] microF1 not improved ({current_f1:.4f} ≤ {self.best_f1:.4f})"

        return should_save, save_reason

    def epochs_since_last_save(self, current_epoch):
        
        if self.last_saved_epoch == -1:
            return current_epoch + 1
        return current_epoch - self.last_saved_epoch

    def get_save_info(self):
        
        return {
            'save_count': self.save_count,
            'last_saved_epoch': self.last_saved_epoch,
            'last_saved_f1': self.last_saved_f1,
            'last_saved_lm_loss': self.last_saved_lm_loss,
            'best_f1': self.best_f1,
            'best_lm_loss': self.best_lm_loss
        }


class EarlyStoppingByModelSave:


    def __init__(self, patience=25):

        self.patience = patience

    def should_stop(self, model_saver, current_epoch):

        epochs_without_save = model_saver.epochs_since_last_save(current_epoch)

        if epochs_without_save >= self.patience:
            save_info = model_saver.get_save_info()
            message = (
                f"\n{'=' * 70}\n"
                f"🛑 Early Stopping Triggered!\n"
                f"   No model saved for {epochs_without_save} epochs (patience: {self.patience})\n"
                f"   Last saved at epoch {save_info['last_saved_epoch']}:\n"
                f"     - microF1: {save_info['last_saved_f1']:.4f}\n"
                f"     - lm_loss: {save_info['last_saved_lm_loss']:.4f}\n"
                f"   Historical best:\n"
                f"     - microF1: {save_info['best_f1']:.4f}\n"
                f"     - lm_loss: {save_info['best_lm_loss']:.4f}\n"
                f"   Total models saved: {save_info['save_count']}\n"
                f"{'=' * 70}"
            )
            return True, message
        else:
            message = f"[WAIT] {epochs_without_save}/{self.patience} epochs without save"
            return False, message


class BaseTrainer():
    def __init__(
            self,
            model: nn.Module,
            data,
            args,
            device: Optional[Union[torch.device, str]] = None,

    ):
        super().__init__()

        if device is None:
            print("`device` is missing, try to train and evaluate the model on default device.")
            if torch.cuda.is_available():
                print("cuda device is available, place the model on the device.")
                self._device = torch.device("cuda")
            else:
                print("cuda device is not available, place the model on cpu.")
                self._device = torch.device("cpu")
        else:
            if isinstance(device, torch.device):
                self._device = device
            else:
                self._device = torch.device(device)
        self._logger = get_logger(
            args.log_dir, __name__, 'info_{}.log'.format(args.n_exp), level=logging.INFO)
        self._model = model
        self._wandb_flag = args.wandb
        self.num_param = self.model.param_num(self.model.name)
        self._logger.info("the number of parameters: {}".format(self.num_param))
        if args.wandb:
            wandb.run.summary["Params"] = self.num_param

        self.model.to(self._device)
        self._data = data
        self.args = args
        print(args.task)
        self._loss_criterion = nn.BCEWithLogitsLoss()
        self._base_lr = args.base_lr
        
        # Optimizer ablation: use standard Adam if ablation_no_generation is True
        if hasattr(args, 'ablation_no_generation') and args.ablation_no_generation:
            print("Using standard Adam optimizer (ablation mode - no generation)")
            self._optimizer = optim.Adam(model.parameters(), lr=args.base_lr)
            self._use_fettergrad = False
        else:
            print("Using FetterGrad optimizer")
            self._optimizer = FetterGrad(optim.Adam(model.parameters(), lr=args.base_lr))
            self._use_fettergrad = True
        
        self._lr_decay_ratio = args.lr_decay_ratio
        self._steps = args.steps
        if args.lr_decay_ratio == 1:
            self._lr_scheduler = None
        else:
            # For standard Adam, access optimizer directly; for FetterGrad, use .optimizer attribute
            base_optimizer = self._optimizer if not self._use_fettergrad else self.optimizer.optimizer
            self._lr_scheduler = MultiStepLR(base_optimizer,
                                             args.steps,
                                             gamma=args.lr_decay_ratio)
        self._clip_grad_value = args.max_grad_norm
        self._max_epochs = args.max_epochs
        self._patience = args.patience
        self._save_iter = args.save_iter
        self._save_path = args.log_dir
        self._n_exp = args.n_exp
        self._data = data

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def loss_criterion(self):
        return self._loss_criterion

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def save_path(self):
        return self._save_path

    def _check_device(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.to(self._device) for tensor in tensors]
        else:
            return tensors.to(self._device)

    def _to_numpy(self, tensors: Union[Tensor, List[Tensor]]):
        if isinstance(tensors, list):
            return [tensor.cpu().detach().numpy() for tensor in tensors]
        else:
            return tensors.cpu().detach().numpy()

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [Tensor(array) for array in nparray]
        else:
            return Tensor(nparray)

    def save_model(self, epoch, save_path, n_exp, microf1, temp_lr_scheduler, stage):
        save_path = save_path + '/' + stage
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filename = 'final_model_{}.pt'.format(n_exp)
        
        # Get optimizer state_dict (handle both FetterGrad and standard Adam)
        optimizer_state = self.optimizer.optimizer.state_dict() if self._use_fettergrad else self.optimizer.state_dict()
        
        torch.save({'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'microf1': microf1,
                    'optimizer': optimizer_state,
                    'lr_scheduler': temp_lr_scheduler.state_dict(),
                    "numpy_random": np.random.get_state(),
                    "torch_random": torch.get_rng_state(),
                    "torch_cuda_random": torch.cuda.get_rng_state_all()
                    }, os.path.join(save_path, filename))
        return os.path.join(save_path, filename)

    def load_model(self, save_path, n_exp, stage):
        save_path = save_path + '/' + stage
        filename = 'final_model_{}.pt'.format(n_exp)
        checkpoint = torch.load(os.path.join(save_path, filename))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint['epoch']
        microf1 = checkpoint['microf1']
        optimizer = checkpoint['optimizer']
        lr_scheduler = checkpoint['lr_scheduler']
        np.random.set_state(checkpoint["numpy_random"])
        torch.set_rng_state(checkpoint["torch_random"])
        torch.cuda.set_rng_state_all(checkpoint["torch_cuda_random"])

        return epoch, microf1, optimizer, lr_scheduler


    def train(self, PPIData_, stage='train'):
        print("start training !!!!!")
        iter = 0

        
        model_saver = AdaptiveModelSaver(
            f1_tolerance=getattr(self.args, 'f1_tolerance', 0.005),  # 0.5%
            min_lm_loss_delta=getattr(self.args, 'min_lm_loss_delta', 0.0)
        )

        # early_stopper = EarlyStoppingByModelSave(
        #     patience=getattr(self.args, 'patience', 25)
        # )

        best_acc = 0
        best_pre = 0
        best_recall = 0
        best_microf1 = 0

        existing_epoch = -1

        try:
            existing_epoch, microf1, optimizer, lr_scheduler = self.load_model(self.save_path, self._n_exp, stage)
            # Load optimizer state (handle both FetterGrad and standard Adam)
            if self._use_fettergrad:
                self.optimizer.optimizer.load_state_dict(optimizer)
            else:
                self.optimizer.load_state_dict(optimizer)
            self.lr_scheduler.load_state_dict(lr_scheduler)

            model_saver.best_f1 = microf1
            model_saver.last_saved_f1 = microf1
            model_saver.last_saved_epoch = existing_epoch
            model_saver.save_count = 1

            print(f'Loaded existing model from epoch {existing_epoch}, microF1: {microf1:.4f}')
        except:
            print('No existing trained model')

        for epoch in range(max(existing_epoch, -1) + 1, self._max_epochs):
            steps = math.ceil(len(self.data.train_mask) / self.args.batch_size)
            self.model.train()
            random.shuffle(self.data.train_mask)

            ppi_loss_sum = 0.0
            lm_loss_sum = 0.0
            kl_loss_sum = 0.0

            start_time = time.time()

            
            for step in range(steps):
                torch.cuda.empty_cache()

                if step == steps - 1:
                    train_edge_id = self.data.train_mask[step * self.args.batch_size:]
                else:
                    train_edge_id = self.data.train_mask[
                                    step * self.args.batch_size:(step + 1) * self.args.batch_size
                                    ]

                train_edge_id_tensor = torch.tensor(train_edge_id, dtype=torch.long, device=self._device)
                
                if self.args.model == 'PPIGEN_diffusion':
                    from src.utils import get_target_seq_embeddings_for_edges
                    target_seq_embeddings = get_target_seq_embeddings_for_edges(
                        PPIData_,
                        train_edge_id_tensor,
                        max_tgt_len=self.args.protein_max_length,
                        esm_model=PPIData_.esm_model,
                        d_esm=1280
                    )
                    
                    B, T, d_esm = target_seq_embeddings.shape
                    fixed_len = self.args.protein_max_length
                    
                    if T < fixed_len:
                        padding = torch.zeros(B, fixed_len - T, d_esm, device=target_seq_embeddings.device)
                        target_seq_embeddings = torch.cat([target_seq_embeddings, padding], dim=1)
                    elif T > fixed_len:
                        target_seq_embeddings = target_seq_embeddings[:, :fixed_len, :]
                    
                    label = self.data.edge_attr[train_edge_id_tensor].type(torch.FloatTensor).to(self._device)

                    pair_logits, diffusion_loss, kl_loss = self.model(
                        data=self.data,
                        edge_id=train_edge_id_tensor,
                        target_seq_embeddings=target_seq_embeddings,
                    )

                    ppi_loss = self.loss_criterion(pair_logits, label)
                    loss = ppi_loss + self.args.lm_weight * diffusion_loss + self.args.kl_weight * kl_loss
                    lm_loss = diffusion_loss
                else:
                    target_seq = get_target_seqs_for_edges(
                        PPIData_,
                        train_edge_id_tensor,
                        max_tgt_len=self.args.protein_max_length
                    )
                    label = self.data.edge_attr[train_edge_id_tensor].type(torch.FloatTensor).to(self._device)

                    pair_logits, _, lm_loss, kl_loss = self.model(
                        data=self.data,
                        edge_id=train_edge_id_tensor,
                        target_seq=target_seq,
                    )

                    ppi_loss = self.loss_criterion(pair_logits, label)
                    loss = ppi_loss + self.args.lm_weight * lm_loss + self.args.kl_weight * kl_loss

                ppi_loss_sum += ppi_loss.item()
                lm_loss_sum += lm_loss.item()
                kl_loss_sum += kl_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iter += 1

            end_time = time.time()

            
            record = self.evaluate(train_loss_sum=ppi_loss_sum, PPIData_=PPIData_)

            if self.lr_scheduler is None:
                new_lr = self._base_lr
            else:
                new_lr = self.lr_scheduler.get_last_lr()[0]

            
            should_save, save_reason = model_saver.should_save(
                current_epoch=epoch,
                current_f1=record['microF1'],
                current_lm_loss=lm_loss_sum
            )

            
            if should_save:
                temp_lr_scheduler = self.lr_scheduler
                model_file_name = self.save_model(
                    epoch,
                    self._save_path,
                    self._n_exp,
                    record['microF1'],
                    temp_lr_scheduler,
                    stage='train'
                )

                save_info = model_saver.get_save_info()

                self._logger.info(
                    f"\n{'=' * 70}\n"
                    f"[SAVE] Model Saved! (#{save_info['save_count']})\n"
                    f"   Reason: {save_reason}\n"
                    f"   Current metrics:\n"
                    f"     - microF1: {record['microF1']:.4f} (best: {save_info['best_f1']:.4f})\n"
                    f"     - lm_loss: {lm_loss_sum:.4f} (best: {save_info['best_lm_loss']:.4f})\n"
                    f"     - acc: {record['acc']:.4f}, pre: {record['pre']:.4f}, recall: {record['recall']:.4f}\n"
                    f"   Saved to: {model_file_name}\n"
                    f"{'=' * 70}"
                )

                if record['microF1'] > best_microf1:
                    best_microf1 = record['microF1']
                    best_acc = record['acc']
                    best_pre = record['pre']
                    best_recall = record['recall']

            
            # should_stop, early_stop_message = early_stopper.should_stop(
            #     model_saver=model_saver,
            #     current_epoch=epoch
            # )
            should_stop = False
            early_stop_message = ""

            
            epochs_without_save = model_saver.epochs_since_last_save(epoch)

            message = (
                "Epoch [{}/{}] ({}) "
                "ppi_loss: {:.4f}, lm_loss: {:.4f}, kl_loss: {:.4f}, total_loss: {:.4f}\n"
                "acc: {:.4f}, pre: {:.4f}, recall: {:.4f}, microF1: {:.4f}\n"
                "{}\n"
                "Save status: {}\n"
                "lr: {:.6f}, {:.1f}s".format(
                    epoch,
                    self._max_epochs,
                    iter,
                    ppi_loss_sum,
                    lm_loss_sum,
                    kl_loss_sum,
                    ppi_loss_sum + self.args.lm_weight * lm_loss_sum + self.args.kl_weight * kl_loss_sum,
                    record['acc'],
                    record['pre'],
                    record['recall'],
                    record['microF1'],
                    save_reason,
                    "[OK] SAVED" if should_save else "[SKIP] Not saved",
                    new_lr,
                    (end_time - start_time),
                )
            )

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self._logger.info(message)

            
            if self._wandb_flag:
                wandb.log({
                    'epoch': epoch,
                    'microF1': record['microF1'],
                    'accuracy': record['acc'],
                    'precision': record['pre'],
                    'recall': record['recall'],
                    
                    'ppi_loss': ppi_loss_sum,
                    'lm_loss': lm_loss_sum,
                    'kl_loss': kl_loss_sum,
                    'total_loss': ppi_loss_sum + self.args.lm_weight * lm_loss_sum + self.args.kl_weight * kl_loss_sum,
                    
                    'best_microF1': best_microf1,
                    'best_accuracy': best_acc,
                    'best_precision': best_pre,
                    'best_recall': best_recall,
                    
                    'model_saved': 1 if should_save else 0,
                    'epochs_without_save': epochs_without_save,
                    'best_f1_so_far': model_saver.best_f1,
                    'best_lm_loss_so_far': model_saver.best_lm_loss,
                    'save_count': model_saver.save_count,
                }, step=epoch)

            
            # if should_stop:
            #     self._logger.info(early_stop_message)
            #
            #     save_info = model_saver.get_save_info()
            #
            #     if self._wandb_flag:
            #         wandb.run.summary[stage + "_total_saves"] = save_info['save_count']
            #         wandb.run.summary[stage + "_final_best_f1"] = save_info['best_f1']
            #         wandb.run.summary[stage + "_final_best_lm_loss"] = save_info['best_lm_loss']
            #         wandb.run.summary[stage + "_last_saved_epoch"] = save_info['last_saved_epoch']
            #         wandb.run.summary[stage + "_stopped_at_epoch"] = epoch
            #         wandb.run.summary[stage + "_best_microF1"] = best_microf1
            #         wandb.run.summary[stage + "_best_accuracy"] = best_acc
            #         wandb.run.summary[stage + "_best_precision"] = best_pre
            #         wandb.run.summary[stage + "_best_recall"] = best_recall
            #
            #     break

        
        save_info = model_saver.get_save_info()
        self._logger.info(
            f"\n{'=' * 70}\n"
            f"   Training Completed!\n"
            f"   Total models saved: {save_info['save_count']}\n"
            f"   Best microF1: {save_info['best_f1']:.4f}\n"
            f"   Best lm_loss: {save_info['best_lm_loss']:.4f}\n"
            f"   Last saved at epoch: {save_info['last_saved_epoch']}\n"
            f"     - microF1: {save_info['last_saved_f1']:.4f}\n"
            f"     - lm_loss: {save_info['last_saved_lm_loss']:.4f}\n"
            f"{'=' * 70}"
        )

        if self._wandb_flag:
            wandb.run.summary[stage + "_total_saves"] = save_info['save_count']
            wandb.run.summary[stage + "_final_best_f1"] = save_info['best_f1']
            wandb.run.summary[stage + "_final_best_lm_loss"] = save_info['best_lm_loss']
            wandb.run.summary[stage + "_best_microF1"] = best_microf1
            wandb.run.summary[stage + "_best_accuracy"] = best_acc
            wandb.run.summary[stage + "_best_precision"] = best_pre
            wandb.run.summary[stage + "_best_recall"] = best_recall
            
            print("\n" + "="*70)
            print("Best Metrics Summary:")
            print(f"   Best microF1:   {best_microf1:.4f}")
            print(f"   Best Accuracy:  {best_acc:.4f}")
            print(f"   Best Precision: {best_pre:.4f}")
            print(f"   Best Recall:    {best_recall:.4f}")
            print("="*70 + "\n")

    def evaluate(self, train_loss_sum, PPIData_=None):
        valid_pre_result_list = []
        valid_label_list = []
        valid_loss_sum = 0.0
        steps = math.ceil(len(self.data.val_mask) / self.args.batch_size)
        saved_pred = []
        with torch.no_grad():
            self.model.eval()
            for step in range(steps):
                if step == steps - 1:
                    valid_edge_id = self.data.val_mask[step * self.args.batch_size:]
                else:
                    valid_edge_id = self.data.val_mask[step * self.args.batch_size:(step + 1) * self.args.batch_size]
                valid_edge_id_tensor = torch.tensor(valid_edge_id, dtype=torch.long, device=self._device)
                
                if self.args.model == 'PPIGEN_diffusion':
                    output, _, _ = self.model(data=self.data, edge_id=valid_edge_id_tensor, target_seq_embeddings=None)
                else:
                    output, _, _, _ = self.model(data=self.data, edge_id=valid_edge_id_tensor, target_seq=None)
                
                label = self.data.edge_attr[valid_edge_id]
                label = label.type(torch.FloatTensor).to(self._device)
                loss = self.loss_criterion(output, label)
                valid_loss_sum += loss.item()
                m = nn.Sigmoid()
                pre_result = (m(output) > 0.5).type(torch.FloatTensor).to(self._device)

                valid_pre_result_list.append(pre_result.cpu().data)
                valid_label_list.append(label.cpu().data)
                saved_pred.append(m(output).to(self._device).cpu().data)

        valid_pre_result_list = torch.cat(valid_pre_result_list, dim=0)
        valid_label_list = torch.cat(valid_label_list, dim=0)

        saved_pred = torch.cat(saved_pred, dim=0)
        metrics = Metrictor_PPI(valid_pre_result_list, valid_label_list)
        record = metrics.append_result(train_loss_sum, valid_loss_sum)

        return record

    def generate_proteins(
            self,
            PPIData_,
            edge_id_list,
            random_sample: bool = True,
            save_path: Optional[str] = None,
            file_prefix: str = "generated_proteins",
            compare_with_original: bool = True,
    ):
        """Generate protein sequences for given PPI edges."""
        _ = self.load_model(self.save_path, self._n_exp, stage='train')
        self.model.eval()

        if isinstance(edge_id_list, torch.Tensor):
            edge_ids = edge_id_list.to(self._device)
        else:
            edge_ids = torch.tensor(edge_id_list, dtype=torch.long, device=self._device)

        with torch.no_grad():
            if self.args.model == 'PPIGEN_diffusion':
                alphabet = PPIData_.alphabet
                esm_model = PPIData_.esm_model.to(self._device)
                esm_model.eval()
                
                with torch.no_grad():
                    all_token_ids = torch.arange(len(alphabet.all_toks), device=self._device).unsqueeze(0)
                    out = esm_model(all_token_ids, repr_layers=[esm_model.num_layers], return_contacts=False)
                    esm_vocab_embeddings = out["representations"][esm_model.num_layers].squeeze(0)  # [vocab_size, d_esm]
                
                token_ids = self.model.generate(
                    data=self.data,
                    edge_id=edge_ids,
                    esm_vocab_embeddings=esm_vocab_embeddings,
                    random_sample=random_sample,
                    return_z=False,
                )  # [B, T]
            else:
                token_ids = self.model.generate(
                    data=self.data,
                    edge_id=edge_ids,
                    random_sample=random_sample,
                    return_z=False,
                )  # [B, T]

        if not hasattr(PPIData_, "alphabet"):
            raise AttributeError(
                "PPIData_ has no alphabet attribute. "
                "Please add `self.alphabet = alphabet` in PPIData.__init__."
            )
        alphabet = PPIData_.alphabet
        pad_idx = PPIData_.pad_value
        eos_idx = pad_idx

        id_to_tok = {i: tok for i, tok in enumerate(alphabet.all_toks)}

        seq_list = []
        for row in token_ids:
            seq_chars = []
            for tid in row.tolist():
                if tid == pad_idx:
                    break
                tok = id_to_tok.get(tid, "")
                if tok in ['A', 'G', 'V', 'I', 'L', 'F', 'P', 'Y', 'M',
                           'T', 'S', 'H', 'N', 'Q', 'W', 'R', 'K', 'D', 'E', 'C']:
                    seq_chars.append(tok)
            seq_list.append("".join(seq_chars))

        for i, s in enumerate(seq_list[:10]):
            print(f"[Gen {i}] len={len(s)}  seq={s[:80]}{'...' if len(s) > 80 else ''}")

        comparison_results = None
        if compare_with_original:
            print(f"\n{'='*80}")
            print(f"Comparing generated sequences with originals...")
            print(f"{'='*80}")
            
            original_seqs = self._get_original_sequences(PPIData_, edge_ids)
            
            if original_seqs is not None:
                comparator = SequenceComparator()
                
                comparison_results = []
                for i, (orig, gen, eid) in enumerate(zip(original_seqs, seq_list, edge_ids.tolist())):
                    verbose = (i < 3)
                    result = comparator.compare(orig, gen, edge_id=eid, verbose=verbose)
                    comparison_results.append(result)
                
                comparator.print_summary(comparison_results)
                
                if save_path is not None:
                    comparison_file = os.path.join(save_path, f"{file_prefix}_comparison.json")
                    comparator.save_results(comparison_file)
            else:
                print("Could not retrieve original sequences, skipping comparison")

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            import pandas as pd
            
            save_data = {
                "edge_id": edge_ids.detach().cpu().tolist(),
                "generated_seq": seq_list,
            }
            
            if compare_with_original and comparison_results is not None:
                original_seqs = self._get_original_sequences(PPIData_, edge_ids)
                if original_seqs is not None:
                    save_data["original_seq"] = original_seqs
                    save_data["similarity"] = [r['similarity'] for r in comparison_results]
                    save_data["edit_distance"] = [r['edit_distance'] for r in comparison_results]
                    save_data["match_rate"] = [r['position_stats']['match_rate'] for r in comparison_results]
                    save_data["length_diff"] = [r['position_stats']['length_diff'] for r in comparison_results]
            
            df = pd.DataFrame(save_data)
            file_name = f"{file_prefix}.csv"
            df.to_csv(os.path.join(save_path, file_name), index=False)
            print(f"\n[Generate] Saved {len(seq_list)} generated sequences to {os.path.join(save_path, file_name)}")

        return seq_list
    
    def _get_original_sequences(self, PPIData_, edge_ids, use_second_node=False):

        try:
            if not hasattr(PPIData_, 'prot_token_ids'):
                return None
            
            alphabet = PPIData_.alphabet
            pad_idx = PPIData_.pad_value
            
            id_to_tok = {i: tok for i, tok in enumerate(alphabet.all_toks)}
            
            edge_index = self.data.edge2
            node_ids = edge_index[:, edge_ids]  # [2, B]
            
            node_idx = 1 if use_second_node else 0
            target_node_ids = node_ids[node_idx]  # [B]
            
            original_seqs = []
            for node_id in target_node_ids.tolist():
                if node_id < len(PPIData_.prot_token_ids):
                    token_ids = PPIData_.prot_token_ids[node_id]
                    
                    seq_chars = []
                    for tid in token_ids.tolist():
                        if tid == pad_idx:
                            break
                        tok = id_to_tok.get(tid, "")
                        if tok in ['A', 'G', 'V', 'I', 'L', 'F', 'P', 'Y', 'M',
                                   'T', 'S', 'H', 'N', 'Q', 'W', 'R', 'K', 'D', 'E', 'C']:
                            seq_chars.append(tok)
                    original_seqs.append("".join(seq_chars))
                else:
                    original_seqs.append("")
            
            return original_seqs
        except Exception as e:
            print(f"Error retrieving original sequences: {e}")
            return None