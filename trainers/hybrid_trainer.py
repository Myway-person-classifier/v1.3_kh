from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
import torch
import inspect
from utils.losses import BPRLoss

class HybridTrainer(Trainer):
    def __init__(self, args_original, **kwargs):
        super().__init__(**kwargs)
        self.args_original = args_original
        self.loss_fn_list = [('bce', nn.BCEWithLogitsLoss(), 1.0)]
        
        if args_original.use_bpr_loss:
            self.loss_fn_list.append(('bpr', BPRLoss(), args_original.bpr_loss_weight))
            print("Using BPR loss")
        
        self.use_infonce_loss = getattr(args_original, 'use_infonce_loss', False)

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        return self._compute_loss_avsh(model, inputs, return_outputs)
    
    def _compute_loss_avsh(self, model, inputs, return_outputs=False):
        inputs = inputs.copy()
        
        total_label = inputs.pop("total_labels", None)
        if total_label is None and "labels" in inputs:
            total_label = inputs["labels"].float()
            
        paragraph_label = inputs.pop("paragraph_labels", None)
        if paragraph_label is not None:
            paragraph_label = paragraph_label.float()

        # 모델 실행
        outputs = model(**inputs)

        # [수정] Output 처리 (Dict vs Tuple 호환)
        cl_loss = None
        if isinstance(outputs, dict):
            total_logits = outputs.get('logits')
            paragraph_logits = outputs.get('paragraph_logits')
            if 'loss' in outputs: # 모델이 이미 loss를 계산했다면 그것을 사용 가능
                pass 
        else:
            total_logits, paragraph_logits = outputs[0], outputs[1]
        
        # Paragraph dummy
        if paragraph_label is None:
            paragraph_label = torch.zeros_like(paragraph_logits, dtype=torch.float, device=paragraph_logits.device)
        
        # Loss 계산
        total_loss = torch.tensor(0.0, device=total_logits.device)
        
        # BCE & BPR Loss 추가 계산
        if not self.args_original.split_valid_by_paragraph:
            for loss_name, loss_fn, weight in self.loss_fn_list:
                if loss_name == 'bce':
                    if total_label is not None:
                        total_loss += loss_fn(total_logits.view(-1), total_label.view(-1)) * weight
                else:
                    if total_label is not None:
                        total_loss += loss_fn(total_logits.view(-1), total_label.view(-1)) * weight

        # Return prep
        label = total_label if total_label is not None else torch.zeros_like(total_logits.view(-1))
        logits = total_logits.view(-1)
        
        if return_outputs:
            # Trainer가 기대하는 형식은 오직 (loss, outputs) 입니다.
            # outputs는 모델이 뱉은 원본이나 딕셔너리 형태여야 합니다.
            # 여기서 outputs에 loss를 포함시켜주면 확실하게 인식합니다.
            if isinstance(outputs, dict):
                outputs["loss"] = total_loss
            else:
                # 튜플인 경우 유연하게 대응
                outputs = (total_loss,) + outputs if isinstance(outputs, tuple) else outputs
            return (total_loss, outputs)       
        return total_loss