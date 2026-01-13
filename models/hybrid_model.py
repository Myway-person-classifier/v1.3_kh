import torch
import torch.nn as nn
from transformers import AutoModel
import sys

# 경로 문제 방지
try:
    from models.AvsHModel import AvsHModel
except ImportError:
    sys.path.append('.') 
    from models.AvsHModel import AvsHModel

class HybridAvsHModel(AvsHModel):
    def __init__(self, args):
        super().__init__(args)
        
        self.use_infonce_loss = getattr(args, 'use_infonce_loss', False)
        self.temperature = getattr(args, 'temperature', 0.07)
        self.lambda_cl = getattr(args, 'lambda_cl', 0.1)
        
        # 메인 Loss 함수 (Trainer가 스칼라 Loss를 요구함)
        self.loss_fct = nn.BCEWithLogitsLoss()
        
        if self.use_infonce_loss:
            print(f"HybridAvsHModel: InfoNCE Loss enabled (temp={self.temperature})")

    def forward(self, input_ids, attention_mask=None, chunk_size=12, **kwargs):
        # Title 처리 (없으면 기본값)
        if 'title' not in kwargs:
            kwargs['title'] = 'No Title'
            
        # 부모 모델 실행
        total_logits, paragraph_logits = super().forward(
            input_ids, attention_mask, chunk_size, **kwargs
        )
        
        # Loss 계산
        loss = None
        labels = kwargs.get('labels')
        
        if labels is not None:
            loss = self.loss_fct(total_logits.view(-1), labels.view(-1))
            
        # Trainer 호환 리턴 (Dictionary)
        output = {
            'logits': total_logits,
            'paragraph_logits': paragraph_logits
        }
        
        if loss is not None:
            output['loss'] = loss
            
        return output