import os
import sys
import numpy as np
import torch
from transformers import TrainingArguments, Trainer, AutoTokenizer

# Import 경로 (custom_datasets 사용)
from custom_datasets.get_dataset import get_dataset
from custom_datasets.text_collator import TextCollator
from trainers.hybrid_trainer import HybridTrainer
from utils.compute_metrics import get_metric
from utils.arguments import get_arguments

class FoldTrainer:
    def __init__(self, args):
        self.args = args
        self.logit_dir = "./outputs/fold_logits"
        os.makedirs(f"{self.logit_dir}/oof", exist_ok=True)
        os.makedirs(f"{self.logit_dir}/test", exist_ok=True)
    
    def train_fold(self, fold_idx: int):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx}")
        print(f"{'='*60}\n")
        
        # Args 업데이트
        self.args.fold_idx = fold_idx
        self.args.is_kfold = True
        
        # Tokenizer 로드
        tokenizer = AutoTokenizer.from_pretrained(self.args.embedding_model)
        
        # 데이터셋 로드
        train_dataset, val_dataset = get_dataset(self.args, tokenizer)
        
        # 모델 로드
        model = self._load_model()
        if torch.cuda.is_available():
            model = model.cuda()
            
        # 출력 디렉토리
        output_dir = f"./outputs/fold_{fold_idx}"
        
        # Training Arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            logging_steps=self.args.logging_steps,
            save_strategy="epoch",
            
            # [수정] 최신 Transformer 대응
            eval_strategy="epoch",
            
            save_total_limit=2,
            
            # [수정] NaN 방지 (안정성 우선)
            fp16=False, 
            
            dataloader_num_workers=0,
            report_to="none",
            
            # [수정] 데이터 컬럼 삭제 방지
            remove_unused_columns=False 
        )
        
        # Collator 생성
        collator = TextCollator(self.args, tokenizer)
        
        # [수정] 함수 호출 결과 전달
        metrics_fn = get_metric(self.args)

        # Trainer 생성
        trainer = HybridTrainer(
            args_original=self.args,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=metrics_fn, 
        )
        
        # 학습 시작
        trainer.train()
        
        # 모델 저장
        trainer.save_model(f"{output_dir}/best_model")
        
        # Logit 수집 및 저장
        self._collect_logits(trainer, val_dataset, collator, fold_idx)
    
    def _load_model(self):
        # HybridAvsH 모델 로드
        if self.args.model_name == 'HybridAvsH':
            from models.hybrid_model import HybridAvsHModel
            return HybridAvsHModel(self.args)
        else:
            from models.AvsHModel import AvsHModel
            return AvsHModel(self.args)

    def _collect_logits(self, trainer, dataset, collator, fold_idx):
        print(f"\nCollecting logits for Fold {fold_idx}...")
        trainer.model.eval()
        
        all_logits = []
        all_labels = []
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=self.args.per_device_eval_batch_size, 
            collate_fn=collator, 
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in dataloader:
                device = trainer.args.device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                outputs = trainer.model(**batch)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs[0]
                
                # NaN 제거 후 저장
                logits_np = logits.detach().cpu().numpy().flatten()
                logits_np = np.nan_to_num(logits_np, nan=0.0)
                
                all_logits.append(logits_np)
                
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu().numpy().flatten())
        
        saved_logits = np.concatenate(all_logits)
        if fold_idx == 0:
            np.save(f"{self.logit_dir}/oof/fold0_logits.npy", saved_logits)
            print("Saved OOF logits.")
        else:
            np.save(f"{self.logit_dir}/test/fold{fold_idx}_logits.npy", saved_logits)
            print("Saved Test logits.")

if __name__ == "__main__":
    args = get_arguments()
    trainer = FoldTrainer(args)
    
    try:
        trainer.train_fold(args.fold_idx)
        print(f"✅ Fold {args.fold_idx} Finished Successfully!")
    except Exception as e:
        print(f"❌ Error in Fold {args.fold_idx}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)