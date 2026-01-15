import torch 
from torch.utils.data import Dataset
import pandas as pd

class TextDataset(Dataset):
    def __init__(self, args, 
                 df,
                 tokenizer, 
                 max_length=512, 
                 is_train=False,
                 is_submission=False):
        self.args = args
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.is_submission = is_submission
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cur_line = self.df.iloc[idx]
        if self.is_submission:
            output = self._getitem_test(cur_line, idx)
        else:
            if self.is_train:
                output = self._getitem_train(cur_line)
            else:
                output = self._getitem_val(cur_line, idx)
        return output
        
    def _getitem_train(self, cur_line):
        # 🔥 [수정] title 컬럼이 없으면 'No Title'로 대체
        if 'title' in cur_line:
            title = str(cur_line['title'])
        else:
            title = 'No Title'
            
        full_text = str(cur_line['full_text'])
        label = cur_line['generated']
        
        # 문단 분리
        paragraph_text = full_text.split('\n')
        paragraph_index = [i for i in range(len(paragraph_text))]
        
        item = {
            "title": title,
            'full_text': full_text,
            "paragraph_index": paragraph_index,
            "paragraph_text": paragraph_text,
            "label": label            
        }        
        return item

    def _getitem_val(self, cur_line, idx):
        # 🔥 [수정] title 컬럼이 없으면 'No Title'로 대체
        if 'title' in cur_line:
            title = str(cur_line['title'])
        else:
            title = 'No Title'
            
        full_text = str(cur_line['full_text'])
        label = cur_line['generated']
        
        paragraph_text = full_text.split('\n')
        paragraph_index = [i for i in range(len(paragraph_text))]
        
        item = {
            "title": title,
            'full_text': full_text,
            "paragraph_index": paragraph_index,
            "paragraph_text": paragraph_text,
            "label": label,
            'idx': idx            
        }        
        return item
    
    def _getitem_test(self, cur_line, idx):
        # ID 처리
        ID = cur_line.get('ID', f'TEST_{idx}')
        
        # 🔥 [수정] title 컬럼이 없으면 'No Title'로 대체
        if 'title' in cur_line:
            title = str(cur_line['title'])
        else:
            title = 'No Title'

        # Paragraph Text 처리
        if 'paragraph_text' in cur_line:
            p_text_raw = cur_line['paragraph_text']
        else:
            # paragraph_text가 없고 full_text만 있는 경우
            p_text_raw = cur_line.get('full_text', '')

        if isinstance(p_text_raw, str):
            paragraph_text = p_text_raw.split('\n')
        else:
            paragraph_text = p_text_raw

        paragraph_index = cur_line.get('paragraph_index', [i for i in range(len(paragraph_text))])

        item = {
            "ID": ID,
            "title": title,
            'full_text': "\n".join(paragraph_text), 
            "paragraph_index": paragraph_index,
            "paragraph_text": paragraph_text,
        }
        return item