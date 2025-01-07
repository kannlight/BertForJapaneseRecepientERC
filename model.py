import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from sklearn.metrics import classification_report

tokenizer_name = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_name)
model_name = 'model_transformers'
model = BertForSequenceClassification.from_pretrained(model_name)

CATEGORIES = [
    'joy',
    'sadness',
    'anticipation',
    'surprise',
    'anger',
    'fear',
    'disgust',
    'trust'
]
MAX_LENGTH = 512

utter1 = '助かる！'
utter2 = '僕も助かる〜！\n心愛さんが眠るまで話そ'

token=tokenizer(
                    utter1, utter2,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                    return_tensors="pt"
                )

output = model(**token)
print(output.logits)