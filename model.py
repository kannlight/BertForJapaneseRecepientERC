import random
import glob
from tqdm import tqdm
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer =BertJapaneseTokenizer.from_pretrained(MODEL_NAME)
# pretrained_bert = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=8)
# pretrained_bert = pretrained_bert.cuda()

# CATEGORIES = [
#     'joy',
#     'sadness',
#     'anticipation',
#     'surprise',
#     'anger',
#     'fear',
#     'disgust',
#     'trust'
# ]
MAX_LENGTH = 512

def load_data():
    dataset_for_loader = []

    for file in tqdm(os.listdir('dataset')):
        # データセットから対話データを読み込む
        data = {}
        with open('DatasetByLuke/'+file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 各対話をトークン化して追加
        if 'data' in data:
            for talk in data['data']:
                # 1発話目(受信者の発話)と2発話目(送信者の発話)を取り出す（複数続いたら改行で繋げる）
                t1 = []
                t2 = []
                for utter in talk['talk']:
                    if utter['type'] == 1:
                        t1.append(utter['utter'])
                    if utter['type'] == 2:
                        t2.append(utter['utter'])
                text1 = '\n'.join(t1)
                text2 = '\n'.join(t2)
                # トークン化
                token=tokenizer(
                    text1, text2,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                )
                # ラベル付け
                token['label'] = talk['label']
                # バリューをテンソル化して追加
                token = { k: torch.tensor(v) for k, v in token.items() }
                dataset_for_loader.append(token)
    return dataset_for_loader

class BertForJapaneseRecepientERC(pl.LightningModule):
    def __init__(self, lr, model_name=MODEL_NAME, num_labels=8):
        # model_name: 事前学習モデル
        # num_labels: ラベル数
        # lr: 学習率
        super().__init__()
        self.save_hyperparameters()

        # 事前学習モデルのロード
        self.pretrained_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # 学習データを受け取って損失を返す関数
        def training_step(self, batch):
            output = self.pretrained_model(**batch)
            loss = output.loss

def test():
    dataset_for_loader = load_data()
    print(len(dataset_for_loader))
    print(dataset_for_loader[0])
    print(dataset_for_loader[-1])

def main():
    dataset_for_loader = load_data()
    random.shuffle(dataset_for_loader)
    n = len(dataset_for_loader)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    dataset_train = dataset_for_loader[:n_train]
    dataset_val = dataset_for_loader[n_train:n_train+n_val]
    dataset_test = dataset_for_loader[n_train+n_val:]
    dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=256)
    dataloader_test = DataLoader(dataset_test, batch_size=256)

if __name__ == "__main__":
    test()