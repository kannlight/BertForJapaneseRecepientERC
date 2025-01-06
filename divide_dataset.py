import random
from tqdm import tqdm
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from sklearn.metrics import classification_report

def load_data():
    # データセットの対話データをトークン化して返す
    talk_data = []
    for file in os.listdir('DatasetByLuke'):
        # データを読み込む
        data = {}
        with open('DatasetByLuke/'+file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 各対話を追加
        if 'data' in data:
            for talk in data['data']:
                talk_data.append(talk)
    return talk_data

def main():
    # データセットから対話データのデータローダを作成
    all_talk_data = load_data()
    # 対話データをシャッフル
    random.shuffle(all_talk_data)
    # 6:2:2で学習:検証:テストデータに分割
    n = len(all_talk_data)
    n_train = int(0.6*n)
    n_val = int(0.2*n)
    dataset_train = {"data": all_talk_data[:n_train]}
    dataset_val = {"data": all_talk_data[n_train:n_train+n_val]}
    dataset_test = {"data": all_talk_data[n_train+n_val:]}
    with open('DatasetTrain.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_train, f, indent=4, ensure_ascii=False)
    with open('DatasetVal.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_val, f, indent=4, ensure_ascii=False)
    with open('DatasetTest.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_test, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()