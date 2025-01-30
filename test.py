import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import json
import sys
import os

tokenizer_name = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_name)
best_model_name = 'model_transformers'
model = BertForSequenceClassification.from_pretrained(best_model_name)
# GPUを使用
model = model.cuda()

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

test_data_file = './DatasetTest.json'

def test():
    # データセットから対話データを読み込む
    data = {}
    with open(test_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_test = []
    # 各対話をトークン化して追加
    if 'data' in data:
        for pack in data['data']:
            for talk in pack:
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
                    return_tensors="pt"
                )
                # GPUを使用
                token = {k: v.cuda() for k, v in token.items()}
                # ラベル付け(これはテンソルでなくて良い)
                token['labels'] = talk['label']
                dataset_test.append(token)

    # テスト
    true_labels = []
    predicted_labels = []
    for data in dataset_test:
        # テストデータのラベル(ここで扱うラベルは確率分布である)
        label = data.pop('labels') # popすることでmodelへの入力にはラベルがない状態に
        true_labels.append(label.index(max(label)))
        # モデルが出力した分類スコアから、最大値となるクラスを取得(torch.argmaxは出力もテンソルとなる点に注意)
        output = model(**data)
        predicted_labels.append(output.logits.argmax(-1).item())

    # precision, recall, f1, データ数 をクラス毎、ミクロ、マクロ、加重平均で算出
    print(classification_report(true_labels, predicted_labels, labels=[0,1,2,3,4,5,6,7], target_names=CATEGORIES))

def list_incorrect(model_name):
    # データセットから対話データを読み込む
    data = {}
    with open(test_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    incorrect_data = []
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
                return_tensors="pt"
            )
            # GPUを使用
            token = {k: v.cuda() for k, v in token.items()}

            # テストデータのラベル(ここで扱うラベルは確率分布である)
            labels = talk['label']
            label = labels.index(max(labels)) # popすることでmodelへの入力にはラベルがない状態に

            # モデルが出力した分類スコアから、最大値となるクラスを取得(torch.argmaxは出力もテンソルとなる点に注意)
            output = model(**token).logits.argmax(-1).item()

            # 予測が間違っているデータをリストに追加
            if label != output:
                incorrect_data.append(talk)

    # jsonファイルで出力
    filename = 'incorrect_data/incorrect_data_'+model_name+'.json'
    i = 1
    while os.path.isfile(filename):
        i += 1
        filename = 'incorrect_data/incorrect_data_'+model_name+'_'+str(i)+'.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(incorrect_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        list_incorrect(sys.argv[1]) # 例:v11
    else:
        test()