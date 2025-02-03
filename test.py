import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report
import json
import sys
import os
from train import BertForJapaneseRecepientERC
from ..PersonalizedBertForJapapaneseRecipientERC.train import PersonalizedBertForJapaneseRecepientERC


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
        with torch.no_grad:
            output = model(**data)
        predicted_labels.append(output.logits.argmax(-1).item())

    # precision, recall, f1, データ数 をクラス毎、ミクロ、マクロ、加重平均で算出
    print(classification_report(true_labels, predicted_labels, labels=[0,1,2,3,4,5,6,7], target_names=CATEGORIES))

def list_incorrect_BJRERC(data, model_BJRE=model):
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
            labels = talk.pop('label') # popすることでmodelへの入力にはラベルがない状態に
            label = labels.index(max(labels))

            # モデルが出力した分類スコアから、最大値となるクラスを取得(torch.argmaxは出力もテンソルとなる点に注意)
            with torch.no_grad:
                output = model_BJRE(**token).logits.argmax(-1).item()

            # 予測が間違っているデータをリストに追加
            if label != output:
                incorrect_data.append(talk)
    return incorrect_data

def list_incorrect_PBJRERC(data, model_PBJRE):
    incorrect_data = []
    if 'data' in data:
        talks = data['data']
        # パックごとに処理
        for index in range( ( ( len(talks) - 1 ) // 8 ) + 1):
            # データセットの残り対話数がちょうど8対話でない場合の処理
            if index*8 + 7 > len(talks) - 1:
                j = (index*8 + 7) - (len(talks) - 1)
            else:
                j = 0

            # 1発話目(受信者の発話)と2発話目(送信者の発話)を取り出す（複数続いたら改行で繋げる）
            # パックサイズ分の処理
            text1 = []
            text2 = []
            label_list = []
            for i in range(8):
                talk = talks[index*8 + i - j]
                t1 = []
                t2 = []
                for utter in talk['talk']:
                    if utter['type'] == 1:
                        t1.append(utter['utter'])
                    if utter['type'] == 2:
                        t2.append(utter['utter'])
                text1.append('\n'.join(t1))
                text2.append('\n'.join(t2))
                
                # テストデータのラベル(ここで扱うラベルは確率分布である)
                labels = talk.pop('label') # popすることでmodelへの入力にはラベルがない状態に
                label_list.append(labels.index(max(labels)))
            
            # パックをトークン化
            token=tokenizer(
                text1, text2,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt"
            )
            # GPUを使用
            token = {k: v.cuda() for k, v in token.items()}

            # モデルが出力した分類スコアから、最大値となるクラスを取得(torch.argmaxは出力もテンソルとなる点に注意)
            with torch.no_grad:
                output = model_PBJRE(**token).logits.argmax(-1).tolist()

            # 予測が間違っているデータをリストに追加
            for i in range(8 - j): 
                if label_list[i + j] != output[i + j]:
                    incorrect_data.append(talks[index*8 + i])
    return incorrect_data

def list_incorrect(model_name):
    # データセットから対話データを読み込む
    data = {}
    with open(test_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 間違いをリストアップ
    incorrect_data = list_incorrect_BJRERC(data)

    # jsonファイルで出力
    filename = 'incorrect_data/incorrect_data_'+model_name+'.json'
    i = 1
    while os.path.isfile(filename):
        i += 1
        filename = 'incorrect_data/incorrect_data_'+model_name+'_'+str(i)+'.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(incorrect_data, f, indent=4, ensure_ascii=False)

def compare_models(path_BJRE, path_PBJRE):
    # データセットから対話データを読み込む
    data = {}
    with open(test_data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # モデルを読み込む
    model_BJRERC = BertForJapaneseRecepientERC.load_from_checkpoint(path_BJRE)
    model_PBJRERC = PersonalizedBertForJapaneseRecepientERC.load_from_checkpoint(path_PBJRE)
    model_BJRERC = model_BJRERC.cuda()
    model_PBJRERC = model_PBJRERC.cuda()

    # 間違いをリストアップ
    incorrect_data_BJRE = list_incorrect_BJRERC(data, model_BJRERC)
    incorrect_data_PBJRE = list_incorrect_PBJRERC(data, model_PBJRERC)

    # BJRERCのみ外した、PBJRERCのみ外した、どっちも外したリスト
    incorrect_BJRE = incorrect_data_BJRE - incorrect_data_PBJRE
    incorrect_PBJRE = incorrect_data_PBJRE - incorrect_data_BJRE
    incorrect_both = incorrect_data_PBJRE & incorrect_data_BJRE

    # jsonファイルで出力
    incorrect_dir = 'incorrect_data/'+path_BJRE+'vs'+path_PBJRE
    i = 1
    while os.path.isdir(incorrect_dir):
        i += 1
        incorrect_dir = incorrect_dir+'_'+str(i)
    filenameB = incorrect_dir+'/incorrect_BJRE.json'
    filenameP = incorrect_dir+'/incorrect_PBJRE.json'
    filenameb = incorrect_dir+'/incorrect_both.json'
    with open(filenameB, 'w', encoding='utf-8') as f:
        json.dump(incorrect_BJRE, f, indent=4, ensure_ascii=False)
    with open(filenameP, 'w', encoding='utf-8') as f:
        json.dump(incorrect_PBJRE, f, indent=4, ensure_ascii=False)
    with open(filenameb, 'w', encoding='utf-8') as f:
        json.dump(incorrect_both, f, indent=4, ensure_ascii=False)\

if __name__ == "__main__":
    if len(sys.argv) == 2:
        list_incorrect(sys.argv[1]) # modelname
    if len (sys.argv) == 3:
        compare_models(sys.argv[1], sys.argv[2]) # BJRERC path, PBJRERC path
    else:
        test()