import random
from tqdm import tqdm
import json
import os
import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, get_scheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report

MODEL_NAME = 'tohoku-nlp/bert-base-japanese-whole-word-masking'
tokenizer =BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

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

class_frequency = [
    4450.92560337,
     1259.4576316,
     2719.97370069,
     1071.27600888,
     96.64192807,
     686.30217162,
     237.07288376,
     12.35007948
]

cf = torch.tensor(class_frequency).cuda()
ICFweight = 1 / cf
ICFweight = ICFweight / torch.sum(ICFweight)

def tokenize_pack(filename):
    # データセットの対話パック(対話データの配列)をトークン化して返す関数
    dataset_for_loader = []
    # データセットから対話データを読み込む
    data = {}
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 各対話をトークン化して追加
    if 'data' in data:
        for pack in data['data']:
            text1 = []
            text2 = []
            for talk in pack:
                # 1発話目(受信者の発話)と2発話目(送信者の発話)を取り出す（複数続いたら改行で繋げる）
                t1 = []
                t2 = []
                for utter in talk['talk']:
                    if utter['type'] == 1:
                        t1.append(utter['utter'])
                    if utter['type'] == 2:
                        t2.append(utter['utter'])
                text1.append('\n'.join(t1))
                text2.append('\n'.join(t2))
            # 1パック分の対話データをまとめてトークン化
            token=tokenizer(
                text1, text2,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt"
            )
            # ラベル付け
            labels = []
            for talk in pack:
                labels.append(talk['label'])
            # バリューをテンソル化して追加
            token['labels'] = torch.tensor(labels)
            dataset_for_loader.append(token)
    return dataset_for_loader

class BertForJapaneseRecepientERC(pl.LightningModule):
    def __init__(
        self,
        model_name=MODEL_NAME,
        num_labels=8,
        lr=0.001,
        weight_decay=0.01,
        dropout=None,
        warmup_steps=None,
        total_steps=None
    ):
        # model_name: 事前学習モデル
        # num_labels: ラベル数
        # lr: 学習率(特に指定なければAdamのデフォルト値を設定)
        # weight_decay: 重み減衰の強度(L2正則化のような役割)
        # dropout: 全結合層の前に適用するdropout率、デフォルトでNoneとしているがこの場合はconfig.hidden_dropout_probが適用される(0.1など)
        super().__init__()
        self.save_hyperparameters()

        # 事前学習モデルのロード
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, classifier_dropout=dropout)

    # 学習データを受け取って損失を返すメソッド
    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )
        # 損失関数にCEではなくICFを用いる
        loss_func = torch.nn.CrossEntropyLoss(weight = ICFweight)
        loss = loss_func(output.logits.view(-1,self.hparams.num_labels), batch['labels'].view(-1,self.hparams.num_labels))

        if self.trainer.accumulate_grad_batches is not None:
            if (batch_idx + 1) % self.trainer.accumulate_grad_batches == 0:
                self.log('train_loss', loss)
        else:
            self.log('train_loss', loss)
        return loss
    
    # 検証データを受け取って損失を返すメソッド
    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            token_type_ids=batch['token_type_ids']
        )
        loss_func = torch.nn.CrossEntropyLoss(weight = ICFweight)
        val_loss = loss_func(output.logits.view(-1,self.hparams.num_labels), batch['labels'].view(-1,self.hparams.num_labels))
        self.log('val_loss', val_loss)

    def on_train_batch_start(self, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr)

    def configure_optimizers(self):
        # オプティマイザーの指定
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        # ウォームアップを適用
        if self.hparams.warmup_steps is not None and self.hparams.total_steps is not None:
            scheduler = get_scheduler(
                name='linear',
                optimizer=optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.total_steps
            )
            if self.trainer.accumulate_grad_batches is not None:
                return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": self.trainer.accumulate_grad_batches}]
            return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
        else:
            return optimizer

def unpack_batch(batch):
    # (batch_size,pack_size)->(batch_size)
    collated_batch = {key: torch.cat([item[key] for item in batch], dim=0) for key in batch[0].keys()}
    return collated_batch

def main():
    # データセットから対話パック(対話データの配列)をトークン化
    # (num_packs,pack_size)
    dataset_train = tokenize_pack('./DatasetForExperiment2/DatasetTrain.json')
    dataset_val = tokenize_pack('./DatasetForExperiment2/DatasetVal.json')

    acc_batches = None # 累積勾配を適用するバッチサイズ(適用しないならNone)
    
    # データローダ作成(呼び出し時にパックをばらす)
    # (num_packs,pack_size)->(num_batches,batch_size*pack_size)
    dataloader_train = DataLoader(
        dataset_train, num_workers=2, batch_size=4,
        shuffle=True, collate_fn=unpack_batch
    )
    dataloader_val = DataLoader(
        dataset_val, num_workers=2, batch_size=4, collate_fn=unpack_batch
    )

    # ハイパーパラメータ
    max_epochs = 10 # 学習のエポック数
    total_steps = len(dataloader_train) * max_epochs
    if acc_batches is not None:
        total_steps /= acc_batches
    warmup_steps = int(0.1 * total_steps) # ウォームアップの適用期間
    lr = 3e-5 # 初期学習率
    wd = 0.1 # 重み減衰率
    dropout = 0.1 # 全結合前のドロップアウト率

    # ファインチューニングの設定
    checkpoint = ModelCheckpoint(
        monitor = 'val_loss',
        mode = 'min', # monitorの値が小さいモデルを保存
        save_top_k = 1, # 最小のモデルだけ保存
        save_weights_only = True, # モデルの重みのみを保存
        dirpath='model/' # 保存先
    )
    # early stopping の設定
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,  # 3エポック性能が向上しなければ停止
        mode='min'
    )
    # 学習方法の指定
    trainer = pl.Trainer(
        accumulate_grad_batches = acc_batches, # 累積勾配4ステップ分
        accelerator = 'gpu', # 学習にgpuを使用
        devices = 1, # gpuの個数
        max_epochs = max_epochs, # 学習のエポック数
        callbacks = [checkpoint, early_stopping],
    )

    # ハイパーパラメータを指定してモデルをロード
    model = BertForJapaneseRecepientERC(
        lr=lr,
        weight_decay=wd,
        dropout=dropout,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    # ファインチューニング
    trainer.fit(model, dataloader_train, dataloader_val)

    # 結果表示
    print('best_model_path:', checkpoint.best_model_path)
    print('val_loss for best_model:', checkpoint.best_model_score)

    best_model = BertForJapaneseRecepientERC.load_from_checkpoint(
        checkpoint.best_model_path
    )
    best_model.model.save_pretrained('./model_transformers')

if __name__ == "__main__":
    main()