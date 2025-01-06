import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import pytorch_lightning as pl
from sklearn.metrics import classification_report

MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'
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

class BertForJapaneseRecepientERC(pl.LightningModule):
    def __init__(self, lr, model_name=MODEL_NAME, num_labels=8):
        # model_name: 事前学習モデル
        # num_labels: ラベル数
        # lr: 学習率
        super().__init__()
        self.save_hyperparameters()

        # 事前学習モデルのロード
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # 学習データを受け取って損失を返すメソッド
    def training_step(self, batch):
        output = self.model(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
    
    # 検証データを受け取って損失を返すメソッド
    def validation_step(self, batch):
        output = self.model(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)

    # テストデータを受け取って評価指標を計算
    def test_step(self, batch):
        # モデルが出力した分類スコアから、最大値となるクラスを取得
        output =self.model(**batch)
        predicted_labels = output.logits.argmax(-1)
        # テストデータのラベル
        true_labels = batch.pop('labels')
        # precision, recall, f1, データ数 をクラス毎、ミクロ、マクロ、加重平均で算出
        self.log('test_report', classification_report(true_labels, predicted_labels, target_names=CATEGORIES))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


utter1 = ''
utter2 = ''

token=tokenizer(
                    utter1, utter2,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding="max_length",
                )

token = { k: torch.tensor(v) for k, v in token.items() }

# 作成中

model = BertForJapaneseRecepientERC.load_from_checkpoint(
    'C:\Users\hsym-\kanno\BertForJapaneseRecipientERC\model\epoch=1-step=396.ckpt'
)
