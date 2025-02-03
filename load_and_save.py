import sys
import os
from train import BertForJapaneseRecepientERC
# 親ディレクトリの絶対パスを `sys.path` に追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from PersonalizedBertForJapaneseRecipientERC.train import PersonalizedBertForJapaneseRecepientERC

def main(model_path, save_dir):
    if os.path.abspath(model_path).startswith(os.path.abspath('../BertForJapaneseRecipientERC')):
        model = BertForJapaneseRecepientERC.load_from_checkpoint(model_path)

    if os.path.abspath(model_path).startswith(os.path.abspath('../PersonalizedBertForJapaneseRecipientERC')):
        model = PersonalizedBertForJapaneseRecepientERC.load_from_checkpoint(model_path)

    model.model.save_pretrained(save_dir)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])