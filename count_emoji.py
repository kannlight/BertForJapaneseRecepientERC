import re
import json

FILE_NAME = 'test.json'

# 絵文字のUnicode範囲を表す正規表現パターン
emoji_pattern = re.compile(
    "["
    "\U0001F300-\U0001F5FF"  # 天気・記号・絵文字
    "\U0001F600-\U0001F64F"  # 顔文字（スマイリーなど）
    "\U0001F680-\U0001F6FF"  # 乗り物・地図記号
    "\U0001F700-\U0001F77F"  # 科学関連
    "\U0001F780-\U0001F7FF"  # 地図記号・記号
    "\U0001F800-\U0001F8FF"  # 拡張記号
    "\U0001F900-\U0001F9FF"  # 人物・ジェスチャー
    "\U0001FA00-\U0001FA6F"  # 動物・手のジェスチャー
    "\U0001FA70-\U0001FAFF"  # アイコン・オブジェクト
    "\U00002702-\U000027B0"  # その他の絵文字（例: ✂, ✈）
    "\U000024C2-\U0001F251"  # 絵文字バッジ（Ⓜ など）
    "]+", flags=re.UNICODE)


# 絵文字が含まれているか判定する関数
def contains_emoji(text):
    return bool(emoji_pattern.search(text))
if __name__ == '__main__':
  data_origin = open(FILE_NAME, 'r')
  data = json.load(data_origin)
  emoji_count = 0
  for taiwa_pack in data["data"]:
    for taiwa in taiwa_pack:
      for hatuwa in taiwa["talk"]:
        # if hatuwa["type"] == 2:
          # print(hatuwa["utter"])
        if hatuwa["type"] == 2 and contains_emoji(hatuwa["utter"]):
          emoji_count += 1
  print(f"絵文字が含まれている発話数: {emoji_count}")