# 仮想入力装置

## pyautogui を用いた処理
  1. pyautogui でアイコンからアプリケーションの起動
   - pyautogui.locationOnScreen 関数は指定された画像のスクリーン上での領域を返します
      - デスクトップ上にある下記アイコンを検索する例です（検出率を上げるため，アイコンの一部だけを画像化しています）
          ![mcico2.png](./mcico2.png)
          
      - サイズの違い，ボケやアーティファクト（チェックマークやショートカットマーク）など変化，背景の違いには対応できません
      - 別のウィンドウなどでアイコンが隠れていると検出できません
   - pyautogui.doubleClick 関数を用いて検出した領域の中心をダブルクリックしてアプリケーションを起動します

  ```python
  # -*- coding: utf-8 -*-
  import pyautogui as agui

  def main():
    box = agui.locateOnScreen("./img/mcico2.png", confidence=0.8)
    print(box)
    if box is None:
      print("ICON cannot find")
    else:
      agui.doubleClick(box[0]+box[2]//2, box[1]+box[3]//2)

  if __name__=='__main__':
      main()
  ```
