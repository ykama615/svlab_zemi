# 仮想入力装置

## pyautogui を用いた処理
  1. pyautogui でアイコンからアプリケーションの起動
     - pyautogui.locationOnScreen 関数は指定された画像のスクリーン上での領域を返します
        - デスクトップ上にある下記アイコンを検索する例です（検出率を上げるため，アイコンの一部だけを画像化しています）
       
          | アイコン画像 |
          |:--:|
          | ![mcico2.png](./mcico2.png) |
       
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

  2. pyautogui でキー入力
     - win32gui モジュールを利用してウィンドウを最前面に表示し，アクティブにします
     - pyautogui.press 関数を利用してキー入力を行います

  ```python
  # -*- coding: utf-8 -*-
  import win32gui, win32con
  import pyautogui as agui

  def main():
    # ウィンドウハンドルを取得
    whand = win32gui.FindWindow(None, "タイトルなし - メモ帳")
    if whand==0:
      return

    # 最小化を解除
    win32gui.ShowWindow(whand, win32con.SW_RESTORE )

    # ウィンドウを左上に固定して最前面に
    win32gui.SetWindowPos(whand, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOSIZE)

    # ウィンドウ領域を取得して，メニューバーの中央あたりをクリック
    # win32gui.SetForegroundWindow(whand)でもアクティブ化可能
    box = win32gui.GetWindowRect(whand)
    agui.click(box[0]+box[2]//2, box[1]+10)

    agui.press('H')
    agui.press('e')
    agui.press('l')
    agui.press('l')
    agui.press('o')
    agui.press(' ')
    agui.press('P')
    agui.press('y')
    agui.press('t')
    agui.press('h')
    agui.press('o')
    agui.press('n')
    agui.press('!')


  if __name__=='__main__':
      main()
  ```
