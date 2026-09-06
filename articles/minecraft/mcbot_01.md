<hr>

**講義ノート・ライブラリ一覧**

<details><summary><b>基礎編（6項目）</b></summary>
  
1. [環境の設定](../../README.md)
2. [基本概要](../basic/BASIC_00.md)
3. [カメラへのアクセスと動画処理](../basic/BASIC_01.md)
4. [顔と顔パーツの検出](../basic/BASIC_02.md)
5. [顔・手・ポーズ検出](../basic/BASIC_03.md)
6. [2つのベクトルのなす角とベクトル演算](../basic/BASIC_FP01.md)
</details>

<details><summary><b>キャプチャ（3項目）</b></summary>
  
7. [動画画像処理 (`my_cap_av2.py`)](../lecnote/lecnote_cap01.md)
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](../lecnote/lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](../lecnote/lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](../lecnote/lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](../lecnote/lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](../lecnote/lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](../lecnote/lecnote_dt04.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](../lecnote/lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](../lecnote/lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](../lecnote/lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](../lecnote/lecnote_an04.md)
</details>

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](../lecnote/lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](../lecnote/lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](../lecnote/lecnote_tl03.md)
</details>

<b>➡その他（3項目）</b>

21. [ドローン/RTPキャプチャ](../drone/lecnote_dr01.md)
22. [スクリーンキャプチャ](../minecraft/lecnote_mc01.md)
23. 【旧】Minecraftコントロール(1)（↓）

<hr>

旧バージョン(2023頃)，未編集

<hr>

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
        - 未編集のメモ帳を開いておきましょう
        - 以下のサンプルを実行するとメモ帳のウィンドウが最前面に表示され，Hello python!と入力，改行されます
     - pyautogui.pressの引数では，\'enter\'，\'esc\'，\'alt\' などが使用できますが，\'@\'，\'\^\'，\'\:\' を入力することができません
        -  \_pyautogui\_win.py（WPy64-39100\\pyton-3.9.10.amd64\\Lib¥¥site-packages\\pyautogui\\）の\_keyDown(key)関数に以下のif文3行を追記しましょう

          ```python
          def _keyDown(key):
              ##(略)##

              needsShift = pyautogui.isShiftCharacter(key)

              if key == '@': needsShift = False
              if key == '^': needsShift = False
              if key == ':': needsShift = False

              """
              # OLD CODE: The new code relies on having all keys be loaded in keyboardMapping from the start.
          ```


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
    agui.press('enter')

  if __name__=='__main__':
      main()
  ```

# [MCBOT]移動
  - キー押下を threading.Timer で管理することで動作中に別動作を割り込めるようにしています
  - threading.Timer 関数の時間管理が不正確なため，引数の秒数も正確ではありません
      - 第3引数 count は指定不要です 
      ```python
      move(動作の継続時間, 入力キー, count=0)
      ```
  
  ```python
  # -*- coding: utf-8 -*-
  import win32gui, win32con
  import pyautogui as agui
  import threading
  import time

  def move(sec, c, count=0):
    global t

    if sec*10<=count:
      t.cancel()
      agui.keyUp(c)
    else:
      agui.keyDown(c) 
      t = threading.Timer(0.005, move, (sec, c, count+1))
      t.start()

  def main():
    # ウィンドウハンドルを取得
    whand = win32gui.FindWindow(None, "Minecraft: Education Edition")
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

    agui.press('esc')

    time.sleep(5) #起動待ち

    move(5, 'w') #5秒前進

    time.sleep(2) #2秒待ち

    move(3, 'a') #前進開始2秒後から左移動を追加

    print("おわり")

  if __name__=='__main__':
      main()
  ```

