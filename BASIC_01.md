<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. カメラへのアクセスと動画処理（↓）
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)

<hr>

# カメラへのアクセス
 もっとも簡単なカメラへのアクセスのサンプルプログラムは以下の通りです．<br>
 - import osとos.environの行は，Webカメラへのアクセスを高速化するための手続きです
 - グローバル変数のdevにカメラのデバイス番号や動画ファイル名を指定します
 - \'q\'ボタンを押すとプログラムが終了します

  ```python
  # script4.py
  # -*- coding: utf-8 -*-  
  import cv2

  dev = 0

  def main():
    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
      ret, frame = cap.read()

      if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q') or ret == False:
        break

      cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

  if __name__=='__main__':
    main()
  ```

  # タイムラプス
  - 15フレームに1回（30FPSであれば0.5秒に1回）ずつ，フレームを deque に追加していくことでタイムラプス動画を作成してみましょう
  - \'q\' ボタンを押すことでタイムラプスの収録を終了し，収録した内容が再生されます
  - 色々なタイミングでフレームを収録していくプログラムに変更してみましょう
  
  ```python
  # -*- coding: utf-8 -*-
  import cv2
  import numpy as np
  from collections import deque # dequeの利用に必要

  dev = 0

  def main():
    cap = cv2.VideoCapture(dev)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timelapse = deque()
    fnum = 0

    while cap.isOpened():
      ret, frame = cap.read()
      if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q') or ret == False:
        break

      cv2.imshow("video", frame)
      if fnum%15==0:
        # (1)dequeの末尾にframeを追加する
        timelapse.append(frame)
      fnum = fnum + 1

    # dequeの内容を再生
    for frame in timelapse:
      cv2.imshow("timelapsex2", frame)
      if cv2.waitKey(int(1000/30)) & 0xFF == ord('q'):
        break

    cv2.waitKey(0)      
    cv2.destroyAllWindows()
    cap.release()

  if __name__ == '__main__':
    main()
  ```

  ## [エクストラ] 配布環境の自作ライブラリの利用
  配布環境には，カメラ制御と画面キャプチャを補助するライブラリ（パッケージ）が用意してあります．
   - mylibs\\myCapture パッケージ内の　camera_selector.py モジュール（CameraSelectorクラス）

  ```python
  ```
