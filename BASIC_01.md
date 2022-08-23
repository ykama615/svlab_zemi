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
   - CameraSelectorのコンストラクタは次の通り
      ```python
      CameraSelector(dnum='デバイス番号', fps='FPS', size='描画画面サイズ', box='キャプチャエリア')
      ```
  - 次のサンプルは，プログラム引数でカメラやそのプロパティを指定できるようにしたものです

  ```python
  # -*- coding: utf-8 -*-
  import cv2
  import argparse
  import myCapture as mycap

  def main(args):
    cap = mycap.CameraSelector(args.device, args.fps, args.size, args.box)

    while cap.isOpened():
      ret, fnum, frame = cap.read()

      if ret:
        cv2.imshow("video", frame)
        if cv2.waitKey(int(1000/cap.fps)) == ord('q'):
          break

    cap.release()
    cv2.destroyAllWindows()

  if __name__=='__main__':
    parser = argparse.ArgumentParser(
      description="--name \'window_name\' \n --device \'camera_num(99 is screen capture)\' \n--fps num")
    parser.add_argument('--device', type=int,
       help="--device \'camera_num(99 is screen capture)\'")
    parser.add_argument('--fps', type=int)
    def stype(ssize): return list(map(int, ssize.split(',')))
    parser.add_argument('--size', type=stype, help="width,height")
    parser.add_argument('--box', type=stype, help="x,y,width,height")
    args = parser.parse_args()
    main(args)
  ```
  - 引数なしの場合， device 0 のカメラをデフォルト状態で起動します
  ```sh
  % python c_select.py 
  -----------------------------------------
  Camera( 0 )
  480.0 x 640.0 @ 30.0
  -----------------------------------------
  ```
  - 引数として \-\-dev オプションでデバイス番号，\-\-size オプションでカメラの画面サイズなどを指定して起動できます
  - 指定するオプションの数，順番は任意です
  - \-\-size や \-\-fpsに，デバイスに対応していない値が設定された場合，自動的にデフォルト値が読み込まれ，その結果が出力されます
  ```sh
  % python c_select.py --dev 1 --size 1280,720
  -----------------------------------------
  Camera( 1 )
  720.0 x 1280.0 @ 30.0
  -----------------------------------------

  % python c_select.py --fps 90　　←対応していない値を指定➡30FPSで起動した
  -----------------------------------------
  Camera( 0 )
  480.0 x 640.0 @ 30.0
  CAUTION: fps cannot be set to the specified value
  -----------------------------------------
  ```
  - \-\-dev オプションに 99 を指定するとデスクトップキャプチャモードで起動します
  - 任意のウィンドウのバーを Ctrl+Click するとそのウィンドウの，任意の位置で Shift+Click すると画面全体がキャプチャされます
  - オプションを追加することで，任意の位置から指定した幅と高さの領域のキャプチャも可能です
  ```sh
  % python c_select.py --dev 99
  -----------------------------------------
  ScreenCapture
  Ctrl+Click: Window Select
  Shift+Click: Area Select
  -----------------------------------------
  
  % python c_select.py --dev 99 --fps 90 --box 100,400,500,500
  -----------------------------------------
  ScreenCapture
  Ctrl+Click: Window Select
  Shift+Click: Area Select
  -----------------------------------------
  [100, 400, 500, 500]
  ```
