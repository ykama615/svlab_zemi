<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. カメラへのアクセスと動画処理（↓）
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. [各種クラス・応用](BASIC_04.md)

<hr>

# カメラへのアクセス
 もっとも簡単なカメラへのアクセスのサンプルプログラムは以下の通りです．<br>
 - import osとos.environの行は，Webカメラへのアクセスを高速化するための手続きです
 - グローバル変数のdevにカメラのデバイス番号や動画ファイル名を指定します
 - \'q\'ボタンを押すとプログラムが終了します

  ```python
  # script4.py
  import cv2

  dev = 0

  def main():
    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
      ret, frame = cap.read()

      if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

      cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

  if __name__=='__main__':
    main()
  ```

  #### 内蔵カメラ以外のWebカメラ等を利用する場合
  USB接続のカメラの場合，cv2.VideoCaptureによるカメラの起動が遅くなります．これを回避するためにメインプログラムの先頭（import cv2より前）に以下の2行を記述します．
  ```python
  # cv2のインポート前にカメラに関する設定を行う
  import os
  os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

  import cv2
  ```

  ## [課題] Let's Selfy プログラム
  キー\'s\'を押すと，そのときのフレームが表示・保存されるプログラムを作成してみましょう．
  - ヒント
  ```python
  key = cv2.waitKey(1)
  if key & 0xFF == ord('q'):
    break
  elif key & 0xFF == ord('s'):
    cv2.imshow("image", frame)
    cv2.imwrite("photo.jpg", frame)
  ```

  ## [課題] グレースケール・ビデオ・プログラム
  映像がグレースケールに変換されるようにするプログラムを作成してみましょう．
  - ヒント
  ```python
  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  ```

  ## [課題] エッジ・ビデオ・プログラム
  グレースケール・ビデオ・プログラムにエッジ検出処理を追加してみましょう．
  - エッジとは画像中の色や明るさが極端に変化する（不連続の）部分のことで，写っているモノとモノの境界を示します
  - ヒント
  ```python
  cv2.Canny(gray, 100, 200)
  ```
  
  # タイムラプス
  - 15フレームに1回（30FPSであれば0.5秒に1回）ずつ，フレームを deque に追加していくことでタイムラプス動画を作成してみましょう
  - \'q\' ボタンを押すことでタイムラプスの収録を終了し，収録した内容が再生されます
  - 色々なタイミングでフレームを収録していくプログラムに変更してみましょう
  
  ```python
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
  <!--
  ## [エクストラ] 配布環境の自作ライブラリの利用
  配布環境には，カメラ制御と画面キャプチャを補助するライブラリ（パッケージ）が用意してあります．
   - mylibs\\myCapture パッケージ内の　camera_selector.py モジュール（CameraSelectorクラス）
   - CameraSelectorのコンストラクタは次の通り
      ```python
      CameraSelector(dnum='デバイス番号', fps='FPS', size='描画画面サイズ', box='キャプチャエリア')
      ```
   - readメソッドの戻り値は次の通り
     - カメラの起動時間とFPSから推定したフレーム番号が返却されます 
      ```python
      [フレーム取得の成否, フレーム番号, フレームデータ] = cap.read()
      ```
  - 次のサンプルは，プログラム引数でカメラやそのプロパティを指定できるようにしたものです

  ```python
  import cv2
  import argparse
  import myCapture as mycap

  def main(args):
    cap = mycap.CameraSelector(args.device, args.fps, args.size, args.box)

    while cap.isOpened():
      ret, fnum, frame = cap.read()

      if ret:
        cv2.imshow("video", frame)
        if cv2.waitKey(1) == ord('q'):
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
  -->
