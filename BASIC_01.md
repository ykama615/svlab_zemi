<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. カメラ，顔・手・ポーズ検出（↓）

<hr>

# カメラへのアクセス
 もっとも簡単なカメラへのアクセスのサンプルプログラムは以下の通りです．<br>
 - import osとos.environの行は，Webカメラへのアクセスを高速化するための手続きです
 - グローバル変数のdevにカメラのデバイス番号や動画ファイル名を指定します
 - \'q\'ボタンを押すとプログラムが終了します

  ```python
  # -*- coding: utf-8 -*-
  import os
  os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
  import cv2

  dev = 0

  def main():
    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
      ret, frame = cap.read()

      if ret:
        cv2.imshow("video", frame)

      if cv2.waitKey(int(1000/fps))==ord('q'):
        break

    cv2.destroyAllWindows()
    cap.release()

  if __name__=='__main__':
      main()
  ```