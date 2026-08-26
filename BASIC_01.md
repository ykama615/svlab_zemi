<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. カメラへのアクセスと動画処理（↓）
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. [各種クラス・応用](BASIC_04.md)

<hr>

# カメラへのアクセス

もっとも簡単なカメラアクセスのサンプルプログラムは以下の通りです。
- グローバル変数 `dev` にカメラのデバイス番号（または動画ファイルパス）を指定します。
- 'q' キーを押すとプログラムが終了します。

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

        if not ret:
            break

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

### 内蔵カメラ以外のWebカメラ等を利用する場合

#### 1. MSMF(Microsoft Media Foundation)の設定

USB接続カメラの場合、`cv2.VideoCapture` によるカメラの起動が遅くなる場合があります。これを回避するため、メインプログラムの先頭（`import cv2` より前）に以下の設定を記述します。

```python
# cv2のインポート前にカメラ環境変数の設定を行う
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

```

#### 2. DirectShow経由で利用する方法

1 のほかに、`VideoCapture` の引数にバックエンドフラグを指定する方法も利用できます。

```python
# DirectShow経由でカメラ映像を取得する
cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)

```

---

# 課題・演習

## 【課題】 Let's Selfy プログラム

キー 's' を押すと、その瞬間のフレームを画面表示・保存するプログラムを作成してみましょう。

* **ヒント**:

```python
key = cv2.waitKey(1)
if key & 0xFF == ord('q'):
    break
elif key & 0xFF == ord('s'):
    cv2.imshow("image", frame)
    cv2.imwrite("photo.jpg", frame)

```

## 【課題】 グレースケール・ビデオ・プログラム

カメラ映像をリアルタイムでグレースケールに変換して表示するプログラムを作成してみましょう。

* **ヒント**:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

```

## 【課題】 エッジ・ビデオ・プログラム

グレースケール・ビデオ・プログラムに Canny 法によるエッジ検出処理を追加してみましょう。

* **エッジとは**: 画像中の色や明るさが極端に変化する（不連続な）部分のことで、写っている物体の境界線を示します。
* **ヒント**:

```python
edges = cv2.Canny(gray, 100, 200)

```

---

# タイムラプス

* 15フレームに1回（30 FPS であれば 0.5 秒に 1 回）ずつ、取得したフレームを `deque` に追加していくことでタイムラプス動画を作成してみましょう。
* 'q' キーを押すとタイムラプスの収録を終了し、収録されたフレームが連続再生されます。
* 収録タイミング（間隔）を変更して動作を確認してみましょう。

```python
import cv2
import numpy as np
from collections import deque

dev = 0

def main():
    cap = cv2.VideoCapture(dev)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # FPS取得失敗（0.0）時のゼロ除算対策
    wait_msec = int(1000.0 / fps) if fps > 0 else 33

    timelapse = deque()
    fnum = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("video", frame)
        
        # 15フレームごとにdequeの末尾へ追加
        if fnum % 15 == 0:
            timelapse.append(frame)
        fnum += 1

        if cv2.waitKey(wait_msec) & 0xFF == ord('q'):
            break

    # 収録したdequeの内容を再生
    for frame in timelapse:
        cv2.imshow("timelapse", frame)
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---

# 【エクストラ】 配布環境の自作ライブラリの利用

配布環境には、カメラ制御と画面キャプチャを補助するライブラリパッケージが用意されています。

* `mylibs/myCapture` パッケージ内 `camera_selector.py` モジュール（`CameraSelector` クラス）
* **コンストラクタ構造**:
```python
CameraSelector(dnum='デバイス番号', fps='FPS', size='描画画面サイズ', box='キャプチャエリア')

```


* **`read()` メソッドの戻り値**:
```python
ret, fnum, frame = cap.read()
# [フレーム取得成否(bool), 推定フレーム番号(int), フレーム画像データ]

```



### コマンドライン引数対応サンプルコード

```python
import cv2
import argparse
import mylibs.myCapture as mycap

def main(args):
    cap = mycap.CameraSelector(args.device, args.fps, args.size, args.box)

    while cap.isOpened():
        ret, fnum, frame = cap.read()

        if ret:
            cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="--device 'camera_num (99 is screen capture)'\n--fps num"
    )
    parser.add_argument('--device', type=int, default=0, help="カメラデバイス番号 (99: 画面キャプチャ)")
    parser.add_argument('--fps', type=int, help="フレームレート")
    
    def stype(ssize):
        return list(map(int, ssize.split(',')))
        
    parser.add_argument('--size', type=stype, help="幅,高さ (例: 1280,720)")
    parser.add_argument('--box', type=stype, help="x,y,幅,高さ")
    args = parser.parse_args()
    
    main(args)

```

* **デフォルトでの実行（デバイス 0）**:
```sh
% python c_select.py 
-----------------------------------------
Camera( 0 )
480.0 x 640.0 @ 30.0
-----------------------------------------

```


* **オプション指定例**:
```sh
% python c_select.py --device 1 --size 1280,720
-----------------------------------------
Camera( 1 )
720.0 x 1280.0 @ 30.0
-----------------------------------------

% python c_select.py --fps 90  # 未対応FPSを指定した場合、デフォルトに自動フォールバック
-----------------------------------------
Camera( 0 )
480.0 x 640.0 @ 30.0
CAUTION: fps cannot be set to the specified value
-----------------------------------------

```


* **デスクトップキャプチャモード (`--device 99`)**:
* 任意のウィンドウのバーを **Ctrl + Click** するとそのウィンドウをキャプチャ
* 画面上を **Shift + Click** すると画面全体を選択キャプチャ


```sh
% python c_select.py --device 99
-----------------------------------------
ScreenCapture
Ctrl+Click: Window Select
Shift+Click: Area Select
-----------------------------------------

% python c_select.py --device 99 --fps 90 --box 100,400,500,500
-----------------------------------------
ScreenCapture
Ctrl+Click: Window Select
Shift+Click: Area Select
-----------------------------------------
[100, 400, 500, 500]

```
