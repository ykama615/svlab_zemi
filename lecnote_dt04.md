# dlib 顔検出・68点ランドマーク抽出ライブラリ (my_dlib.py) の使い方

[トップページへ戻る](https://www.google.com/search?q=../README.md)

---

## 目的

* 本ドキュメントでは、`my_cap_av2.py` の `VideoCapture` クラスを用いて映像を取り込み、`my_dlib.py`（`MyDlib` クラス）を使用して dlib による顔検出および 68 点の顔ランドマーク抽出を行う方法について解説します。


* テンプレートマッチングを用いた高速な単一顔トラッキング機能（`get_single_face_fast`）の活用方法についても説明します。



## 前提条件

* デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
* **【重要】** `my_cap_av2.py` および `my_dlib.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください。


* **【重要】** 68点ランドマーク予測用モデルファイル `shape_predictor_68_face_landmarks.dat` が `C:\oit\home\ipbl\learned_models\` 内に配置されている必要があります。


* ターミナルで以下のコマンドを実行してプログラムを動作させます。
```sh
C:\oit\home\ipbl> python XXX.py

```



---

## :red_square: my_dlib.py の概要と特徴

`MyDlib` は、機械学習ライブラリ `dlib` の HOG (Histogram of Oriented Gradients) ベースの顔検出器および 68 点ランドマーク推論モデル（`shape_predictor`）を Python から手軽に利用するためのラップクラスです。

### 主な特徴

1. **標準 HOG 顔検出 (`get_multiple_face`)**:
* `dlib.get_frontal_face_detector()` を利用し、画像内の複数の顔領域（`dlib.rectangle`）と信頼度スコアを一括取得します。




2. **テンプレートマッチングによる高速トラッキング (`get_single_face_fast`)**:
* 毎フレーム重い HOG 検出を実行する代わりに、前フレームで検出した顔画像をテンプレートとして `cv2.matchTemplate` で追跡します。スコアが閾値（0.7）を下回った場合のみ再検出を行うことで処理を大幅に高速化します。




3. **68 点ランドマークの NumPy 配列変換 (`get_facemark`)**:
* `imutils.face_utils.shape_to_np` を内部で使用し、抽出したランドマーク座標を OpenCV 等で扱いやすい `(68, 2)` の NumPy 配列（`int32`）として返却します。





---

## :red_square: my_cap_av2 と連携した基本サンプルコード

`my_cap_av2.py` の `VideoCapture` で映像を入力し、複数人の顔検出と 68 点のランドマーク描画を行う基本プログラムです。

### dlib_face_viewer.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

# ライブラリのインポート
from my_libs.my_cap_av2 import VideoCapture
from my_libs.my_dlib import MyDlib

device_input = 0 # 0: Webカメラ, または動画ファイルパス指定

def main():
    global device_input

    # 1. キャプチャと MyDlib の初期化
    cap = VideoCapture(device_input)
    mydlib = MyDlib()[cite: 9]

    print("dlib Face & Landmark Tracking Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. 複数顔の検出
        dets, scores, idx = mydlib.get_multiple_face(frame)[cite: 9]

        if len(dets) > 0:
            for i, dface in enumerate(dets):
                # 顔バウンディングボックスの描画
                x1, y1, x2, y2 = dface.left(), dface.top(), dface.right(), dface.bottom()[cite: 9]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)[cite: 9]

                # 3. 68点顔ランドマークの取得と描画
                parts = mydlib.get_facemark(frame, dface)[cite: 9]
                for p in parts:
                    cv2.circle(frame, (p[0], p[1]), 2, (0, 255, 0), -1)[cite: 9]

        # 4. 画面表示
        cv2.imshow("dlib Face Landmarks (my_cap_av2)", frame)[cite: 9]

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---

## :red_square: MyDlib の主なメソッド一覧

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__()` | なし | なし | HOG 顔検出器および 68 点ランドマーク予測器を初期化

 |
| `get_multiple_face(frame)` | `frame`: BGR画像 | `(dets, scores, idx)` | 画面内の全顔領域 (`dlib.rectangle` リスト) と検出スコアを取得

 |
| `get_single_face_fast(frame)` | `frame`: BGR画像 | `([dets], [scores], [idx])` | テンプレートマッチングを併用して単一の顔を高速追跡

 |
| `get_facemark(frame, dface)` | `frame`: BGR画像, `dface`: `dlib.rectangle` | `parts`: `ndarray` | 指定された顔領域内の 68 箇所キーポイント座標 `[[x, y], ...]` を返却

 |

---

## :red_square: 演習 (`dlib_fast_tracking.py`)

* [`dlib_face_viewer.py`](https://www.google.com/search?q=%23dlib_face_viewerpy) をベースに、顔検出処理を `get_multiple_face()` から `get_single_face_fast()` に変更した `dlib_fast_tracking.py` を作成してください。


* トラッキング動作時のフレーム処理スピードや追跡の挙動の違いを確認してください。


* **ヒントコード**:
```python
# 毎フレーム重い検出を行わず、テンプレートマッチングで高速追跡
dets, scores, idx = mydlib.get_single_face_fast(frame)

if len(dets) > 0 and len(dets[0]) > 0 if isinstance(dets, list) else True:
    dface = dets[0]
    parts = mydlib.get_facemark(frame, dface)
    for p in parts:
        cv2.circle(frame, (p[0], p[1]), 2, (255, 255, 0), -1)

```



---

[トップページへ戻る]()
