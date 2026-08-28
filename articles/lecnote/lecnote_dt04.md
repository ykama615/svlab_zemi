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

7. [動画画像処理 (`my_cap_av2.py`)](lecnote_cap01.md)
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](lecnote_cap03.md)
</details>
   
<b>➡検出・推定（4項目）</b>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](lecnote_dt03.md)
13. dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)（↓）

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](lecnote_an04.md)
</details>

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>

自作ライブラリ `my_libs` 内の各クラス（映像キャプチャ `VideoCapture` および dlib 統合クラス `MyDlib`）を活用し、dlib による顔検出と 68 点顔ランドマーク抽出、高速トラッキングを実装するための解説ドキュメントです。

<hr>

# dlib 顔検出・68点ランドマーク抽出ライブラリ (my_dlib.py) の使い方

## 概要

- `./my_libs/video_capture/my_cap_av2.py` 内の `VideoCapture` クラスを用いてカメラ映像を取り込みます。
- `./my_libs/detector/my_dlib.py` 内の `MyDlib` クラスを使用し、dlib による以下の機能を処理します。
  - **HOG ベースの複数顔検出**
  - **68 点の顔ランドマーク抽出**
  - **テンプレートマッチングを用いた高速単一顔トラッキング（`get_single_face_fast`）**

---

## 前提条件

- **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
  - `my_cap_av2.py`: `./my_libs/video_capture/`
  - `my_dlib.py`: `./my_libs/detector/`
- **【重要】** 68点ランドマーク予測用モデルファイル `shape_predictor_68_face_landmarks.dat` が `./learned_models/dlib/` 内に正しく配置されている必要があります。

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
| `__init__()` | なし | なし | HOG 顔検出器および 68 点ランドマーク予測器を初期化 |
| `get_multiple_face(frame)` | `frame`: BGR画像 | `(dets, scores, idx)` | 画面内の全顔領域 (`dlib.rectangle` リスト) と検出スコアを取得 |
| `get_single_face_fast(frame)` | `frame`: BGR画像 | `([dets], [scores], [idx])` | テンプレートマッチングを併用して単一の顔を高速追跡 |
| `get_facemark(frame, dface)` | `frame`: BGR画像, `dface`: `dlib.rectangle` | `parts`: `ndarray` | 指定された顔領域内の 68 箇所キーポイント座標 `[[x, y], ...]` を返却 |

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
