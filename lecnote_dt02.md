# OpenMMLab 顔検出・キーポイント抽出ライブラリ (my_mmface.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、`my_cap_av2.py` の `VideoCapture` クラスを用いて映像を取り込み、`my_mmface.py`（`MyMMFace` クラス）を使用して OpenMMLab (MMDetection / MMPose) による高精度な顔検出および顔キーポイント（68点ランドマーク相当）の抽出を行う方法について解説します[cite: 7]。

## 前提条件
- デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
- **【重要】** `my_cap_av2.py` および `my_mmface.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください[cite: 7]。
- **【重要】** MMDetection / MMPose 用の設定ファイル（`.py`）および重みファイル（`.pth`）が `C:\oit\home\ipbl\learned_models\mmpose\` 内に配置されている必要があります[cite: 7]。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

```

---

## :red_square: my_mmface.py の概要と特徴

`MyMMFace` は、OpenMMLab の物体検出ライブラリ（MMDetection）と姿勢推定ライブラリ（MMPose）を組み合わせ、Top-down 方式で高精度な顔バウンディングボックス検出とキーポイント検出を実行するカスタムクラスです。

### 主な特徴

1. **RTMDet-Face による顔検出 (`getFaceDet`)**:
* 超軽量・高速な RTMDet-Nano モデルを利用して画像中の顔領域（BBox）とその信頼度スコアを取得します。




2. **RTMPose-Face によるキーポイント抽出 (`getFacePose`)**:
* 検出された顔領域を入力として、高精度な 68 点相当の顔キーポイント（目・眉・鼻・口・輪郭）と各点の信頼度スコアを抽出します。




3. **描画用の接続データ生成 (`get_connection`)**:
* 抽出されたキーポイント間を結ぶ骨格ライン（顎線、眉、鼻、目、唇）のインデックスペアリストを自動生成し、輪郭描画を容易にします。




4. **計算デバイス（CPU/GPU）の切り替え**:
* 初期化時の `device` 引数（例: `'cpu'` や `'cuda:0'`）により、実行環境に応じた推論デバイスを選択可能です。





---

## :red_square: my_cap_av2 と連携した基本サンプルコード

`my_cap_av2.py` の `VideoCapture` で映像を入力し、`MyMMFace` で顔とキーポイントをリアルタイム検出し描画する基本プログラムです。

### mm_face_viewer.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np

# ライブラリのインポート
from my_libs.my_cap_av2 import VideoCapture
from my_libs.my_mmface import MyMMFace

device_input = 0 # 0: Webカメラ, または動画ファイルパス指定

def main():
    global device_input

    # 1. キャプチャと MMFace の初期化 (GPUを使用する場合は device='cuda:0' に変更)
    cap = VideoCapture(device_input)
    mm_face = MyMMFace(device='cpu')

    # キーポイント接続線のインデックスリストを取得
    connections = mm_face.get_connection()

    print("MMFace Landmark Tracking Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. 顔バウンディングボックスの検出
        bbox, score = mm_face.getFaceDet(frame)

        if bbox is not None and score > 0.5:
            x1, y1, x2, y2 = bbox.astype(int)
            # 顔枠の描画
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {score:.2f}", (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 3. 顔キーポイントの抽出
            kpts, kpt_scores = mm_face.getFacePose(frame, bbox)

            if kpts is not None:
                # 骨格（接続線）の描画
                for start_idx, end_idx in connections:
                    if kpt_scores[start_idx] > 0.3 and kpt_scores[end_idx] > 0.3:
                        pt1 = tuple(kpts[start_idx].astype(int))
                        pt2 = tuple(kpts[end_idx].astype(int))
                        cv2.line(frame, pt1, pt2, (255, 200, 0), 1)

                # キーポイント（点）の描画
                for pt, k_score in zip(kpts, kpt_scores):
                    if k_score > 0.3:
                        cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 0, 255), -1)

        # 4. 画面表示
        cv2.imshow("MMFace Detection & Landmark (my_cap_av2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---

## :red_square: MyMMFace の主なメソッド一覧

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__(device='cpu')` | `device`: 計算デバイス | なし | RTMDet-Face および RTMPose-Face の設定・モデルを初期化

 |
| `getFaceDet(frame)` | `frame`: BGR画像 (`ndarray`) | `(bbox, score)` | 最高スコアの顔枠 `[x1, y1, x2, y2]` と信頼度スコアを返却。未検出時は `(None, None)`<br> |
| `getFacePose(frame, bbox)` | `frame`: BGR画像, `bbox`: 顔枠 | `(kpts, scores)` | 顔の 68 点キーポイント座標配列と各点のスコアを返却

 |
| `get_connection()` | なし | `connections` | 輪郭・目・口・鼻を結ぶインデックスペアリスト `[(0, 1), ...]` を取得

 |

---

## :red_square: 演習 (`mm_face_crop.py`)

* [`mm_face_viewer.py`](https://www.google.com/search?q=%23mm_face_viewerpy) を参考にして、検出された顔領域（BBox）をトリミングし、別ウィンドウ `"Cropped Face"` に拡大表示する `mm_face_crop.py` を作成してください。


* **ヒントコード**:
```python
bbox, score = mm_face.getFaceDet(frame)

if bbox is not None and score > 0.5:
    x1, y1, x2, y2 = bbox.astype(int)

    # 画面枠外へのはみ出しを防止
    ht, wt, _ = frame.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(wt, x2), min(ht, y2)

    # 顔部分のクロップ
    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size > 0:
        cv2.imshow("Cropped Face", face_crop)

```



---

[トップページへ戻る]()
