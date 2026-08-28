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
11. OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)（↓）
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](lecnote_dt04.md)

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

自作ライブラリ my_libs 内の各クラス（映像キャプチャ VideoCapture および OpenMMLab 顔認識統合クラス MyMMFace）を活用し、高精度な顔検出とキーポイント抽出（68点ランドマーク相当）を実装するための解説ドキュメントです。

<hr>

# OpenMMLab 顔検出・キーポイント抽出ライブラリ (my_mmface.py) の使い方

## 概要
- 本ドキュメントでは、`my_cap_av2.py` の `VideoCapture` クラスを用いて映像を取り込み、`my_mmface.py`（`MyMMFace` クラス）を使用して OpenMMLab (MMDetection / MMPose) による高精度な顔検出および顔キーポイント（68点ランドマーク相当）の抽出を行う方法について解説します。

## 前提条件
- `./my_libs/video_capture/my_cap_av2.py` 内の `VideoCapture` クラスを用いてカメラ映像を取り込みます。
- `./my_libs/detector/my_mmface.py` 内の `MyMMFace` クラスを使用し、 `OpenMMLab (MMDetection / MMPose)` による以下の機能を処理します。
  - **RTMDet-Face による高速・高精度な顔領域（BBox）検出**
  - **RTMPose-Face による顔キーポイント（68点ランドマーク相当）抽出**
  - **輪郭描画用の骨格接続データ自動生成**
  
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


