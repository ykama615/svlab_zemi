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
12. OpenMMLab 統合姿勢推定 (`my_mmpose.py`)（↓）
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

自作ライブラリ `my_libs` 内の各クラス（映像キャプチャ `VideoCapture` および OpenMMLab 姿勢推定クラス `MyMMPose`）を活用し、全身・手・足・顔などの統合姿勢推定（Pose Estimation）を実装するための解説ドキュメントです。

<hr>

# OpenMMLab 統合姿勢推定ライブラリ (my_mmpose.py) の使い方

## 概要

- `./my_libs/video_capture/my_cap_av2.py` 内の `VideoCapture` クラスを用いてカメラ映像を取り込みます。
- `./my_libs/detector/my_mmpose.py` 内の `MyMMPose` クラスを使用し、OpenMMLab (RTMDet / RTMPose) による以下の姿勢推定機能を処理します。
  - **身体 17 箇所キーポイントの標準検出（`coco` モード）**
  - **全身・手・足・顔を含む全 133 箇所の一括検出（`whole` モード）**
  - **人物検出器（RTMDet）と姿勢推定器（RTMPose）の分離制御**
  - **関節存在確率ヒートマップの出力**

---

## 前提条件

- **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
  - `my_cap_av2.py`: `./my_libs/video_capture/`
  - `my_mmpose.py`: `./my_libs/detector/`
- **【重要】** MMDetection / MMPose 用の設定ファイル（`.py`）および重みファイル（`.pth`）が `./learned_models/mmpose/` 内に正しく配置されている必要があります。

---

## :red_square: my_mmpose.py の概要と特徴

`MyMMPose` は、RTMDet（人物検出器）と RTMPose（姿勢推定器）を分離制御し、柔軟な処理設計を可能にした高度な姿勢推定ラップクラスです。

### 主な特徴

1. **2つの動作モード（`coco` / `whole`）**:
* `model='coco'`: 身体の主要 17 関節点を高速に推論します。


* `model='whole'`: 身体（17）、足（6）、顔（68）、手（42）を含む最大 133 箇所の Wholebody キーポイントを一括抽出します。




2. **Detector と Pose の分離設計**:
* `get_RTMDet()` で人物の BBox を検出し、`get_RTMPose()` でその BBox 内の姿勢のみを推定できます。毎フレーム Detector を回さずトラッキングと組み合わせることで処理を軽量化できます。




3. **キーポイント接続データ（Skeleton）の自動生成**:
* 身体・足・手・顔のパーツ別に、可視化用の骨格接続ペアリスト（`get_pose_connections` 等）を自動取得できます。




4. **ヒートマップ（確率マップ）出力・補助機能**:
* 推論オプション（`heatmap=True`）により、各関節の存在確率を示すヒートマップ（JETカラーマップによる疑似カラー可視化画像）の出力機能や、顔キーポイントから顔用 BBox を逆算する `make_face_bbox()` 等のユーティリティを備えています。





---

## :red_square: my_cap_av2 と連携した基本サンプルコード

`my_cap_av2.py` の `VideoCapture` で映像を入力し、 Wholebody モード（`model='whole'`）で全身・手・足の関節点を描画する基本プログラムです。

### mm_pose_viewer.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2

# ライブラリのインポート
from my_libs.my_cap_av2 import VideoCapture
from my_libs.my_mmpose import MyMMPose

device_input = 0 # 0: Webカメラ, または動画ファイルパス指定

def main():
    global device_input

    # 1. キャプチャと MyMMPose の初期化 (全身モデル)
    cap = VideoCapture(device_input)
    mm_pose = MyMMPose(device='cpu', model='whole') # GPUの場合は device='cuda:0'[cite: 8]

    # 骨格接続情報の取得
    pose_conn = mm_pose.get_pose_connections()[cite: 8]
    hand_conn = mm_pose.get_hand_connections()[cite: 8]

    print("MMPose Wholebody Tracking Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. 人物バウンディングボックスの検出
        bbox, score = mm_pose.get_RTMDet(frame)[cite: 8]

        if bbox is not None and score > 0.5:
            # 3. 姿勢推定を実行
            res = mm_pose.get_RTMPose(frame, bbox)[cite: 8]

            if res is not None:
                # 身体（Pose）の描画
                kpts, k_scores = res['pose'][cite: 8]
                for p1, p2 in pose_conn:
                    if k_scores[p1] > 0.3 and k_scores[p2] > 0.3:
                        pt1 = tuple(kpts[p1].astype(int))
                        pt2 = tuple(kpts[p2].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # 手（Hands）の描画
                hands_kpts, hands_scores = res['hands'][cite: 8]
                if hands_kpts is not None:
                    for side in ['left', 'right']:
                        h_kpts = hands_kpts[side][cite: 8]
                        h_scores = hands_scores[side][cite: 8]
                        for p1, p2 in hand_conn[side]:
                            if h_scores[p1] > 0.3 and h_scores[p2] > 0.3:
                                pt1 = tuple(h_kpts[p1].astype(int))
                                pt2 = tuple(h_kpts[p2].astype(int))
                                cv2.line(frame, pt1, pt2, (255, 128, 0), 1)

        # 4. 画面表示
        cv2.imshow("MMPose Wholebody (my_cap_av2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---

## :red_square: MyMMPose の主なメソッド一覧

### モデル・推論メソッド

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__(device, model, heatmap)` | `device='cpu'`, `model='coco'`, `heatmap=False` | なし | モデル読み込みと初期化 (`model='whole'` で 133 点モード) |
| `get_RTMDet(frame)` | `frame`: BGR画像 | `(bbox, score)` | 最高スコアの人物 BBox `[x1, y1, x2, y2]` とスコアを返却 |
| `get_RTMPose(frame, bbox)` | `frame`: BGR画像, `bbox`: 人物BBox | `dict` | 姿勢データを保持する辞書 (`'pose'`, `'face'`, `'hands'`, `'feet'`, `'heatmap'`) を返却 |

### パーツ別データ取得ゲッター

| メソッド | 戻り値 | 説明 |
| --- | --- | --- |
| `get_RTMFace()` | `(face_kpts, face_scores)` | `whole` モード時の顔 68 点座標とスコアを取得 |
| `get_RTMHands()` | `(hand_kpts, hand_scores)` | `whole` モード時の左右手（各21点）座標とスコアの辞書を取得 |
| `get_RTMFeet()` | `(foot_kpts, foot_scores)` | `whole` モード時の左右足（各3点）座標とスコアの辞書を取得 |

### 骨格接続取得・補助ユーティリティ

| メソッド | 説明 |
| --- | --- |
| `get_pose_connections()` | 身体主要 17 関節の接続ペアリストを取得 |
| `get_hand_connections()` | 手（21点）の接続ペアリスト（左右別辞書）を取得 |
| `get_face_connections()` | 顔（68点）の輪郭・パーツ接続ペアリストを取得 |
| `get_foot_connections()` | 足裏（3点）の三角形接続ペアリストを取得 |
| `make_face_bbox(kpts, scores)` | 身体キーポイント（目・耳・鼻）から顔部分の拡大 BBox を算出 |

---

## :red_square: 演習 (`mm_pose_heatmap.py`)

* [`mm_pose_viewer.py`](https://www.google.com/search?q=%23mm_pose_viewerpy) をベースに、初期化時に `heatmap=True` を設定して推論を行い、`get_RTMPose()` の返却値に含まれる `'heatmap'`（関節の存在確率を示す確信度マップ画像）を別ウィンドウ `"Joint Probability Heatmap"` で表示するプログラムを作成してください。


* **ヒントコード**:
```python
# ヒートマップ（確率マップ）描画有効で初期化
mm_pose = MyMMPose(device='cpu', model='coco', heatmap=True)

# 推論結果から関節確信度マップ画像（JETカラーマップ表示）を取得
res = mm_pose.get_RTMPose(frame, bbox)
if res is not None and res['heatmap'] is not None:
    cv2.imshow("Joint Probability Heatmap", res['heatmap'])

```


[トップページへ戻る]()
