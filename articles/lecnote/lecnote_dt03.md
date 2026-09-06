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

<details><summary><b>その他（3項目）</b></summary>

21. [ドローン/RTPキャプチャ](../drone/lecnote_dr01.md)
22. [スクリーンキャプチャ](../minecraft/lecnote_mc01.md)
23. [【旧】Minecraftコントロール(1)](../minecraft/mcbot_01.md)    
</details>

<hr>

自作ライブラリ `my_libs` 内の各クラス（映像キャプチャ `VideoCapture` および OpenMMLab 姿勢推定クラス `MyMMPose`）を活用し、全身・手・足・顔などの統合姿勢推定（Pose Estimation）を実装するための解説ドキュメントです。

<hr>

# OpenMMLab 統合姿勢推定ライブラリ (my_mmpose.py) の使い方

## 概要

* `./my_libs/video_capture/my_cap_av2.py` 内の `VideoCapture` クラスを用いてカメラ映像を取り込みます。
* **【標準との違い】** OpenCV標準の `cv2.VideoCapture` でも同様のメソッドは存在しますが、環境や入力ソースによって経過時間の取得が不安定になったり正確性を欠く場合があります。本自作クラスでは `time.perf_counter()` や PTS を用いて一貫した正確なミリ秒タイムスタンプを担保し、実験ログの出力やイベント同期に活用します。


* `./my_libs/detector/my_mmpose.py` 内の `MyMMPose` クラスを使用し、OpenMMLab (MMDetection / MMPose) による以下の姿勢推定機能を処理します。
* **身体 17 箇所キーポイントの標準検出（`coco` モード）**
* **全身・手・足・顔を含む全 133 箇所の一括検出（`whole`, `dwpose`, `rtmw` モード）**
* **人物検出器（RTMDet）と姿勢推定器（RTMPose）の分離制御**
* **関節存在確率ヒートマップ（JETカラーマップ）の出力機能**



---

## 前提条件

* **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
* `my_cap_av2.py`: `./my_libs/video_capture/`
* `my_mmpose.py`: `./my_libs/detector/`


* **【重要】** OpenMMLab関連のモデルウェイトや実行環境が正しく構築されている必要があります。

---

## :red_square: my_mmpose.py の概要と特徴

`MyMMPose` は、RTMDet（人物検出器）と RTMPose（姿勢推定器）を分離制御し、標準の 17 点検出から WholeBody系の高精度な全身・手・顔・足の 133 点検出まで柔軟に切り替えられる高度な姿勢推定ラップクラスです。

### 主な特徴

1. **多彩なモデル・モード選択（`model` 引数）**:
* `model='coco'`: 身体の主要 17 関節点を高速に推論します。
* `model='whole'`, `'dwpose'`, `'rtmw'`: 身体、足、顔、手を含む最大 133 箇所の WholeBody キーポイントを一括抽出します。


2. **Detector と Pose の分離設計**:
* `get_RTMDet(frame)` で人物の BBox を検出し、`get_RTMPose(frame, bbox)` でその BBox 内の姿勢のみを推定できます。


3. **パーツ別の専用ゲッターと接続データ自動生成**:
* 推論後、`get_RTMFace()`, `get_RTMHands()`, `get_RTMFeet()` により顔・手・足をそれぞれ個別に取り出せます。また、各パーツの接続ペアリスト取得メソッドを備えています。


4. **ヒートマップ（確率マップ）出力機能**:
* 初期化時の設定条件を満たすことで、関節の存在確率を示すサーモグラフィ画像を取得可能です。



---

## :red_square: my_cap_av2 と連携した基本サンプルコード

`my_cap_av2.py` の `VideoCapture` で正確な経過時間を取得しながら、WholeBody モード（推奨モデル）で全身・手・足の関節点を描画するプログラムです。

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
    mm_pose = MyMMPose(device='cpu', model='whole') # GPUの場合は device='cuda:0'

    # 骨格接続情報の取得
    pose_conn = mm_pose.get_pose_connections()
    hand_conn = mm_pose.get_hand_connections()

    print("MMPose Wholebody Tracking Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 経過時間（ミリ秒）の取得
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        # 2. 人物バウンディングボックスの検出
        bbox, score = mm_pose.get_RTMDet(frame)

        if bbox is not None and score > 0.5:
            # 3. 姿勢推定を実行
            res = mm_pose.get_RTMPose(frame, bbox)

            if res is not None and res['pose'] is not None:
                # 身体（Pose）の描画
                kpts, k_scores = res['pose']
                for p1, p2 in pose_conn:
                    if k_scores[p1] > 0.3 and k_scores[p2] > 0.3:
                        pt1 = tuple(kpts[p1].astype(int))
                        pt2 = tuple(kpts[p2].astype(int))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                # 手（Hands）の描画
                hands_kpts, hands_scores = res['hands']
                if hands_kpts is not None:
                    for side in ['left', 'right']:
                        h_kpts = hands_kpts[side]
                        h_scores = hands_scores[side]
                        for p1, p2 in hand_conn[side]:
                            if h_scores[p1] > 0.3 and h_scores[p2] > 0.3:
                                pt1 = tuple(h_kpts[p1].astype(int))
                                pt2 = tuple(h_kpts[p2].astype(int))
                                cv2.line(frame, pt1, pt2, (255, 128, 0), 1)

        # タイムスタンプの画面表示
        cv2.putText(frame, f"Time: {current_msec:.1f} ms", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

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
| `__init__(device, model)` | `device='cpu'`, `model='coco'` (`'whole'`, `'dwpose'`, `'rtmw'`) | なし | 検出器および指定モードに応じた姿勢推定モデルを初期化 |
| `get_RTMDet(frame)` | `frame`: BGR画像 | `(bbox, score)` | 最高スコアの人物 BBox `[x1, y1, x2, y2]` と信頼度スコアを返却 |
| `get_RTMPose(frame, bbox)` | `frame`: BGR画像, `bbox`: 人物BBox | `dict` | 姿勢データを保持する辞書 (`'pose'`, `'face'`, `'hands'`, `'feet'`, `'heatmap'`) を返却 |

### パーツ別データ取得ゲッター

| メソッド | 戻り値 | 説明 |
| --- | --- | --- |
| `get_RTMFace()` | `(face_kpts, face_scores)` | `whole`系モード時の顔（68点）座標とスコアを取得 |
| `get_RTMHands()` | `(hand_kpts, hand_scores)` | `whole`系モード時の左右手（各21点）座標とスコアの辞書を取得 |
| `get_RTMFeet()` | `(foot_kpts, foot_scores)` | `whole`系モード時の左右足パーツ座標とスコアの辞書を取得 |

### 骨格接続取得・補助ユーティリティ

| メソッド | 戻り値 | 説明 |
| --- | --- | --- |
| `get_pose_connections()` | `list` | 標準身体 17 関節の接続ペアリストを取得 |
| `get_foot_connections()` | `dict` | 足パーツ（左右）の三角形接続ペアリストを取得 |
| `get_hand_connections()` | `dict` | 手（左右各21点）の接続ペアリストを取得 |
| `get_face_connections()` | `list` | 顔（68点）の輪郭・パーツ接続ペアリストを取得 |
| `make_face_bbox(kpts, scores, expand)` | `list` | 身体キーポイントから顔部分の拡大 BBox を算出 |
| `get_template_score(frame, bbox, template_img)` | `float` | 指定領域とテンプレート画像との類似度を算出 |

---

## :red_square: ノート：モデル・環境に関する補足事項

### 1. RTMW と DWPose の特徴と違い

* **RTMW (`rtmw`)**:
* MMPose公式が提供する軽量かつ高精度な全身モデルです。動作が軽快で、手や足先、顔のパーツまでバランスよく追従します。リアルタイム性を重視する場合に最適です。


* **DWPose (`dwpose`)**:
* ControlNet等で広く使われる高精度ポーズ推定パイプラインの実装です。極めて高いロバスト性（隠れに強い）を持ち、手の細かなジェスチャーや顔の向きの変化を正確に捉えたい場合に適しています。



### 2. CPU 推論と GPU 推論の違い

* **CPU 推論 (`device='cpu'`)**:
* 追加の設定が不要ですぐ実行できますが、`model='whole'` などの 133 点モデルではフレームレート（FPS）が大きく低下します。


* **GPU 推論 (`device='cuda:0'`)**:
* NVIDIA製GPUの並列演算により推論速度が劇的に向上し、フル推定でも滑らかなリアルタイム処理が可能になります。
