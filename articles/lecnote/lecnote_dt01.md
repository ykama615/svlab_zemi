# MediaPipe統合処理ライブラリ (my_mediapipe_n.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、`my_cap_av2.py` の `VideoCapture` クラスを用いて映像を取り込み、`my_mediapipe_n.py`（`MyMediaPipeN` クラス）を使用して顔検出・顔メッシュ・手・姿勢（Pose）・ジェスチャ・セグメンテーションなどの認知機能を統合的に処理する方法について解説します。

## 前提条件
- デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
- **【重要】** `my_cap_av2.py` および `my_mediapipe_n.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください。
- **【重要】** 学習済みモデルファイル群（`.task`, `.tflite`）が `C:\oit\home\ipbl\learned_models\mediapipe\` 内に正しく配置されている必要があります。
- 作成するPythonプログラム（`.py`）は `C:\oit\home\ipbl` に保存します。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

```

---

## :red_square: my_mediapipe_n.py の概要と特徴

`MyMediaPipeN` は、MediaPipe Tasks API (Python) をラップし、OpenCV形式の画像フレームに対して多様な認識（顔・手・姿勢・ジェスチャ・顔メッシュ・表情スコア・背景セグメンテーション等）を簡潔なメソッド呼び出しで実現するカスタムクラスです。

### 主な特徴

1. **統合された認識タスク**:
* 顔検出（Face）、顔メッシュ（Face Mesh）、手検出（Hands）、姿勢推定（Pose）、ジェスチャ認識（Gesture）、セルフィーセグメンテーション（Selfie Segmentation）を一括管理します。


2. **OpenCV との親和性**:
* OpenCVで取得した BGR 画像（`ndarray`）を `get_mp_image()` で MediaPipe 専用の `mp.Image` オブジェクトへ手軽に変換可能です。


3. **表情解析（BlendShapes）とアライメント制御**:
* 顔メッシュ検出時に内部で BlendShapes（52種類の表情パラメータ）や 4x4 変換行列のキャッシュを自動更新し、ゲッター経由で取得可能です。


4. **描画支援ユーティリティ**:
* 目蓋の開き具合の計測（`get_vertical_eyelid`）や、検出部位間の接続情報（`get_connections`）の参照機能を備えています。



---

## :red_square: my_cap_av2 と連携した基本サンプルコード

`my_cap_av2.py` の `VideoCapture` で映像を入力し、`MyMediaPipeN` で手のランドマークを検出・描画する基本プログラムです。

### mp_hand_viewer.py

```python
import os
# OpenCVのキャプチャ遅延を防ぐ設定
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
# ライブラリのインポート
from my_libs.my_cap_av2 import VideoCapture
from my_libs.my_mediapipe_n import MyMediaPipeN

device = 0 # 0: Webカメラ, または動画ファイルパス指定

def main():
    global device

    # 1. キャプチャと MediaPipe の初期化
    cap = VideoCapture(device)
    mp_nn = MyMediaPipeN(detect_num=1) # 1人分を検出設定

    print("MediaPipe Hand Tracking Started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 2. BGR画像を MediaPipe 用 Image オブジェクトへ変換 (f_flip=1で左右反転可能)
        mp_image = mp_nn.get_mp_image(frame, f_flip=0)

        # 3. 手の検出を実行
        hands_dict = mp_nn.get_hand(mp_image)

        # 4. 検出された手の関節点を描画 (左手・右手)
        for side in ["left", "right"]:
            for hand_points in hands_dict[side]:
                for pt in hand_points:
                    x, y, z, vis, pres = pt
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # 5. 画面表示
        cv2.imshow("MediaPipe Hands (my_cap_av2)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---

## :red_square: MyMediaPipeN の主なメソッド一覧

### 画像変換・基本操作

| メソッド | 説明 |
| --- | --- |
| `get_mp_image(frame, f_flip=0)` | OpenCVの BGR 画像を MediaPipe 用の `mp.Image` に変換。`f_flip=1` で左右反転 |
| `set_detect_number(detect_num)` | 検出対象人数/手の数の上限設定を更新 |

### 各種タスク実行メソッド

| メソッド | 引数 | 戻り値の概要 |
| --- | --- | --- |
| `get_face(mp_image, getkeys=True)` | `mp_image` | 顔のバウンディングボックス `[x, y, w, h]` と主要キーポイントのリスト |
| `get_face_mesh(mp_image)` | `mp_image` | 顔メッシュ（468/478点）の 3D 座標リスト `[x, y, z]` |
| `get_dlib_landmark(mp_image)` | `mp_image` | dlib互換（68点）の顔ランドマーク座標および左右目のポイント群 |
| `get_iris(mp_image)` | `mp_image` | 左右の虹彩（瞳孔周辺）座標 `{'leye': [...], 'reye': [...]}` |
| `get_hand(mp_image)` | `mp_image` | 左右別の手関節座標群 `{'left': [...], 'right': [...]}` |
| `get_pose(mp_image)` | `mp_image` | 身体の姿勢ランドマーク（33点）座標群 `[x, y, z, visibility]` |
| `get_gesture_data(mp_image, gesture_name, side)` | `mp_image`, `"Open_Palm"`, `"Left"` など | 指定ジェスチャの信頼度スコア (`float`) と手関節座標 |
| `get_segment_image(mp_image, dep=0.5)` | `mp_image`, 閾値 `dep` | 人物領域の背景切り抜き用ブールマスク (`bool ndarray`) |

### 表情・変換行列の取得（ゲッター）

| メソッド | 説明 |
| --- | --- |
| `get_blendshapes()` | 直近の `get_face_mesh` または `get_dlib_landmark` 実行時に更新された表情スコア（BlendShapes辞書リスト）を取得 |
| `get_transformation_matrices()` | 顔の 4x4 姿勢変換行列リストを取得 |

---

## :red_square: 応用サンプル: 表情（BlendShapes）と姿勢推定の表示

`get_pose()` と `get_blendshapes()` を組み合わせた応用例です。

### mp_pose_and_face.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from my_libs.my_cap_av2 import VideoCapture
from my_libs.my_mediapipe_n import MyMediaPipeN

def main():
    cap = VideoCapture(0)
    mp_nn = MyMediaPipeN(detect_num=1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        mp_image = mp_nn.get_mp_image(frame)

        # 1. 姿勢推定の描画
        pose_list = mp_nn.get_pose(mp_image)
        for pose in pose_list:
            for pt in pose:
                x, y, z, vis = pt
                if vis > 0.5: # 信頼度が50%以上の点のみ描画
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

        # 2. 顔メッシュ実行 (内部で表情BlendShapesスコアが更新される)
        _ = mp_nn.get_face_mesh(mp_image)
        blendshapes = mp_nn.get_blendshapes()

        # 3. 笑顔 (jawOpen や smile) などのスコア表示
        if blendshapes:
            face0 = blendshapes[0]
            smile_score = face0.get("mouthSmileLeft", 0.0)
            cv2.putText(frame, f"Smile Score: {smile_score:.2f}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Pose & Expression", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---

## :red_square: 演習 (`mp_selfie_segmentation.py`)

* [`mp_hand_viewer.py`](https://www.google.com/search?q=%23mp_hand_viewerpy) を参考にして、`get_segment_image()` を使用して人物の背景を特定の色（例: 青色）に置き換える `mp_selfie_segmentation.py` を作成してください。
* **ヒントコード**:
```python
# セグメンテーションマスクの取得 (True: 人物, False: 背景)
condition = mp_nn.get_segment_image(mp_image, dep=0.5)

# 背景用カラー画像を作成 (例: 青色)
bg_image = np.full(frame.shape, (255, 0, 0), dtype=np.uint8)

# 人物領域は元フレーム、背景領域は bg_image を合成
output_frame = np.where(condition, frame, bg_image)
cv2.imshow("Background Replacement", output_frame)

```



---

[トップページへ戻る]()
