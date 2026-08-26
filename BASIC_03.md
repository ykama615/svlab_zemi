<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. 顔・手・ポーズ検出（↓）
6. [各種クラス・応用](BASIC_04.md)

<hr>

# MediaPipe
MediaPipe の新 API（**Tasks API** / `mediapipe.tasks`）では、タスクごとに学習済みモデル（`.task` または `.tflite` ファイル）を読み込み、専用の検出器（Detector / Landmarker / Recognizer / Segmenter）を構成して処理を行います。

---

## モデルファイルの配置場所

本環境における MediaPipe の学習済みモデルは、すべて以下の相対パス配下に配置して読み込みます。

**プロジェクト内のモデル配置構造**

```text
project_root/
│
├── learned_models/
│   └── mediapipe/                          ← モデルファイルの配置フォルダ
│       ├── hand_landmarker.task            (手の検出用)
│       ├── pose_landmarker_lite.task       (姿勢推定用)
│       ├── face_landmarker.task            (顔メッシュ・BlendShapes用)
│       ├── blaze_face_short_range.tflite   (顔検出用)
│       ├── gesture_recognizer.task         (ジェスチャ認識用)
│       └── selfie_segmentation.tflite      (背景分離用)
│
└── main.py                                 ← 実行用スクリプト

```

---

## MediaPipe Tasks API の機能一覧

`mediapipe.tasks.vision` モジュールで提供されている主要な検出器と、対応するモデルファイルは以下の通りです。

| 機能 | タスク名 (`mediapipe.tasks.vision`) | 使用モデルファイルパス | 概要 |
| --- | --- | --- | --- |
| **Hands** | `HandLandmarker` | `./learned_models/mediapipe/hand_landmarker.task`[cite: 5] | 手の21箇所ランドマークと左右判定[cite: 5] |
| **Pose** | `PoseLandmarker` | `./learned_models/mediapipe/pose_landmarker_lite.task`[cite: 5] | 身体33箇所のランドマークを取得[cite: 5] |
| **Face Mesh** | `FaceLandmarker` | `./learned_models/mediapipe/face_landmarker.task`[cite: 5] | 顔のメッシュおよび表情スコア（BlendShapes）取得[cite: 5] |
| **Face** | `FaceDetector` | `./learned_models/mediapipe/blaze_face_short_range.tflite`[cite: 5] | 顔のバウンディングボックスとキーポイント検出[cite: 5] |
| **Gesture** | `GestureRecognizer` | `./learned_models/mediapipe/gesture_recognizer.task`[cite: 5] | グー・チョキ・パー等のハンドジェスチャ認識[cite: 5] |
| **Segment** | `ImageSegmenter` | `./learned_models/mediapipe/selfie_segmentation.tflite`[cite: 5] | 人物領域を抽出する背景分離マスク処理[cite: 5] |

---

## 標準 Tasks API による処理の基本の流れ

いずれのタスクも、標準的な実装手順は以下の 5 ステップで共通しています。

1. **`BaseOptions` の設定**: `./learned_models/mediapipe/` 配下のモデルパスを指定[cite: 5]
2. **`Options` の構築**: 動作モード（`RunningMode.IMAGE` 等）や各種パラメータを設定[cite: 5]
3. **検出器の生成**: 各クラスの `create_from_options()` を呼び出してインスタンス化[cite: 5]
4. **`mp.Image` への変換**: OpenCV の BGR 画像を `mp.Image` オブジェクトへ変換[cite: 5]
5. **推論と結果取得**: `detect()` や `recognize()`, `segment()` を呼び出して処理を実行[cite: 5]

---

## Selfie Segmentation（背景置き換え）

`ImageSegmenter` と `selfie_segmentation.tflite` を使用し、人物領域のマスクを取得して背景をマゼンタ色に置換します[cite: 5]。

```python
import cv2
import numpy as np
import mediapipe as mp

# 1. オプション設定とモデル指定
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

segment_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path="./learned_models/mediapipe/selfie_segmentation.tflite"),
    running_mode=VisionRunningMode.IMAGE,
    output_category_mask=True
)

def main():
    cap = cv2.VideoCapture(0)

    # 2. Segmenter インスタンスの生成
    with ImageSegmenter.create_from_options(segment_options) as segmenter:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 3. mp.Image への変換（BGR -> RGB）
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 4. 推論実行とマスク取得
            segmentation_result = segmenter.segment(mp_image)
            
            if segmentation_result.category_mask is not None:
                mask = segmentation_result.category_mask.numpy_view()
                # しきい値適用（人物領域の判別）
                condition = np.stack((mask.squeeze() <= 0.5,) * 3, axis=2)

                # 背景画像（マゼンタ色）の作成
                bg_image = np.zeros(frame.shape, dtype=np.uint8)
                bg_image[:] = (255, 0, 255)

                # 合成処理
                frame = np.where(condition, frame, bg_image)

            cv2.imshow('Selfie Segmentation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

---

## 挙手判定サンプル（Pose Landmarker）

`PoseLandmarker` と `pose_landmarker_lite.task` を使用し、身体の 33 箇所キーポイントから挙手を判定します[cite: 5]。

```python
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./learned_models/mediapipe/pose_landmarker_lite.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5
)

def judge_raise_hand(pose_landmarks):
    # NormalizedLandmark: 0=Nose, 15=Left Wrist, 16=Right Wrist
    nose_y = pose_landmarks[0].y
    left_wrist_y = pose_landmarks[15].y
    right_wrist_y = pose_landmarks[16].y

    is_left_up = left_wrist_y < nose_y
    is_right_up = right_wrist_y < nose_y

    if is_left_up and is_right_up:
        return "both"
    elif is_left_up:
        return "left"
    elif is_right_up:
        return "right"
    return ""

def main():
    cap = cv2.VideoCapture(0)

    with PoseLandmarker.create_from_options(pose_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ht, wt, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            result = landmarker.detect(mp_image)

            if result.pose_landmarks:
                pose_landmarks = result.pose_landmarks[0]

                # 座標描画
                for lm in pose_landmarks:
                    cx, cy = int(lm.x * wt), int(lm.y * ht)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

                res_text = judge_raise_hand(pose_landmarks)
                cv2.putText(frame, res_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            cv2.imshow('MediaPipe Pose', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

---

## 人差し指の座標表示（Hand Landmarker）

`HandLandmarker` と `hand_landmarker.task` を使用し、手の 21 箇所ランドマークおよび左右判定（Handedness）を取得します[cite: 5]。

```python
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hands_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./learned_models/mediapipe/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

def main():
    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(hands_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ht, wt, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # 左右判別
                    handedness = results.handedness[idx][0].category_name
                    
                    # Index 8: 人差し指先端
                    pt8 = hand_landmarks[8]
                    cx, cy = int(pt8.x * wt), int(pt8.y * ht)

                    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(frame, f"{handedness} ({cx}, {cy})", (cx - 20, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('MediaPipe Hands', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

---

## 顔メッシュと BlendShapes（Face Landmarker）

`FaceLandmarker` と `face_landmarker.task` を使用します[cite: 5]。`output_face_blendshapes=True` を指定することで、表情のスコア（笑顔や目の開き具合等）を取得可能です[cite: 5]。

```python
import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

fmesh_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./learned_models/mediapipe/face_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=True  # 表情スコアの出力を有効化
)

def main():
    cap = cv2.VideoCapture(0)

    with FaceLandmarker.create_from_options(fmesh_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ht, wt, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = landmarker.detect(mp_image)

            # メッシュ（座標）の描画
            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    for lm in face_landmarks:
                        cx, cy = int(lm.x * wt), int(lm.y * ht)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

            # BlendShapes（表情データ）の取得
            if detection_result.face_blendshapes:
                first_face_shapes = {cat.category_name: cat.score for cat in detection_result.face_blendshapes[0]}
                smile_score = first_face_shapes.get("mouthSmileLeft", 0.0)
                cv2.putText(frame, f"Smile Score: {smile_score:.2f}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            cv2.imshow('MediaPipe Face Mesh', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

---

## チョキの判定サンプル（Hand Landmarker）

`HandLandmarker` から取得した 3D 座標を元にベクトル内積計算を行い、指の開き・曲がり具合を計算して「チョキ」を判定します[cite: 5]。

```python
import cv2
import math
import numpy as np
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hands_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./learned_models/mediapipe/hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)

def calc_angle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)
    if v1_n == 0 or v2_n == 0:
        return 0.0
    cos_theta = np.clip(np.inner(v1, v2) / (v1_n * v2_n), -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))

def main():
    cap = cv2.VideoCapture(0)

    with HandLandmarker.create_from_options(hands_options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            ht, wt, _ = frame.shape
            zt = math.sqrt(wt * wt + ht * ht)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    # 座標の格納
                    pts = [np.array([lm.x * wt, lm.y * ht, lm.z * zt]) for lm in hand_landmarks]

                    for p in pts:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

                    # 距離と角度の算出
                    third_m = np.linalg.norm(pts[0] - pts[14])
                    pinky_m = np.linalg.norm(pts[0] - pts[18])
                    third_t = np.linalg.norm(pts[0] - pts[16])
                    pinky_t = np.linalg.norm(pts[0] - pts[20])

                    thumb_l = np.linalg.norm(pts[17] - pts[4])
                    hand_wt = np.linalg.norm(pts[17] - pts[5])

                    first_d = calc_angle(pts[7] - pts[6], pts[5] - pts[6])
                    secnd_d = calc_angle(pts[11] - pts[10], pts[9] - pts[10])

                    # チョキ判定
                    if (third_t < third_m) and (pinky_t < pinky_m) and (thumb_l < hand_wt) and (first_d > 140) and (secnd_d > 140):
                        cv2.putText(frame, "choki", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("Janken Check", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

---

## [課題] じゃんけん判定

1. 上記のチョキ判定コードを拡張して、「グー」「チョキ」「パー」のすべての手を判定できるようにプログラムを改良してみましょう。
2. ランダムにコンピュータの手（`0`:グー, `1`:チョキ, `2`:パー）を決定し、カメラに映ったプレイヤーの手と勝敗判定を行う「じゃんけんゲーム」を作成してみましょう。
