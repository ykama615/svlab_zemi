<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. 顔・手・ポーズ検出（↓）

<hr>

# MediaPipe
 MediaPipeで提供されている機能のうちPythonでサポートされているものは，2022.08現在，以下の通りである．
  - Face Detection
  - Face Mesh
  - Hands
  - Pose
  - Holistic
  - Selfie Segmentation
  - Objectron

  ## Selfie Segmentationを例とした基本的な使い方
  以下は，MediaPipe の Selfie Segmentation 機能を利用した例です．他の機能を利用する場合も基本手順は同じとなります．
  1. mp.solutionsパッケージから使いたい機能を呼び出す
  2. processメソッドを用いて検出を行う
  3. processの戻り値（results）を分解して必要な情報を抽出する

  ```python
  #-*- coding: utf-8 -*-
  import cv2
  import numpy as np
  import mediapipe as mp

  device = 0

  def main():
    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # SelfieSegmentationを利用する準備
    segment = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
                           #↑この部分を変更すると検出されるものが変わる

    while cap.isOpened() :
      ret, frame = cap.read()

      # frameへのmediapipe(SelfieSegmentation)の適用
      results = segment.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

      # 検出結果(results)が存在した場合
      if results.segmentation_mask is not None:
        condition = np.stack((results.segmentation_mask, )*3, axis=-1)>0.5 #↑前景と背景の境界（0.5※適当）でマスクを作成

        bg = np.ones(frame.shape, dtype=np.uint8)*255 #magentaの背景を
        bg[:,:,1] = 0                                 #作っています

        #conditionの切り分けに従って，frameのままとbgへの置き換えを行う
        frame = np.where(condition, frame, bg)

      if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q') or ret == False:
        break

      cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

  if __name__ == '__main__':
    main()
  ```

  ## どの手が挙がっているかを評価するサンプル
  - MediaPipeのPoseを使って左右どちらの手を挙げているか（または両方）を画面上に
  - innerカメラは左右が反転している（鏡状になっている）ので，cv2.flip関数を使って反転しています
  - MediaPipeはRGBカラー，VideoCapture（OpenCV）はBGRカラーなのでcv2.cvtColor関数で順序の入れ替えを行っています
   -- cv2.imshowの前にもう一度cv2.cvtColor関数を使ってBGRカラーに戻しています
  ```python
  import cv2
  import mediapipe as mp
  import numpy as np
  mp_drawing = mp.solutions.drawing_utils
  mp_pose = mp.solutions.pose

  device = 0 # cameera device number

  def main():
    # For webcam input:
    cap = cv2.VideoCapture(device)
    pose = mp_pose.Pose( min_detection_confidence=0.5, min_tracking_confidence=0.5 )

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
      image = cv2.cvtColor( cv2.flip(frame, 1), cv2.COLOR_BGR2RGB )

      # To improve performance, optionally mark the image as not writeable to pass by reference.
      image.flags.writeable = False
      results = pose.process( image )

      # Draw the pose annotation on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # judge
        cv2.putText(image, judge_raise_hand(image, results.pose_landmarks), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

      cv2.imshow('MediaPipe Pose', image)
      if cv2.waitKey(5) & 0xFF == 27:
        break
    cap.release()

  # Judgment of raising hand
  def judge_raise_hand(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
      if landmark.visibility < 0 or landmark.presence < 0:
        continue

      # Convert the obtained landmark values x, y, z to the coordinates on the image
      landmark_x = min(int(landmark.x * image_width), image_width - 1)
      landmark_y = min(int(landmark.y * image_height), image_height - 1)
      landmark_z = landmark.z

      landmark_point.append(np.array([landmark_x, landmark_y, landmark_z], dtype=int))

    if len(landmark_point) != 0:
      if landmark_point[0][1] > landmark_point[20][1] and landmark_point[0][1] > landmark_point[19][1]:
        return "both"
      elif landmark_point[0][1] > landmark_point[20][1]:
        return "left"
      elif landmark_point[0][1] > landmark_point[19][1]:
        return "right"
      else:
        return ""

  if __name__ == '__main__':
    main()
  ```
