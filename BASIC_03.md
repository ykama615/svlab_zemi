<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. 顔・手・ポーズ検出（↓）

<hr>

# MediaPipe（旧バージョン用）
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

      if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
        break

      cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

  if __name__ == '__main__':
    main()
  ```

  ## どの手が挙がっているかを評価するサンプル
  - MediaPipeのPoseを使って左右どちらの手を挙げているか（または両方）を画面上に表示します
  - innerカメラは左右が反転している（鏡状になっている）ので，cv2.flip関数を使って反転しています
  - MediaPipeはRGBカラー，VideoCapture（OpenCV）はBGRカラーなのでcv2.cvtColor関数で順序の入れ替えを行っています
   - cv2.imshowの前にもう一度cv2.cvtColor関数を使ってBGRカラーに戻しています
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
      if cv2.waitKey(1) & 0xFF == 27:
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
  
  ## 人差し指の座標を表示するサンプル
  - MediaPipeのHandsを使って左右の人差し指の位置座標を表示します
  - innerカメラは左右が反転している（鏡状になっている）ので，cv2.flip関数を使って反転しています
  - MediaPipeはRGBカラー，VideoCapture（OpenCV）はBGRカラーなのでcv2.cvtColor関数で順序の入れ替えを行っています
    - cv2.imshowの前にもう一度cv2.cvtColor関数を使ってBGRカラーに戻しています
  - 人差し指の先はLandmarkリスト（配列）の添字8に割り当てられています
    - 他のLandmarkは，[マニュアル](https://google.github.io/mediapipe/solutions/hands.html) で確認してください 
  ```python
 import cv2
 import mediapipe as mp
 import numpy as np
 import time
 mp_drawing = mp.solutions.drawing_utils
 mp_hands = mp.solutions.hands

 device = 0 # cameera device number

 def getFrameNumber(start:float, fps:int):
   now = time.perf_counter() - start
   frame_now = int(now * 1000 / fps)

   return frame_now

 # Draw a circle on index finger
 def drawFingertip(image, landmarks):
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

   # Draw a circle on index finger and display the coordinate value
   cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), 3)
   cv2.putText(image, "(" + str(landmark_point[8][0]) + ", " + str(landmark_point[8][1]) + ")", 
     (landmark_point[8][0] - 20, landmark_point[8][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)


 def main():
   # For webcam input:
   global device

   cap = cv2.VideoCapture(device)
   fps = cap.get(cv2.CAP_PROP_FPS)
   wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
   ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

   print("Size:", ht, "x", wt, "/Fps: ", fps)

   start = time.perf_counter()
   frame_prv = -1

   cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)

   hands = mp_hands.Hands(
   min_detection_confidence=0.5,
   min_tracking_confidence=0.5)

   while cap.isOpened():
     frame_now=getFrameNumber(start, fps)
     if frame_now == frame_prv:
         continue
     frame_prv = frame_now

     ret, frame = cap.read()
     if not ret:
       print("Ignoring empty camera frame.")
       # If loading a video, use 'break' instead of 'continue'.
       continue

     # Flip the image horizontally for a later selfie-view display, and convert
     # the BGR image to RGB.
     frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

     # To improve performance, optionally mark the image as not writeable to
     # pass by reference.
     frame.flags.writeable = False
     results = hands.process(frame)

     # Draw the index finger annotation on the image.
     frame.flags.writeable = True
     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
     if results.multi_hand_landmarks:
         for hand_landmarks in results.multi_hand_landmarks:
             drawFingertip(frame, hand_landmarks)
     cv2.imshow('MediaPipe Hands', frame)
     if cv2.waitKey(1) & 0xFF == 27:
           break
   cap.release()

 if __name__ == '__main__':
   main()
 ```
 
 ## FaceMeshを表示するサンプル
 - MediaPipeのFaceMeshを使って468点の顔の3D特徴点を表示します
 - innerカメラは左右が反転している（鏡状になっている）ので，cv2.flip関数を使って反転しています
 - MediaPipeはRGBカラー，VideoCapture（OpenCV）はBGRカラーなのでcv2.cvtColor関数で順序の入れ替えを行っています
   - cv2.imshowの前にもう一度cv2.cvtColor関数を使ってBGRカラーに戻しています
 - Landmarkの詳しい情報は，[マニュアル](https://google.github.io/mediapipe/solutions/face_mesh.html) で確認してください 

 ```python
 import cv2
 import mediapipe as mp
 import time
 import numpy as np
 mp_drawing = mp.solutions.drawing_utils
 mp_face_mesh = mp.solutions.face_mesh

 device = 0 # cameera device number

 def getFrameNumber(start:float, fps:int):
   now = time.perf_counter() - start
   frame_now = int(now * 1000 / fps)

   return frame_now

 def main():
   # For webcam input:
   global device

   cap = cv2.VideoCapture(device)
   fps = cap.get(cv2.CAP_PROP_FPS)
   wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
   ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

   print("Size:", ht, "x", wt, "/Fps: ", fps)

   start = time.perf_counter()
   frame_prv = -1

   drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

   face_mesh = mp_face_mesh.FaceMesh(
   min_detection_confidence=0.5,
   min_tracking_confidence=0.5)

   while cap.isOpened():
     frame_now=getFrameNumber(start, fps)
     if frame_now == frame_prv:
       continue
     frame_prv = frame_now

     ret, frame = cap.read()
     if not ret:
       print("Ignoring empty camera frame.")
       # If loading a video, use 'break' instead of 'continue'.
       continue

     # Flip the image horizontally for a later selfie-view display, and convert
     # the BGR image to RGB.
     frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
     # To improve performance, optionally mark the image as not writeable to
     # pass by reference.
     frame.flags.writeable = False
     results = face_mesh.process(frame)

     # Draw the face mesh annotations on the image.
     frame.flags.writeable = True
     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
     if results.multi_face_landmarks:
       for face_landmarks in results.multi_face_landmarks:
         #mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACE_CONNECTIONS)
         my_draw_face(frame, face_landmarks)
     cv2.imshow('MediaPipe FaceMesh', frame)
     if cv2.waitKey(1) & 0xFF == 27:
       break
   cap.release()

 def my_draw_face(image, landmarks):
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
     for i in range(0, len(landmark_point)):
       cv2.circle(image, (int(landmark_point[i][0]),int(landmark_point[i][1])), 1, (0, 255, 0), 1)

 if __name__ == '__main__':
   main()
 ```
<!--
## [エクストラ] 配布環境の自作ライブラリの利用

配布環境には，mediapipeのうち，フレーム画像を渡すと，Face Detection，Hands，Poseの結果の座標リスト，Selfie Segmentationの結果画像を返却する補助するライブラリ（パッケージ）が用意してあります．
 - 準備
   - パッケージをimportし，インスタンスを生成しておきます 
   - フレーム画像は反転，色変換（RGBにする）をしておきます
    - imshowする際に，色を再変換（BGRに戻す）する必要があります 
   ```python
   import myPhysiology as mp
   mpd = mp.MpDetector()
   frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
   ```
   
 - getFace(frame, bool=ランドマーク座標返却の要・不要) -> 検出した顔の領域の座標とランドマーク座標
   ```python
   detections = mpd.getFace(frame, True)
   for [box, keys] in detections:
     if box!=[]:
       cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), [0,255,0], 1)
     if keys!=[]:
       for points in keys:
         cv2.circle(frame, points, 3, [255,0,0], -1)
   ```

 - getHand(frame) -> 左右の指の座標リストのリスト
   ```python
   hands = mpd.getHand(frame)
   for hand in hands:
     left = hand[0]
     right = hand[1]
     for point in left:
       x, y, z = point
       cv2.circle(frame, [x, y], 3, [255,0,255], -1)
     for point in right:
       x, y, z = point
       cv2.circle(frame, [x, y], 3, [0,255,0], -1)
   ```
 
 - getPose(frame) -> ポーズの座標リスト
   ```python
   poses = mpd.getPose(frame)
   for point in poses:
     x, y, z, ret = point
     if ret:
       scale = abs(int(5 * (z/cap.wt-1.0)))
       cv2.circle(frame, [x, y], scale, [0,0,255], -1)
   ```

  ## チョキを判定するサンプル
  - カメラの呼び出しには [配布環境のパッケージの関数](BASIC_0.md) を利用しています（VideoCaptureに置き換え可能です）
  - MediaPipe Handには 配布環境のパッケージの関数（前節）を使っています（標準関数に置き換え可能です）
  - ベクトルの長さ，角度の算出方法（calcAngle関数）については [こちら](BASIC_FP01.md#2つのベクトルのなす角) を確認してください

  ```python
  # -*- coding: utf-8 -*-
  import cv2
  import numpy as np
  import myCapture as mycap
  import myPhysiology as mp
  import time

  dev = 0

  def main():
      cap = mycap.CameraSelector(dnum=dev, size=[640, 480])
      mpd = mp.MpDetector()

      while cap.isOpened():
          ret, fnum, frame = cap.read()
          start = time.perf_counter()
          frame = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),1)

          if ret:
            hands = mpd.getHand(frame)
            for hand in hands:
              right = hand[1]

              if len(right)>0: 
                for point in right:
                  x, y, _ = point
                  cv2.circle(frame, [x, y], 3, [0,255,0], -1)

                # 手の付け根と指の先端，第2関節までの長さを算出（指を閉じているかどうかの判定）
                third_m = np.linalg.norm(np.array(right[0])-np.array(right[14])) # 手の付根-薬指第２関節
                pinky_m = np.linalg.norm(np.array(right[0])-np.array(right[18])) # 手の付根-子指第２関節
                third_t = np.linalg.norm(np.array(right[0])-np.array(right[16])) # 手の付根-薬指先端
                pinky_t = np.linalg.norm(np.array(right[0])-np.array(right[20])) # 手の付根-薬指先端

                # 手のひらの幅，手のひらの外側から親指の先端までの長さを算出（親指を折り曲げているかどうかの判定）
                thumb_l = np.linalg.norm(np.array(right[17])-np.array(right[4]))
                hand_wt = np.linalg.norm(np.array(right[17])-np.array(right[5]))

                # 第2関節-第3関節と第2関節-先端のなす角を算出（指を伸ばしているかどうかの判定）
                first_d = calcAngle(np.array(right[7])-np.array(right[6]),np.array(right[5])-np.array(right[6]))
                secnd_d = calcAngle(np.array(right[11])-np.array(right[10]),np.array(right[9])-np.array(right[10]))

                if (third_t < third_m) and (pinky_t < pinky_m) and (thumb_l < hand_wt) and (first_d > 140) and (secnd_d > 140):
                  print("choki")

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("video", frame)
            if cv2.waitKey(1) == ord('q'):
              break

      cap.release()
      cv2.destroyAllWindows()

  def calcAngle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)

    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)

    return np.rad2deg(np.arccos(cos_theta))

  if __name__=='__main__':
      main()
      
  ```

  ## [課題] じゃんけん判定
  1. ぐー，ちょき，ぱーを判定できるように改良してみましょう
  2. 適当なタイミングでコンピュータの手を決定し，カメラに映った人の手と比較して勝敗を判定しじゃんけんができるようにしてみましょう

  - 0以上3未満（0,1,2）の整数乱数を発生させるサンプルは以下のようになります
    
```python
import random

cpu = random.randint(0,3)
```
-->
