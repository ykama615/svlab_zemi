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
      results = segment.process(frame)

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
