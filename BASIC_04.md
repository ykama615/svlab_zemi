<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. 顔・手・ポーズ検出（↓）

<hr>

# MediaPipeのクラス化サンプル
 MediaPipe新バージョン（2023.4版，mediapipe0.10.0で検証済）でも旧バージョンの利用法で利用が可能である．<br>
  ・FaceMeshにIris（中心1点，周辺4点x2の計10点）が追加された（新・旧いずれの使い方でも利用可能）
  ・新バージョンでの利用方法は学習ファイルを外部ファイルとして指定するため，再学習や学習ファイルの変更が可能
  ・処理速度は旧バージョンの利用法の方が上
  ・新バージョンの利用法と旧バージョンの利用法を混在させるとうまく動作しないという報告がある（未確認）
 下記は旧バージョンでの利用方法を検出部位ごとに関数化し，クラス化したもの（mediapipe0.10.0で検証済）．

 ```python
import cv2
import numpy as np
import mediapipe as mp

class myMrdiapipe:
    def __init__(self, detection=0.2, tracking=0.2):
        global hands, pose, fmesh, face, segment

        self.hands = mp.solutions.hands.Hands(min_detection_confidence = detection, 
                                              min_tracking_confidence = tracking)
        self.pose = mp.solutions.pose.Pose(static_image_mode = False, 
                                           model_complexity = 1, 
                                           enable_segmentation = False, 
                                           min_detection_confidence = detection, 
                                           min_tracking_confidence = tracking)
        self.fmesh = mp.solutions.face_mesh.FaceMesh(static_image_mode = False, 
                                                     max_num_faces = 1, 
                                                     refine_landmarks = True, 
                                                     min_detection_confidence = detection)
        self.face = mp.solutions.face_detection.FaceDetection(model_selection = 0, 
                                                              min_detection_confidence = detection)
        self.segment = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection = 0)

    def getMPImage(self, frame):
        mp_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return mp_image

    def getFace(self, image, wt, ht, getkeys=True):
        results = self.face.process(image)
        point_list = []
        face_box = []
        if results.detections is not None:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_box = [int(bbox.xmin*wt), int(bbox.ymin*ht), int(bbox.width*wt), int(bbox.height*ht)]
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1))] for landmark in detection.location_data.relative_keypoints]
                point_list.append([face_box, tmp])

        return point_list

    def getFaceMesh(self, image, wt, ht, getkeys=True):
        results = self.fmesh.process(image)
        point_list = []
        if results.multi_face_landmarks is not None:
            for one_face_landmarks in results.multi_face_landmarks:
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * wt)] for landmark in one_face_landmarks.landmark]
                point_list.append(tmp)

        return point_list
    
    def getIris(self, image, wt, ht, getkeys=True):
        results = self.fmesh.process(image)
        point_list = []
        if results.multi_face_landmarks is not None:
            for one_face_landmarks in results.multi_face_landmarks:
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * wt)] for landmark in one_face_landmarks.landmark[-10:]]
                point_list.append(tmp)

        return point_list

    def getHand(self, image, wt, ht):
        results = self.hands.process(image)
        hpoint_list = []
        lhand_points = []
        rhand_points = []

        if results.multi_hand_landmarks is not None:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                if results.multi_handedness[i].classification[0].label == "Left":
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = max(1, min(int(landmark.x * wt), wt-1))
                        y = max(1, min(int(landmark.y * ht), ht-1))
                        lhand_points.append([int(x), int(y), landmark.z])
                elif results.multi_handedness[i].classification[0].label == "Right":
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = max(1, min(int(landmark.x * wt), wt-1))
                        y = max(1, min(int(landmark.y * ht), ht-1))
                        rhand_points.append([int(x), int(y), landmark.z])
            hpoint_list.append([lhand_points, rhand_points])
        return hpoint_list

    def getPose(self, frame): # Mediapipe can detect only one person.
        ht, wt, _ = frame.shape
        results = self.pose.process(frame)
        pose_points = []
        pose_list = []
        if results.pose_landmarks is not None:
            for i, point in enumerate(results.pose_landmarks.landmark):
                x = max(1, min(int(point.x * wt), wt-1))
                y = max(1, min(int(point.y * ht), ht-1))
                z = int(point.z * wt)
                pose_points.append([x, y, z, point.visibility])
            pose_list.append(pose_points)
        return pose_list

    def getSegmentImage(self, frame, bgimage=[], dep=0.1):
        res = []
        sresults = self.segment.process(frame)

        if sresults.segmentation_mask is not None:
            condition = np.stack(
                (sresults.segmentation_mask, )*3, axis=-1) > dep
            if len(bgimage)==0:
                bg = np.ones(frame.shape, dtype=np.uint8)*255
            else:
                ht, wt, _ = frame.shape
                bg = cv2.resize(bgimage, (wt, ht))

            res = np.where(condition, frame, bg)

        return res
 ```

 このクラスの使い方は次の通り．
 ```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
from myMediapipe import myMrdiapipe

dev = 0

def main():
    global dev

    sub = myMrdiapipe()

    cap = cv2.VideoCapture(dev)
    ht = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    wt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    back = cv2.imread("./img/swan.jpg")
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = sub.getMPImage(frame)
        outframe = frame.copy()
        '''
        ## Hands #############################################################
        hands = sub.getHand(image, wt, ht)
        if len(hands)==1:
            left = hands[0][0]
            right = hands[0][1]

            if len(left)>0:
                for point in left:
                    cv2.circle(outframe, (point[0], point[1]), 5, [255,0,0], -1)
                    
            if len(right)>0:
                for point in right:
                    cv2.circle(outframe, (point[0], point[1]), 5, [0,0,255], -1)
        ## Fase #############################################################
        face = sub.getFace(image, wt, ht)
        if len(face)==1:
            box, keypoints = face[0]
            cv2.rectangle(outframe, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), [255, 255, 0], 1)
            for point in keypoints:
                cv2.circle(outframe, (point[0], point[1]), 5, [255,255,0], -1)
        '''
        ## FaseMesh #############################################################
        fmesh = sub.getIris(image, wt, ht)
        if len(fmesh)==1:
            meshKeys = fmesh[0]
            for point in meshKeys:
                cv2.circle(frame, (point[0], point[1]), 2, [255,0,255], -1)
        '''
        ## Pose #############################################################
        pose = sub.getPose(image, wt, ht)
        if len(pose)==1:
            poseKeys = pose[0]
            for point in poseKeys:
                cv2.circle(outframe, (point[0], point[1]), 5, [0,255, 0], -1)

        ## Segments #############################################################
        segs = sub.getSegmentImage(frame, bgimage=back)
        if len(segs)>0:
            cv2.imshow("seg", segs)            
        '''
        
        if cv2.waitKey(1)==ord('q') or ret == False:
            break
        cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()
    
if __name__=='__main__':
    main()

  ## Selfie Segmentationを例とした基本的な使い方
  以下は，MediaPipe の Selfie Segmentation 機能を利用した例です．他の機能を利用する場合も基本手順は同じとなります．
  1. mp.solutionsパッケージから使いたい機能を呼び出す
  2. processメソッドを用いて検出を行う
  3. processの戻り値（results）を分解して必要な情報を抽出する
 
 ## FaceMeshを表示するサンプル
 - MediaPipeのFaceMeshを使って468点の顔の3D特徴点を表示します
 - innerカメラは左右が反転している（鏡状になっている）ので，cv2.flip関数を使って反転しています
 - MediaPipeはRGBカラー，VideoCapture（OpenCV）はBGRカラーなのでcv2.cvtColor関数で順序の入れ替えを行っています
   - cv2.imshowの前にもう一度cv2.cvtColor関数を使ってBGRカラーに戻しています
 - Landmarkの詳しい情報は，[マニュアル](https://google.github.io/mediapipe/solutions/face_mesh.html) で確認してください 

 ```python
 ```
