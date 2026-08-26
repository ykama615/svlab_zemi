<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. 顔と顔パーツの検出（↓）
5. [顔・手・ポーズ検出](BASIC_03.md)
6. [各種クラス・応用](BASIC_04.md)

<hr>

# 顔検出

## 準備
**※ `learned_models` フォルダがない場合、`learned_models.zip` をダウンロード・解凍し、ソースコードフォルダ内に配置してください。**

> **補足**: `cv2.face` (Facemark API) を使用するには、`opencv-contrib-python` のインストールが必要です。
> ```sh
> % pip install opencv-contrib-python
> ```

---

## Haar-like 特徴量を用いた顔検出

Haar-like 特徴量（矩形領域の濃淡パターン）を使った顔検出です。OpenCV では学習済みの分類器 XML ファイルが提供されています。
以下のサンプルは、静止画像に対して正面顔（`haarcascade_frontalface_default.xml`）と、検出領域内の目（`haarcascade_eye.xml`）を検出する例です。

```python
import cv2

mdl_folder = "./learned_models/"  # 学習済みモデルのパス
img_folder = "./img/standard/"     # 画像ファイルのパス

def main():
    face_cascade = cv2.CascadeClassifier(mdl_folder + "haarcascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(mdl_folder + "haarcascades/haarcascade_eye.xml")
    
    img  = cv2.imread(img_folder + "Girl.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ######## 顔の検出 ########
    # detectMultiScale(画像, 縮小スケール比率, 最小近傍矩形数)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # 検出された顔領域を順に取り出す
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        ######## 顔領域内から目を検出 ########
        face_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 1)

    cv2.imshow("haar-like", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

| 元画像 | 検出結果 |
| --- | --- |
|  |  |

---

## LBF（Local Binary Features）を用いた顔パーツ検出

Local Binary Features（輝度値の差分情報）を使った顔パーツ検出です。
Haar-like により検出した正面顔領域に対して FaceMark API を適用し、68 点のランドマーク（特徴点頂点）を描画します。学習済みモデルには `lbfmodel.yaml` を利用します。

```python
import cv2
import numpy as np

mdl_folder = "./learned_models/"  # 学習済みモデルのパス
img_folder = "./img/standard/"     # 画像ファイルのパス

def main():
    face_cascade = cv2.CascadeClassifier(mdl_folder + "haarcascades/haarcascade_frontalface_default.xml")
    fmdetector = cv2.face.createFacemarkLBF()
    fmdetector.loadModel(mdl_folder + "lbfmodel.yaml")
    
    img = cv2.imread(img_folder + "Girl.bmp")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ######## 顔の検出 ########
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        ######## 顔の68点特徴点を検出 ########
        ok, landmarks = fmdetector.fit(img, np.array([face]))
        if ok:
            # parts[0]～[67] にランドマーク座標が格納される
            parts = np.array(landmarks[0][0], dtype=np.int32)

            for point in parts:  # point[0]: x, point[1]: y
                cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)

    cv2.imshow("LBF", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

| 検出結果 |
| --- |
|  |

---

## dlib を用いた顔パーツ検出

dlib C++ Library に実装されたアルゴリズム（Kazemi らの CVPR 2014 論文に基づく）を使用した高精度な顔および顔パーツ検出です。
学習済みモデルには `shape_predictor_68_face_landmarks.dat` を使用します。また、座標変換を補佐するために `imutils` ライブラリを利用します。

> **imutils ライブラリのインストール**:
> ```sh
> % pip install imutils
> 
> ```
> 
> 

```python
import cv2
import dlib
from imutils import face_utils
import numpy as np

mdl_folder = "./learned_models/"  # 学習済みモデルのパス
img_folder = "./img/standard/"     # 画像ファイルのパス

def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(mdl_folder + "shape_predictor_68_face_landmarks.dat")
    
    img = cv2.imread(img_folder + "Girl.bmp")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. 顔の検出
    dets, scores, _ = detector.run(rgb, 1)

    for i, face in enumerate(dets):
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 2. 顔の中から68点の特徴点を検出
        shape = predictor(rgb, face)
        parts = face_utils.shape_to_np(shape, dtype=np.int32)

        for j, point in enumerate(parts):  # parts[j][0]: x, parts[j][1]: y
            cv2.circle(img, (point[0], point[1]), 2, (0, 255, 0), -1)

    cv2.imshow("dlib", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```

| 検出結果 |
| --- |
|  |

---

# 課題・演習

## 【課題】 カメラ（動画）からの顔・顔パーツ検出

Haar-like + LBF や dlib による顔・顔パーツ検出を、リアルタイムのカメラ映像に適用するプログラムを作成してみましょう。

* **ポイント**: 学習済みモデルの読み込み (`CascadeClassifier` や `shape_predictor`) は、フレームを取得する `while` ループの外で行ってください。

## 【課題】 まばたきシャッター / スマイルシャッター

顔パーツのランドマーク（上まぶた・下まぶた、または口角の左右座標）のインデックス番号を調べ、その距離や位置の変化を検知した瞬間にフレーム画像を自動保存するプログラムを作成してみましょう。

* フレームの保存方法については [BASIC_01.md](https://www.google.com/search?q=BASIC_01.md) を参照してください。

---

# 【エクストラ】 配布環境の自作ライブラリの利用

配布環境には、OpenCV や dlib による顔・顔パーツ検出を簡単に行えるクラスパッケージが用意されています。

* `mylibs/myPhysiology` パッケージ内 `cv_face_detector.py` (`cvFaceDetector` クラス)
* `mylibs/myPhysiology` パッケージ内 `dlib_face_detector.py` (`DlibFaceDetector` クラス)

```python
import cv2
import time
import mylibs.myCapture as mycap
import mylibs.myPhysiology as myphy

dev = 0

def main():
    cap = mycap.CameraSelector(dnum=dev, fps=30, size=[720, 1280])
    dlib_detector = myphy.DlibFaceDetector() # OpenCVの場合は myphy.cvFaceDetector()

    while cap.isOpened():
        ret, fnum, frame = cap.read()

        if ret:
            dets, _, _ = dlib_detector.getFace(frame)

            if len(dets) > 0:
                for face in dets:
                    cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), [0, 0, 255], 1)

                    parts = dlib_detector.getFacemark_detection(frame, face)
                    for x, y in parts:
                        cv2.circle(frame, (x, y), 3, [0, 255, 0], -1)

            cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```
