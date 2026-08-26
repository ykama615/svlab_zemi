<hr>

1. [環境の設定](README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. 各種クラス・応用
   - [REALSENSE（pyrealsense2）のクラス化サンプル](#realsensepyrealsense2のクラス化サンプル)
   - [PyQtGraphのクラス化サンプル](#pyqtgraphのクラス化サンプル)
   - [MediaPipeのクラス化サンプル](#mediapipelegacy-version旧バージョンのクラス化サンプル)
   - [キーボード入力のクラス化サンプル](#キーボード入力のクラス化サンプルmy_key_press_releasepy)
   - [Minecraftをジェスチャで動かすサンプル（2023年以降未整備）](#Minecraftをジェスチャで動かすサンプル2023年以降未整備)

<hr>

# REALSENSE（pyrealsense2）のクラス化サンプル
※7/28UPDATEよりPython 3.12.9にも対応<br>
RGB-DカメラはRGB画像と同時に深度（距離）の計測が可能なカメラです．その一つである[Intel Realsense](https://www.intelrealsense.com/)はpyrealsense2ライブラリを追加すると利用可能となります．<br>
RGBのカメラと距離（Depth）センサは別々のため，alignment（位置合わせ）が必要です．

## クラスのサンプル(my_realsense2.py)
 ```python
import sys
import cv2
import numpy as np
import pyrealsense2 as rs

class myRealsense2:
  RS_PROP_FRAME_WIDTH = 0
  RS_PROP_FRAME_HEIGHT = 1
  RS_PROP_FPS = 2

  def __init__(self, width=1280, height=720, fps=30, bag=None):
    conf = rs.config()
    if bag is not None:
      rs.config.enable_device_from_file(conf, bag, repeat_playback=False)
    else:
      conf.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    conf.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    self._wt = width
    self._ht = height
    self._fps = fps

    self.pipeline = rs.pipeline()
    self.profile = self.pipeline.start(conf)

    align_to = rs.stream.color
    self.align = rs.align(align_to)

    self.colorizer = rs.colorizer()

    #depth_sensor = profile.get_device().first_depth_sensor()
    #depth_scale = depth_sensor.get_depth_scale()

  def get(self, const_str):
    if const_str == self.RS_PROP_FRAME_WIDTH:
      return self._wt
    elif const_str == self.RS_PROP_FRAME_HEIGHT:
      return self._ht
    elif const_str == self.RS_PROP_FPS:
      return self._fps
    else:
      sys.exit("Error: invalid configuration")

  def isOpened(self):
    try:
      # デバイス情報などが取得できるかチェック
      dev = self.profile.get_device()
      sensors = dev.query_sensors()
      return len(sensors) > 0
    except Exception as e:
      print("Pipeline not properly initialized:", e)
      return False

  def read(self):
    frames = self.pipeline.wait_for_frames()
    aligned_frames = self.align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)
    depth_image = np.asanyarray(depth_frame.get_data())
    colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

    return True, {'rgb': cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), 'depth': depth_image, 'colormap': colormap}

  def release(self):
    self.pipeline.stop()

 ```
## クラスの利用サンプル
 ```python
import cv2
from my_realsense2 import myRealsense2

def main():
    cap = myRealsense2(width=640, height=480, fps=30)
    ht = int(cap.get(myRealsense2.RS_PROP_FRAME_HEIGHT))
    wt = int(cap.get(myRealsense2.RS_PROP_FRAME_WIDTH))
    fps = cap.get(myRealsense2.RS_PROP_FPS)

    print('REALSENSE2')
    print('Frame Height x Width: ', ht, " x ", wt)
    print('FPS: ', fps)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret==False:
            break

        image = frame['rgb']
        depth = frame['depth'] #depth array
        colormap = frame['colormap']

        cv2.imshow("image", image)
        cv2.imshow("depth", colormap)

        if cv2.waitKey(1)==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
 ```


# PyQtGraphのクラス化サンプル
 Pythonでグラフを描くための代表的なものには，matplotlibを用いる方法とpyqtgraphを用いるものがあります．<br>
 MATLABという科学計算ソフトウェアの機能をPythonで実現しているmatplotlibを利用するとかなり様々なグラフを描くことができますが，リアルタイムな動的描画が少し難しいです．<br>
 pyqtgraphは，Qtと呼ばれれるC++によるUIデザインやソフトウェア開発を行うためのライブラリ／ソフトウェアを利用したグラフ描画機能に特化したライブラリです．ビジュアル性の高いグラフや動的グラフの描画が可能です．<br>
 ここでは，pyqtgraphを用いて時系列波形を動的/静的に描画するためのクラスを紹介します．

## クラスのサンプル(my_qt_graph.py)
 ```python
import sys
import pyqtgraph as qtg
from PyQt5 import QtWidgets, QtCore
from collections import deque

class myGraph(qtg.GraphicsLayoutWidget):
	__instance = None
	canvas = None
	curve  = None
	cvmap = None
	t1 = None
	__curve_to_canvas = {}
	__point_to_canvas = {}
	__xcap_list = {}
	__ycap_list = {}
	__xrange_list = {}
	__yrange_list = {}

	@staticmethod
	def getInstance():
		if myGraph.__instance == None:
				myGraph()
		return myGraph.__instance

	def __init__(self):
		if myGraph.__instance == None:
			myGraph.__instance = self
			self.SOLIDLINE = QtCore.Qt.SolidLine
			self.DASHLINE  = QtCore.Qt.DashLine    # 破線
			self.DOTLINE   = QtCore.Qt.DotLine     # 点線
			self.DASHDOTLINE = QtCore.Qt.DashDotLine   # 1点破線
			self.DATHDOTDOTLINE = QtCore.Qt.DashDotDotLine # 2点は線

			self.initialize()

	def initialize(self):
		self.app = QtWidgets.QApplication(sys.argv)
		qtg.setConfigOptions(antialias=True, foreground='k', background='w')
		self.win = qtg.GraphicsLayoutWidget()
		self.win.show()

		self.canvas = {}
		self.curve = {}
		self.point = {}

	def setWindowSize(self, width, height):
		self.win.resize(width, height)

	def setPlotCanvas(self, title=[], col=0, row=0):
		tmp = self.win.addPlot(show=False, size=None, title=title, col=col, row=row)
		self.canvas[id(tmp)] = tmp
		return id(tmp)

	def setCanvas(self, id):
		return self.canvas[id]

	def setCurve(self, canvasid=None, maxdatasize=300, pen=None):
		if pen==None:
			mypen = qtg.mkPen(color=(0, 255, 0), style=self.SOLIDLINE, width=3)
		else:
			mypen = pen

		if self.canvas[canvasid]==None:
			print('Canvas does not exist')
			return None

		tmp = self.canvas[canvasid].plot([], [], pen=mypen)
		self.curve[id(tmp)] = [tmp, deque(maxlen=maxdatasize), deque(maxlen=maxdatasize)]
		self.__curve_to_canvas[id(tmp)] = canvasid
		return id(tmp)

	def setCurveData(self, curveid, xlist, ylist):
		self.curve[curveid][1].extend(xlist)
		self.curve[curveid][2].extend(ylist)

	def setPoint(self, canvasid=None, symbol="o", incolor='r', outcolor='r', size="5"):
		tmp = self.canvas[canvasid].plot([], [], symbol, symbolPen=outcolor, symbolBrush=incolor, symbolSize=size, pen=None)
		self.point[id(tmp)] = [tmp, deque(), deque()]
		self.__point_to_canvas[id(tmp)] = canvasid
		return id(tmp)
	
	def setPointData(self, pointid, xlist, ylist):
		self.point[pointid][1].extend(xlist)
		self.point[pointid][2].extend(ylist)

	def makePen(self, color, style, width):
		return qtg.mkPen(color=color, style=style, width=width)

	def setStatus(self, canvasid=None, xrange=[], yrange=[], xcap=[], ycap=[]): #xcap = ["time", "s"]
		if len(xcap)>0:
			self.__xcap_list[canvasid] = xcap
			if len(xcap)==2:
				unit = xcap[1]
			else:
				unit = ''
			self.canvas[canvasid].setLabel('bottom', xcap[0], units=unit)

		if len(ycap)>0:
			self.__ycap_list[canvasid] = ycap
			if len(xcap)==2:
				unit = xcap[1]
			else:
				unit = ''
			self.canvas[canvasid].setLabel('left', ycap[0], units='')

		if len(xrange)==2:
			self.__xrange_list[canvasid] = xrange

		if len(yrange)==2:
			self.__yrange_list[canvasid] = yrange

	def show(self):
		if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
			QtWidgets.QApplication.instance().exec_()

	def startRefresh(self, interval=1000):
		self.t1 = QtCore.QTimer()
		self.t1.setInterval(interval)
		self.t1.timeout.connect(lambda:self.refresh())
		self.t1.start()

	def refresh(self):
		xlist = []
		xmin = None
		xmax = None
		for curveid, curvepack in self.curve.items():
			canvasid = self.__curve_to_canvas[curveid]
			if canvasid in self.__xrange_list:
				values = self.__xrange_list[canvasid]
				if len(values)==2:
					xmin = values[0]
					xmax = values[1]
			curve = curvepack[0]
			indices = [i for i, x in enumerate(curvepack[1]) if (xmin is None or x >= xmin) and (xmax is None or x <= xmax)]
			xlist = [curvepack[1][i] for i in indices]
			ylist = [curvepack[2][i] for i in indices]
			curve.setData(xlist, ylist)
			if xmin!=None or xmax!=None:
				self.canvas[canvasid].setXRange(xmin, xmax)

		for pointid, pointpack in self.point.items():
			canvasid = self.__point_to_canvas[pointid]
			if canvasid in self.__xrange_list:
				values = self.__xrange_list[canvasid]
				if len(values)==2:
					xmin = values[0]
					xmax = values[1]
			point = pointpack[0]
			indices = [i for i, x in enumerate(pointpack[1]) if (xmin is None or x >= xmin) and (xmax is None or x <= xmax)]
			xlist = [pointpack[1][i] for i in indices]
			ylist = [pointpack[2][i] for i in indices]
			point.setData(xlist, ylist)

		for id, values in self.__yrange_list.items():
			if len(values)==2:
				self.canvas[id].setYRange(values[0], values[1])

	def stopRefresh(self):
		self.t1.stop()

	def destroyRefresh(self):
		self.t1.deleteLater()

	def destroyWindow(self):
		self.win.deleteLater()
 ```

## クラス利用のサンプル
このクラスの使い方(1.静的グラフ, 2.動的グラフ)は次の通り．描画ウィンドウを右クリックすると，グラフの一部や全部を.pngや.jpgなど種々の画像ファイルに保存することができます．<br>
<span style="color: red;">※なお，動的グラフ作成時にmain関数でcv2.waitKey()やQTimer以外の時間制御（timeの関数やthreadingの関数の利用）を行った場合，グラフ描画クラスのスレッド（更新）も同時に停止するので注意（GIL）．</span>

### 静的グラフのサンプル
 ```python
import cv2
import numpy as np
from collections import deque
 
from my_qt_graph import myGraph
 
def main():
   graphWindow = myGraph.getInstance() #ウィンドウは1つしか生成できない
   graphWindow.setWindowSize(800,400)
 
   canvas1 = graphWindow.setPlotCanvas(title='sin', col=0,row=0) #グラフはタイル表示
   canvas2 = graphWindow.setPlotCanvas(title='cos', col=0,row=1) #グラフはタイル表示
 
   pen = graphWindow.makePen([255,0,255], graphWindow.DASHLINE, 2) #ペンを指定しなければグリーン，実線，太さ2
 
   #描画データを生成
   x  = [x for x in range(0, 720, 10)]
   sin_y = [np.sin(np.radians(y)) for y in x]
   cos_y = [np.cos(np.radians(y)) for y in x]
 
   curve1_1 = graphWindow.setCurve(canvasid=canvas1) #同じグラフに重ね描き
   curve1_2 = graphWindow.setCurve(canvasid=canvas1, pen=pen) #同じグラフに重ね描き
   curve2   = graphWindow.setCurve(canvas2)
 
   graphWindow.setCurveData(curve1_1, x, sin_y) #グラフにデータを描画
   graphWindow.setCurveData(curve1_2, x, cos_y) #グラフにデータを描画
   graphWindow.setCurveData(curve2, x, sin_y)  #グラフにデータを描画 
 
   graphWindow.setStatus(canvasid=canvas2, xrange=[0, 360], yrange=[-1, 1], xcap=['degree', '°'], ycap=['(a.u.)'])
 
   graphWindow.refresh() #グラフをウィンドウに反映
 
   graphWindow.show() #ウインドウの描画
 
if __name__=='__main__':
   main()
 ```

### 動的グラフのサンプル
 ```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
from my_qt_graph import myGraph

def main():
  graphWindow = myGraph.getInstance() #ウィンドウは1つしか生成できない
  graphWindow.setWindowSize(800,400)

  canvas1 = graphWindow.setPlotCanvas(title='sin+cos', col=0,row=0) #グラフはタイル表示
  canvas2 = graphWindow.setPlotCanvas(title='cos', col=0,row=1) #グラフはタイル表示

  pen = graphWindow.makePen([255,0,255], graphWindow.DASHLINE, 2) #ペンを指定しなければグリーン，実線，太さ2

  curve1_1 = graphWindow.setCurve(canvasid=canvas1) #同じグラフに重ね描き
  curve1_2 = graphWindow.setCurve(canvasid=canvas1, pen=pen) #同じグラフに重ね描き
  curve2   = graphWindow.setCurve(canvas2)

  graphWindow.startRefresh(interval=500) #500ミリ秒ごとに描画の繰り返しをスタート

  #描画データを生成
  for t in range(0, 9000, 10):
    sin_y = np.sin(np.radians(t))
    cos_y = np.cos(np.radians(t))

    graphWindow.setCurveData(curve1_1, [t], [sin_y]) #グラフにデータを描画
    graphWindow.setCurveData(curve1_2, [t], [cos_y]) #グラフにデータを描画
    graphWindow.setCurveData(curve2, [t], [sin_y])  #グラフにデータを描画 

    graphWindow.setStatus(canvasid=canvas1, xrange=[t-360, t], xcap=["degree"])
    graphWindow.setStatus(canvasid=canvas2, xrange=[t-360, t], xcap=["degree"])

    cv2.waitKey(100)

if __name__=='__main__':
  main()
 ```

# MediaPipe（Legacy Version／旧バージョン）のクラス化サンプル
 Legacy Version（旧バージョン）を利用したMediaPipeのクラス化です．<br>
  - 検出の処理速度は旧バージョンの利用法の方が上です
  - 新バージョンの利用法と旧バージョンの利用法を混在させるとうまく動作しないことがあります

## クラス化サンプル(my_mediapipe.py)
 ```python
import cv2
import numpy as np
import math
import mediapipe as mp

class MyMediaPipe:
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

    def getMPImage(self, frame, f_flip=0):
        """
        MediaPipe Image オブジェクトへ変換
        - f_flip=0: 反転なし
        - f_flip=1: 左右反転
        """
        if f_flip == 1:
            frame = cv2.flip(frame, 1)  # 左右反転

        mp_image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

        return mp_image

    def getFace(self, mp_image, getkeys=True):
        ht, wt, _ = mp_image.shape

        results = self.face.process(mp_image)

        point_list = []
        face_box = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                face_box = [int(bbox.xmin*wt), int(bbox.ymin*ht), int(bbox.width*wt), int(bbox.height*ht)]
                keys = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1))] for landmark in detection.location_data.relative_keypoints]
                point_list.append([face_box, keys])

        return point_list

    def getFaceMesh(self, mp_image, getkeys=True):
        ht, wt, _ = mp_image.shape
        zt = math.sqrt(wt**2 + ht**2)

        results = self.fmesh.process(mp_image)
        point_list = []
        if results.multi_face_landmarks:
            for one_face_landmarks in results.multi_face_landmarks:
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * zt)] for landmark in one_face_landmarks.landmark]
            point_list.append(tmp)

        return point_list

    def getDlibLandmark(self, mp_image, getkeys=True):
        ht, wt, _ = mp_image.shape
        zt = math.sqrt(wt**2 + ht**2)

        detection_result = self.fmesh.process(mp_image)

        landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                        296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                        380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]
        
        results = {'faces': []}
        
        if detection_result.multi_face_landmarks:
            for one_face_landmarks in detection_result.multi_face_landmarks:
                parts = [[int(one_face_landmarks.landmark[i].x * wt), int(one_face_landmarks.landmark[i].y * ht), int(one_face_landmarks.landmark[i].z * zt)] for i in landmark_points_68]

                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * zt)] for landmark in one_face_landmarks.landmark[-10:-5]]
                rpoint_list = tmp
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * zt)] for landmark in one_face_landmarks.landmark[-5:]]
                lpoint_list = tmp

                results['faces'].append({'parts': parts, 'leye': lpoint_list, 'reye': rpoint_list})
        
        return results

    def getIris(self, mp_image, getkeys=True):
        ht, wt, _ = mp_image.shape
        zt = math.sqrt(wt * wt + ht * ht)

        results = self.fmesh.process(mp_image)
        lpoint_list = []
        rpoint_list = []
        if results.multi_face_landmarks:
            for one_face_landmarks in results.multi_face_landmarks:
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * zt)] for landmark in one_face_landmarks.landmark[-10:-5]]
                rpoint_list = tmp
                tmp = [[max(1, min(int(landmark.x * wt), wt-1)), max(1, min(int(landmark.y * ht), ht-1)), int(landmark.z * zt)] for landmark in one_face_landmarks.landmark[-5:]]
                lpoint_list = tmp

        return {'leye': lpoint_list, 'reye': rpoint_list}

    def getHand(self, mp_image):
        ht, wt, _ = mp_image.shape
        zt = math.sqrt(wt*wt + ht*ht)

        tmp = cv2.flip(mp_image, 1)
        results = self.hands.process(tmp)
        hpoint_list = {"left": [], "right": []}

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                hand_points = []
                if results.multi_handedness[i].classification[0].label == "Left":
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = max(1, min(int(landmark.x * wt), wt-1))
                        y = max(1, min(int(landmark.y * ht), ht-1))
                        hand_points.append([int(abs(x-wt)), int(y), int(landmark.z*zt), landmark.visibility, landmark.presence])
                    hpoint_list["left"].append(hand_points)
                elif results.multi_handedness[i].classification[0].label == "Right":
                    for i, landmark in enumerate(hand_landmarks.landmark):
                        x = max(1, min(int(landmark.x * wt), wt-1))
                        y = max(1, min(int(landmark.y * ht), ht-1))
                        hand_points.append([int(abs(x-wt)), int(y), int(landmark.z*zt), landmark.visibility, landmark.presence])
                    hpoint_list["right"].append(hand_points)
            #hpoint_list = {'left': lhand_points, 'right': rhand_points}
        return hpoint_list

    def getPose(self, mp_image): # Mediapipe can detect only one person.
        ht, wt, _ = mp_image.shape
        zt = math.sqrt(wt*wt + ht*ht)

        results = self.pose.process(mp_image)
        pose_points = []
        pose_list = []
        if results.pose_landmarks is not None:
            for i, point in enumerate(results.pose_landmarks.landmark):
                x = max(1, min(int(point.x * wt), wt-1))
                y = max(1, min(int(point.y * ht), ht-1))
                z = int(point.z * zt)
                pose_points.append([x, y, z, point.visibility, point.presence])
            pose_list.append(pose_points)
        return pose_list

    def getSegmentImage(self, mp_image, dep=0.5):
        condition = []
        sresults = self.segment.process(mp_image)

        if sresults.segmentation_mask is not None:
            condition = np.stack((sresults.segmentation_mask,)*3, axis=-1) > dep

        return condition.astype(np.bool_)

    def getConnections(self, solution_name):
        if solution_name=='hands':
            return mp.solutions.hands.HAND_CONNECTIONS
        elif solution_name=='pose':
            return mp.solutions.pose.POSE_CONNECTIONS
        elif solution_name=='mesh':
            return mp.solutions.face_mesh.FACEMESH_TESSELATION
 ```

## クラスの利用サンプル
### 全メソッド利用
 各キーを押すことで検出を切り替えできます
 ```python
import cv2
import numpy as np
import time 
from my_mediapipe import MyMediaPipe

dev = 0

def main():

    mymp = MyMediaPipe()
    back = cv2.imread("./img/swan.jpg")

    cap = cv2.VideoCapture(dev)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sr = int(1000/fps)

    flags = {'mesh': False, 'pose': False, 'hands': False, 'face': False, 'dlib': False, 'iris': False, 'segment': False}

    fno = 0
    prv = time.perf_counter()
    while cap.isOpened():
        ret, frame = cap.read()
        image = mymp.getMPImage(frame)

        now = time.perf_counter()
        pst = int((now-prv)*1000)
        fno = fno + 1 + (pst//sr)
        sfps = min(int(fps), 1000//pst)

        key = cv2.waitKey(max(1, sr-pst))
        if ret == False or key == ord('q'):
            break

        if key == ord('m') or key == ord('a'):
            flags['mesh'] = not flags['mesh']
        if key == ord('d') or key == ord('a'):
            flags['dlib'] = not flags['dlib']
        if key == ord('p') or key == ord('a'):
            flags['pose'] = not flags['pose']
        if key == ord('h') or key == ord('a'):
            flags['hands'] = not flags['hands']
        if key == ord('f') or key == ord('a'):
            flags['face'] = not flags['face']
        if key == ord('i') or key == ord('a'):
            flags['iris'] = not flags['iris']
        if key == ord('s') or key == ord('a'):
            flags['segment'] = not flags['segment']        

        if flags['mesh']:
            plists = mymp.getFaceMesh(image)
            if plists:
                for plist in plists:
                    if len(plist)==478:
                        for id1, id2 in mymp.getConnections('mesh'):
                            cv2.line(frame, (int(plist[id1][0]), int(plist[id1][1])), (int(plist[id2][0]), int(plist[id2][1])), [255,255,255], 2)
                    for p in plist[:468]:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [0, 255, 0], -1)

        if flags['face']:
            plists = mymp.getFace(image)
            if plists:
                if len(plists[0])==2:
                    frect, plist = plists[0]
                    cv2.rectangle(frame, (frect[0], frect[1]), (frect[0]+frect[2], frect[1]+frect[3]), [255,255,255], 2)
                    for p in plist:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [0, 0, 255], -1)

        if flags['dlib']:
            plist = mymp.getDlibLandmark(image)
            if plist['faces']:  # 検出された顔がある場合
                for face_data in plist['faces']:  # 複数の顔を処理
                    if face_data['parts']:
                        for p in face_data['parts']:  # 各顔のランドマークを描画
                            cv2.circle(frame, (int(p[0]), int(p[1])), 2, [0, 255, 0], -1)


        if flags['iris']:
            plist = mymp.getIris(image)
            if plist:
                if len(plist['leye'])>0:
                    lp2d = np.array([(p[0], p[1]) for p in plist['leye']])
                    (x,y),radius = cv2.minEnclosingCircle(lp2d.astype(int))
                    cv2.circle(frame, (int(x), int(y)), int(radius), [0, 255, 255], 1)
                    for i, p in enumerate(plist['leye']):
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [0, 0, 255], -1)
                        cv2.circle(frame, (int(plist['leye'][i][0]), int(plist['leye'][i][1])), 2, [0, 0, 255], -1)
                if len(plist['reye'])>0:
                    rp2d = np.array([(p[0], p[1]) for p in plist['reye']])
                    (x,y),radius = cv2.minEnclosingCircle(rp2d.astype(int))
                    cv2.circle(frame, (int(x), int(y)), int(radius), [0, 255, 255], 1)
                    for i, p in enumerate(plist['reye']):
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [0, 0, 255], -1)
                        cv2.circle(frame, (int(plist['reye'][i][0]), int(plist['reye'][i][1])), 2, [0, 0, 255], -1)

        if flags['hands']:
            plists = mymp.getHand(image)
            if plists['right'] or plists['left']:
                landmarks_r = plists['right']
                landmarks_l = plists['left']

                for landmark in landmarks_r:
                    if len(landmark)==21:
                        for id1, id2 in mymp.getConnections('hands'):
                            cv2.line(frame, (int(landmark[id1][0]), int(landmark[id1][1])), (int(landmark[id2][0]), int(landmark[id2][1])), [255,255,255], 2)
                    for p in landmark:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [0, 255, 0], -1)

                for landmark in landmarks_l:
                    if len(landmark)==21:
                        for id1, id2 in mymp.getConnections('hands'):
                            cv2.line(frame, (int(landmark[id1][0]), int(landmark[id1][1])), (int(landmark[id2][0]), int(landmark[id2][1])), [255,255,255], 2)
                    for p in landmark:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [255, 0, 0], -1)

        if flags['pose']:
            plists = mymp.getPose(image)
            if plists:
                for plist in plists:
                    if len(plist)==33:
                        for id1, id2 in mymp.getConnections('pose'):
                            cv2.line(frame, (int(plist[id1][0]), int(plist[id1][1])), (int(plist[id2][0]), int(plist[id2][1])), [255,255,255], 2)
                    for p in plist:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 2, [255, 0, 255], -1)

        if flags['segment']:
            sg = mymp.getSegmentImage(image)
            frame[sg==False] = 255


        cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

if __name__=='__main__':
    main()
 ```
### ポーズ検出だけの利用サンプル
 ```python
import sys
import cv2
import numpy as np

from my_mediapipe import MyMediaPipe

dev = 0

def main():
    global dev 
    args = sys.argv[1:]
    if len(args) > 1:
        if args[0].isdigit():
            dev = int(args[0])
        else:
            dev = args[0]

    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mymp = MyMediaPipe()
    clists = mymp.getConnections('pose')
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        # CLAHEの適用
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])  # Yチャンネル（輝度）
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        image = mymp.getMPImage(frame)
        plists = mymp.getPose(image)
        if plists:
            plist = plists[0]
            connect = []
            for i, p in enumerate(plist):
                if p[3] > 0.6:
                    connect = [i1 for i0, i1 in clists if i0 == i]
                    for c in connect:
                        if plist[c][3] > 0.6:
                            cv2.line(frame, (int(plist[i][0]), int(plist[i][1])), (int(plist[c][0]), int(plist[c][1])), [255,255,255], 2)

                    cv2.circle(frame, (p[0], p[1]), 3, [255,0,0], -1)
        
        cv2.imshow("video", frame)

        wt = 1
        if cv2.waitKey(wt) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
 ```

 # キーボード入力のクラス化サンプル(my_key_press_release.py)

 w/a/s/dを押す（TRUE），離す（FALSE）関数と， キーコードを送るとキー入力を行う関数を実装したクラス．

 ## クラス化サンプル
 ```python
 import win32gui
 import time
 from ctypes import windll
 
 class KeyPressRelease:
 
   CODE = {'backspace':0x08, 'tab':0x09, 'clear':0x0C, 'enter':0x0D, 'shift':0x10, 'ctrl':0x11,
     'alt':0x12, 'pause':0x13, 'caps_lock':0x14, 'esc':0x1B, 'spacebar':0x20, 'page_up':0x21, 'page_down':0x22,
     'end':0x23, 'home':0x24, 'left_arrow':0x25, 'up_arrow':0x26, 'right_arrow':0x27, 'down_arrow':0x28,
     'select':0x29, 'print':0x2A, 'execute':0x2B, 'print_screen':0x2C, 'ins':0x2D, 'del':0x2E, 'help':0x2F,
     '0':0x30, '1':0x31, '2':0x32, '3':0x33, '4':0x34, '5':0x35, '6':0x36, '7':0x37, '8':0x38, '9':0x39,
     'a':0x41, 'b':0x42, 'c':0x43, 'd':0x44, 'e':0x45, 'f':0x46, 'g':0x47, 'h':0x48, 'i':0x49, 'j':0x4A,
     'k':0x4B, 'l':0x4C, 'm':0x4D, 'n':0x4E, 'o':0x4F, 'p':0x50, 'q':0x51, 'r':0x52, 's':0x53, 't':0x54,
     'u':0x55, 'v':0x56, 'w':0x57, 'x':0x58, 'y':0x59, 'z':0x5A, 'numpad_0':0x60, 'numpad_1':0x61, 'numpad_2':0x62,
     'numpad_3':0x63, 'numpad_4':0x64, 'numpad_5':0x65, 'numpad_6':0x66, 'numpad_7':0x67, 'numpad_8':0x68,
     'numpad_9':0x69, 'multiply_key':0x6A, 'add_key':0x6B, 'separator_key':0x6C, 'subtract_key':0x6D,
     'decimal_key':0x6E, 'divide_key':0x6F, 'F1':0x70, 'F2':0x71, 'F3':0x72, 'F4':0x73, 'F5':0x74, 'F6':0x75,
     'F7':0x76, 'F8':0x77, 'F9':0x78, 'F10':0x79, 'F11':0x7A, 'F12':0x7B, 'F13':0x7C, 'F14':0x7D, 'F15':0x7E,
     'F16':0x7F, 'F17':0x80, 'F18':0x81, 'F19':0x82, 'F20':0x83, 'F21':0x84, 'F22':0x85, 'F23':0x86, 'F24':0x87,
     'num_lock':0x90, 'scroll_lock':0x91, 'left_shift':0xA0, 'right_shift ':0xA1, 'left_control':0xA2, 'right_control':0xA3,
     'left_menu':0xA4, 'right_menu':0xA5, 'browser_back':0xA6, 'browser_forward':0xA7, 'browser_refresh':0xA8,
     'browser_stop':0xA9, 'browser_search':0xAA, 'browser_favorites':0xAB, 'browser_start_and_home':0xAC,
     'volume_mute':0xAD, 'volume_Down':0xAE, 'volume_up':0xAF, 'next_track':0xB0, 'previous_track':0xB1,
     'stop_media':0xB2, 'play/pause_media':0xB3, 'start_mail':0xB4, 'select_media':0xB5, 'start_application_1':0xB6,
     'start_application_2':0xB7, 'attn_key':0xF6, 'crsel_key':0xF7, 'exsel_key':0xF8, 'play_key':0xFA,
     'zoom_key':0xFB, 'clear_key':0xFE, '+':0xBB, ',':0xBC, '-':0xBD, '.':0xBE, '/':0xBF, '`':0xC0, ';':0xBA,
     '[':0xDB, '\\':0xDC, ']':0xDD, "'":0xDE, '`':0xC0
   }
 
   def __init__(self):
     self.press_up  = False
     self.press_dwn = False
     self.press_lft = False
     self.press_rgt = False
 
   def inputKey(self, code):
     windll.user32.keybd_event(code, 0, 0, 0)      
     time.sleep(0.05)
     windll.user32.keybd_event(code, 0, 2, 0)      
 
   def setWindow2Foreground(self, game_name):
     sleep_time = 0.05
     mcapp = win32gui.FindWindow(None,game_name)
     time.sleep(sleep_time)
     win32gui.SetForegroundWindow(mcapp)         #ウィンドウの指定
     time.sleep(sleep_time)
     hwnd = win32gui.GetForegroundWindow()
     time.sleep(sleep_time)
     win32gui.MoveWindow(hwnd, 0, -10, 940, 1000, True)
 
   def printStatus(self):
     list = []
     if self.press_up:
       list.append('w')
     if self.press_dwn:
       list.append('s')
     if self.press_lft:
       list.append('a')
     if self.press_rgt:
       list.append('d')
 
     print('press: ', list)
 
   def up(self, bool):
     if bool:
       if self.press_up==False:
         self.press_up = True
         windll.user32.keybd_event(self.CODE['w'], 0, 0, 0)
     else:
       if self.press_up==True:
         self.press_up = False
         windll.user32.keybd_event(self.CODE['w'], 0, 2, 0)
 
   def down(self, bool):
     if bool:
       if self.press_dwn==False:
         self.press_dwn = True
         windll.user32.keybd_event(self.CODE['s'], 0, 0, 0)
     else:
       if self.press_dwn==True:
         self.press_dwn = False
         windll.user32.keybd_event(self.CODE['w'], 0, 2, 0)
 
   def left(self, bool):
     if bool:
       if self.press_lft==False:
         self.press_lft = True
         windll.user32.keybd_event(self.CODE['a'], 0, 0, 0)
     else:
       if self.press_lft==True:
         self.press_lft = False
         windll.user32.keybd_event(self.CODE['a'], 0, 2, 0)
 
   def right(self, bool):
     if bool:
       if self.press_rgt==False:
         self.press_rgt = True
         windll.user32.keybd_event(self.CODE['d'], 0, 0, 0)
     else:
       if self.press_rgt==True:
         self.press_rgt = False
         windll.user32.keybd_event(self.CODE['d'], 0, 2, 0)
 ```

## クラスの利用サンプル
 メモ帳を開いておくとメモ帳に何秒かごとのタイミングでキー入力されるサンプルは以下の通り．
 
 ```python
   import time
   from KeyPressRelease import KeyPressRelease

   def main():
     control = KeyPressRelease()
  
     control.setWindow2Foreground('タイトルなし - メモ帳')
  
     for i in range(5):
       time.sleep(3)    
       control.up(True)
       control.printStatus()
       time.sleep(1)    
       control.right(True)
       control.printStatus()
       time.sleep(1)    
       control.right(False)
       control.printStatus()
       control.left(True)
       control.printStatus()
       time.sleep(2)    
       control.left(False)
       control.printStatus()
  
     time.sleep(3)    
     control.up(False)
     control.printStatus()
     control.down(True)
     control.printStatus()
     time.sleep(2)
     control.down(False)
     control.printStatus()

     control.inputKey(control.CODE['f'])
   
 if __name__=='__main__':
   main()
 ```

# Minecraftをジェスチャで動かすサンプル（2023年以降未整備）
  ・以下のファイルと↑の myMediapipe.py を利用した例
    ・mc_init.py（initクラス：ウィンドウの名前からハンドルを取得して，ウィンドウのサイズと位置を指定してアクティブにするクラス）
    ・mc_movePlayer.py（movePlayerクラス：wasdのキーダウン，アップを行う関数を実装したクラス）

 ```python
# mc_init.py
import time

import win32gui
import ctypes
import ctypes.wintypes
from ctypes import windll

################################
game_name = 'Minecraft Education'
sleep_time = 0.1
################################

def init():
    mcapp = win32gui.FindWindow(None, game_name)
    windll.user32.SetForegroundWindow(mcapp)         #ウィンドウの指定
    time.sleep(sleep_time)
    hwnd = windll.user32.GetForegroundWindow()
    windll.user32.MoveWindow(hwnd, 0, 0, 1600, 900, True)
    rect = ctypes.wintypes.RECT()
    windll.user32.GetWindowRect(hwnd, ctypes.pointer(rect))

    print(rect.left , rect.top , rect.right , rect.bottom)

if __name__ == '__main__':
    init()
 ```

 ```python
# mc_movePrayer.py
import ctypes
import ctypes.wintypes
from ctypes import windll

class movePlayer:
    def __init__(self):
        self.press_fwd = False
        self.press_bck = False
        self.press_lft = False
        self.press_rgt = False

    def forward(self, bool):
        if bool:
            
            if self.press_fwd==False:
                self.press_fwd = True
                windll.user32.keybd_event(0x57, 0, 0, 0)
        else:
            if self.press_fwd==True:
                self.press_fwd = False
                windll.user32.keybd_event(0x57, 0, 2, 0)

    def backward(self, bool):
        if bool:
            if self.press_bck==False:
                self.press_bck = True
                windll.user32.keybd_event(0x53, 0, 0, 0)
        else:
            if self.press_bck==True:
                self.press_bck = False
                windll.user32.keybd_event(0x53, 0, 2, 0)

    def left(self, bool):
        if bool:
            if self.press_lft==False:
                self.press_lft = True
                windll.user32.keybd_event(0x41, 0, 0, 0)
        else:
            if self.press_lft==True:
                self.press_lft = False
                windll.user32.keybd_event(0x41, 0, 2, 0)

    def right(self, bool):
        if bool:
            if self.press_rgt==False:
                self.press_rgt = True
                windll.user32.keybd_event(0x44, 0, 0, 0)
        else:
            if self.press_rgt==True:
                self.press_rgt = False
                windll.user32.keybd_event(0x44, 0, 2, 0)


if __name__=='__main__':
    import time
    import mc_init
    mc_init.init()
    windll.user32.keybd_event(0x1b, 0, 1, 0)
    test = movePlayer()
    for i in range(5):
        time.sleep(3)    
        test.forward(True)
        time.sleep(1)    
        test.right(True)
        time.sleep(1)    
        test.right(False)
        test.left(True)
        time.sleep(2)    
        test.left(False)

    time.sleep(3)    
    test.forward(False)
    test.backward(True)
    time.sleep(2)
    test.backward(False)
 ```

  動作サンプルプログラムは以下の通り．
   1. mincraftのウィンドウをアクティブにする
   2. 別のディスプレイにOpenCVのウィンドウを開いてカメラ映像に操作用のイラストを重畳して表示
   3. 左手の人差し指の先端がどこにあるかによってwasdが押・離されてmincraft内のキャラクターが動く
     <img src="./control.jpg" width="200px">

 ```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import math

from myMediapipe import myMrdiapipe

from mc_init import init
from mc_movePlayer import movePlayer

dev = 0

def main():
    mp = myMrdiapipe(detection=0.8, tracking=0.8)
    move = movePlayer()

    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cv2.namedWindow("video", cv2.WINDOW_NORMAL)
    cv2.moveWindow("video", 1600, 0)

    back = cv2.imread("./back2.bmp")
    imh, imw, _ = back.shape
    ratio = ht/imh
    back = cv2.resize(back, None, fx=ratio, fy=ratio)
    imh, imw, _ = back.shape
    lx, ly = [wt//2-imw//2, 0]
    cx, cy = [wt//2, ht//2]
    white = np.ones((ht, wt, 3), dtype=np.uint8)*255
    white[ly:ly+imh, lx:lx+imw] = back

    init()
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        image = mp.getMPImage(frame)

        ## Hands #############################################################
        hands = mp.getHand(image, wt, ht)

        frame[white<=250] = white[white<=250]
        if len(hands)==1:
            left = hands[0][0]

            r = 0
            deg = 0
            if len(left)>0:
                cv2.circle(frame, (left[8][0], left[8][1]), 5, [0,0,255], -1)
                dx = left[8][0] - cx
                dy = left[8][1] - cy
                r = int(math.dist([0, 0], [dx, dy]))
                deg = int(math.degrees(math.atan2(-dy, dx)))

                if (100*ratio<=r) and (r<=150*ratio):
                    #print(r, deg)
                    if (0<= abs(deg)) and (abs(deg)<=60):
                        move.right(True)
                    elif ( abs(deg)< 0 or 60 < abs(deg)):
                        move.right(False)

                    if (120<= abs(deg)) and (abs(deg)<=180):
                        move.left(True)
                    else:
                        move.left(False)

                    if (30<= deg) and (deg<=150):
                        #print("foward")
                        move.forward(True)
                    else:
                        move.forward(False)

                    if (-150<= deg) and (deg<=-30):
                        #print("backward")
                        move.backward(True)
                    else:
                        move.backward(False)

                else:
                    move.forward(False)
                    move.backward(False)
                    move.left(False)
                    move.right(False)
            else:
                move.forward(False)
                move.backward(False)
                move.left(False)
                move.right(False)

        if cv2.waitKey(1) & 0xFF == ord('q') or ret == False:
            break

        cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

if __name__=='__main__':
    main()
 ```
