import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2
import numpy as np
import time
import math
from collections import deque

from myMediapipe import myMediapipe
from myQtGraph import myGraph

import random
import pygame

dev = 0
max_length = 300
fps = 30

def polygon_area(N, P):
    return abs(sum(P[i][0]*P[i-1][1] - P[i][1]*P[i-1][0] for i in range(N))) / 2.

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def read_images(path, wwt=0, wht=0):
    imlist = deque()
    flist = os.listdir(path)
    for name in flist:
        img = cv2.imread(path+"/"+name)
        if wwt!=0 and wht!=0:
            ht, wt, _ = img.shape
            rate = 1
            rate = min(wwt,wht)/min(wt,ht)
            tmp = cv2.resize(img, None, fx=rate, fy=rate)
            img = tmp[int(wht//2-wht//2):int(wht//2+wht//2+1), int(wwt//2-wwt//2):int(wwt//2+wwt//2+1)]
            img = cvtPygameImage(img)
        imlist.append(img)

    return imlist

def cvtPygameImage(img):
    return pygame.image.frombuffer(img.tostring(), img.shape[1::-1], 'BGR')

def main():
    cap = cv2.VideoCapture(dev)
    ht = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    wt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    d_lblink = deque(maxlen=max_length)
    d_rblink = deque(maxlen=max_length)

    imlist = read_images("./img/photo", 800, 800)

    mymp = myMediapipe()
    graph = myGraph.getInstance()

    # Pygameの初期化
    pygame.init()  
    # タイトルバーの設定（大きさ600*500）
    screen = pygame.display.set_mode((800,800))
    # # タイトルバーの設定（表示する文字を指定）
    pygame.display.set_caption("Test") 
    # # タイトルバーの設定（表示する文字を指定）
    pygame.display.set_caption("Test") 

    graph.setWindowSize(1200,600)

    canvas1 = graph.setPlotCanvas("Blinks",0,0)
    curve1 = graph.setCurve(canvasid=canvas1, maxdatasize=max_length)
    curve2 = graph.setCurve(canvasid=canvas1, maxdatasize=max_length, pen=graph.makePen([255, 0, 255], graph.SOLIDLINE, width=2))

    canvas2 = graph.setPlotCanvas("Eye moving(L-R)",0,1)
    curve3 = graph.setCurve(canvasid=canvas2, maxdatasize=max_length)
    curve4 = graph.setCurve(canvasid=canvas2, maxdatasize=max_length, pen=graph.makePen([255, 0, 255], graph.SOLIDLINE, width=2))

    canvas3 = graph.setPlotCanvas("Eye moving(U-D)",0,2)
    curve5 = graph.setCurve(canvasid=canvas3, maxdatasize=max_length)
    curve6 = graph.setCurve(canvasid=canvas3, maxdatasize=max_length, pen=graph.makePen([255, 0, 255], graph.SOLIDLINE, width=2))

    graph.startRefresh()
    count = 0
    imnum = 0
    bg = time.perf_counter()
    while cap.isOpened():
        st = time.perf_counter()
        count = int((st-bg)*1000//fps)
        ret, frame = cap.read()

        image = mymp.getMPImage(frame)
        frame = np.ones((frame.shape[:3]), np.uint8)*255

        seg = mymp.getSegmentImage(image)

        plist = mymp.getDlibLandmark(image, wt, ht)
        if plist is not None:
            leyle = [[int(p[0]), int(p[1])] for p in plist['parts'][42:48]]
            reyle = [[int(p[0]), int(p[1])] for p in plist['parts'][36:42]]
            lcnt = [int((leyle[3][0]+leyle[0][0])//2), int((leyle[3][1]+leyle[0][1])//2)]
            rcnt = [int((reyle[3][0]+reyle[0][0])//2), int((reyle[3][1]+reyle[0][1])//2)]
            if len(leyle)==6:
                lblnk = polygon_area(6, leyle)/distance(leyle[3], leyle[0])
                d_lblink.append(lblnk)
                graph.setCurveData(curve1,[count],[d_lblink[-1]])
                for i, p in enumerate(leyle[:-1]):
                    cv2.line(frame, (int(p[0]), int(p[1])), (int(leyle[i+1][0]), int(leyle[i+1][1])), [0,128,255], 2)
                cv2.line(frame, (int(leyle[-1][0]), int(leyle[-1][1])), (int(leyle[0][0]), int(leyle[0][1])), [0,128,255], 2)
                cv2.circle(frame, (lcnt[0], lcnt[1]), 3, [0,128,255], -1)
                #points = np.array([[int(p[0]), int(p[1])] for p in leyle]) 
                #cv2.fillConvexPoly(frame, points, color=(0,255,0))

            if len(reyle)==6:
                rblnk = polygon_area(6, reyle)/distance(reyle[3], reyle[0])
                d_rblink.append(rblnk)
                graph.setCurveData(curve2,[count],[d_rblink[-1]])
                for i, p in enumerate(reyle[:-1]):
                    cv2.line(frame, (int(p[0]), int(p[1])), (int(reyle[i+1][0]), int(reyle[i+1][1])), [0,128,255], 2)
                cv2.line(frame, (int(reyle[-1][0]), int(reyle[-1][1])), (int(reyle[0][0]), int(reyle[0][1])), [0,128,255], 2)
                cv2.circle(frame, (rcnt[0], rcnt[1]), 3, [0,128,255], -1)
                #points = np.array([[int(p[0]), int(p[1])] for p in reyle]) 
                #cv2.fillConvexPoly(frame, points, color=(255,0,255))


            /*瞬目判定してimnumを決定する*/

            leye = plist['leye']
            reye = plist['reye']
            if leye is not None:
                graph.setCurveData(curve3, [count], [(leye[-1][0]-lcnt[0])/distance(leyle[3], leyle[0])])
                graph.setCurveData(curve5, [count], [(leye[-1][1]-lcnt[1])/distance(leyle[3], leyle[0])])
                for p in leye:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, [0,255,0], -1)
            if reye is not None:
                graph.setCurveData(curve4, [count], [(reye[-1][0]-rcnt[0])/distance(reyle[3], reyle[0])])
                graph.setCurveData(curve6, [count], [(reye[-1][1]-rcnt[1])/distance(reyle[3], reyle[0])])
                for p in reye:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 3, [255,0,255], -1)


        mask = np.all(seg == 0, axis=-1)
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2BGRA)
        seg[mask,3] = 0
        rate = 800/min(ht,wt)
        seg = cv2.resize(seg, None, fx=rate, fy=rate)
        pgseg = pygame.image.frombuffer(seg.tobytes(), seg.shape[1::-1], 'RGBA')
        screen.blit(imlist[imnum], (0,0))
        screen.blit(pgseg, (0,0))

        pygame.display.update()

        cv2.imshow("frame", frame)
        ed = time.perf_counter()
        key = cv2.waitKey(max(int((ed-st)*1000)%fps,1))
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()