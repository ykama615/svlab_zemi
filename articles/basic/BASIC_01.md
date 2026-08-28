<hr>

**講義ノート・ライブラリ一覧**

<b>➡基礎編</b>
1. [環境の設定](../../README.md)
2. [基本概要](BASIC_00.md)
3. カメラへのアクセスと動画処理（↓）
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. [2つのベクトルのなす角とベクトル演算](BASIC_FP01.md)

<details><summary><b>キャプチャ（3項目）</b></summary>

7. [動画画像処理 (`my_cap_av2.py`)](../lecnote/lecnote_cap01.md)
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](../lecnote/lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](../lecnote/lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](../lecnote/lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](../lecnote/lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](../lecnote/lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](../lecnote/lecnote_dt04.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](../lecnote/lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](../lecnote/lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](../lecnote/lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](../lecnote/lecnote_an04.md)
</details>

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](../lecnote/lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](../lecnote/lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](../lecnote/lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>

OpenCVを用いたWebカメラの制御手法から画像処理演習による環境構築までを実例付きでまとめた解説ドキュメントです．

<hr>

# カメラへのアクセス

もっとも簡単なカメラアクセスのサンプルプログラムは以下の通りです．
- グローバル変数 `dev` にカメラのデバイス番号（または動画ファイルパス）を指定します．
- 'q' キーを押すとプログラムが終了します．

```python
# script4.py
import cv2

dev = 0

def main():
    cap = cv2.VideoCapture(dev)
    ht  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    wt  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

### 内蔵カメラ以外のWebカメラ等を利用する場合

#### 1. MSMF(Microsoft Media Foundation)の設定

USB接続カメラの場合，`cv2.VideoCapture` によるカメラの起動が遅くなる場合があります．これを回避するため，メインプログラムの先頭（`import cv2` より前）に以下の設定を記述します．

```python
# cv2のインポート前にカメラ環境変数の設定を行う
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2

```

#### 2. DirectShow経由で利用する方法

1 のほかに，`VideoCapture` の引数にバックエンドフラグを指定する方法も利用できます．

```python
# DirectShow経由でカメラ映像を取得する
cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)

```

---

# 課題・演習

## 【課題】 Let's Selfy プログラム

キー 's' を押すと，その瞬間のフレームを画面表示・保存するプログラムを作成してみましょう．

* **ヒント**:

```python
key = cv2.waitKey(1)
if key & 0xFF == ord('q'):
    break
elif key & 0xFF == ord('s'):
    cv2.imshow("image", frame)
    cv2.imwrite("photo.jpg", frame)

```

## 【課題】 グレースケール・ビデオ・プログラム

カメラ映像をリアルタイムでグレースケールに変換して表示するプログラムを作成してみましょう．

* **ヒント**:

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

```

## 【課題】 エッジ・ビデオ・プログラム

グレースケール・ビデオ・プログラムに Canny 法によるエッジ検出処理を追加してみましょう．

* **エッジとは**: 画像中の色や明るさが極端に変化する（不連続な）部分のことで，写っている物体の境界線を示します．
* **ヒント**:

```python
edges = cv2.Canny(gray, 100, 200)

```

---

# タイムラプス

* 15フレームに1回（30 FPS であれば 0.5 秒に 1 回）ずつ，取得したフレームを `deque` に追加していくことでタイムラプス動画を作成してみましょう．
* 'q' キーを押すとタイムラプスの収録を終了し，収録されたフレームが連続再生されます．
* 収録タイミング（間隔）を変更して動作を確認してみましょう．

```python
import cv2
import numpy as np
from collections import deque

dev = 0

def main():
    cap = cv2.VideoCapture(dev)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # FPS取得失敗（0.0）時のゼロ除算対策
    wait_msec = int(1000.0 / fps) if fps > 0 else 33

    timelapse = deque()
    fnum = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow("video", frame)
        
        # 15フレームごとにdequeの末尾へ追加
        if fnum % 15 == 0:
            timelapse.append(frame)
        fnum += 1

        if cv2.waitKey(wait_msec) & 0xFF == ord('q'):
            break

    # 収録したdequeの内容を再生
    for frame in timelapse:
        cv2.imshow("timelapse", frame)
        if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

---
