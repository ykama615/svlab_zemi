<hr>

**講義ノート・ライブラリ一覧**

<details><summary><b>基礎編（6項目）</b></summary>
  
1. [環境の設定](../../README.md)
2. [基本概要](../basic/BASIC_00.md)
3. [カメラへのアクセスと動画処理](../basic/BASIC_01.md)
4. [顔と顔パーツの検出](../basic/BASIC_02.md)
5. [顔・手・ポーズ検出](../basic/BASIC_03.md)
6. [2つのベクトルのなす角とベクトル演算](../basic/BASIC_FP01.md)
</details>

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

<b>➡その他（3項目）</b>

21. ドローン/RTPキャプチャ（↓）
22. [スクリーンキャプチャ](../minecraft/lecnote_mc01.md)
23. [【旧】Minecraftコントロール(1)](../minecraft/mcbot_01.md)

<hr>

自作ライブラリ my_libs.capture 内の VideoCapture を活用し、RTP ストリーミングの受信、重複フレームの破棄、およびVFR対応の録画機能を実装するための解説ドキュメントです。

<hr>

# RTPストリーミング用ライブラリ (`my_cap_rtp.py`) の使い方

## 概要

* `./my_libs/capture/my_cap_rtp.py` 内の `VideoCapture` および `VideoWriter` クラスを用いて、Hulaドローン（`pyhula` SDK）によるRTPストリーム映像の取得・録画を行います。
* **公式 `pyhula` パッケージの `UserApi` インスタンス連携による安全なRTPストリーム制御**
* **RTPタイムスタンプを活用した重複フレームの動的ドロップ処理**
* **ネットワーク揺らぎ・遅延に対応したVFR（可変フレームレート）前提の堅牢な H.264 録画 (`VideoWriter`)**

---

## 前提条件

* **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
* `my_cap_rtp.py`: `./my_libs/capture/`


* **【重要】** 公式の `pyhula` ライブラリがインストールされ、ドローンとのWi-Fi接続および `UserApi` が正常に初期化できる環境が必要です。

---

## **my_cap_rtp.py の概要と特徴**

`my_cap_rtp.py` は、ファイル読み出しやシーク処理などの不要なロギングを一切持たず、リアルタイムRTPパケットの受信と重複フレームの破棄、およびネットワーク遅延に耐性を持つ動画保存に特化した専用ライブラリです。

### 主な特徴

1. **公式 `pyhula` 連携によるデバイス初期化**:
* `pyhula.UserApi()` インスタンスを内部で安全にラップし、RTPストリーミングの有効化コマンドや初期フレームの受信待機を自動で行います。


2. **タイムスタンプ重複チェック**:
* ストリームから取得するフレームのRTPタイムスタンプを監視し、同一または古いタイムスタンプの重複フレームを自動的にドロップしてカクつきを防ぎます。


3. **RTP・VFR最適化 Writer**:
* ネットワークの変動による遅延を正確に記録・エンコードできるよう、実測ミリ秒をPTSに直接反映する堅牢な `VideoWriter` を備えています。



---

## **RTPストリーミング受信用サンプル (`rtp_viewer.py`)**

公式の `pyhula` SDKから取得した `UserApi` インスタンスを `VideoCapture` に渡してリアルタイム映像を受信・表示する基本サンプルです。

### rtp_viewer.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import pyhula
from my_libs.capture.my_cap_rtp import VideoCapture

def main():
    # 公式 pyhula SDK の UserApi インスタンスを取得・初期化
    api_source = pyhula.UserApi()
    if not api_source.is_open():
        print("pyhula の接続に失敗しました。")
        return

    cap = VideoCapture(api_source)
    
    if not cap.isOpened():
        print("RTPストリームを開けませんでした。")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("RTP Size:", ht, "x", wt, "/ Fps:", fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.imshow("RTP Stream", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

### VideoCapture (RTP) メソッドとプロパティ

| コード | 内容・説明 |
| --- | --- |
| `VideoCapture(api_source)` | `pyhula.UserApi` インスタンスを受け取り、RTPの初期化とストリーム接続を実行 |
| `cap.read()` | RTPタイムスタンプの重複チェックを行った上で1フレーム（BGR形式）を取得 |
| `cap.get(cv2.CAP_PROP_POS_MSEC)` | 接続開始からの実経過ミリ秒 |
| `cap.get(cv2.CAP_PROP_FPS)` | 固定の基準値（`30.0`） |
| `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | 固定の画面幅（`1280.0` または対応解像度） |
| `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` | 固定の画面高さ（`720.0` または対応解像度） |

---

## **RTP専用 VideoWriter による録画とログ出力**

RTPストリームの録画では、ネットワーク遅延やフレームドロップによる破綻を防ぐため、受け取った実測ミリ秒をそのままPTSに反映する `VideoWriter` を使用します。保存と同時に `.log.csv` が出力されます。

### rtp_recorder.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import pyhula
from my_libs.capture.my_cap_rtp import VideoCapture, VideoWriter

def main():
    api_source = pyhula.UserApi()
    if not api_source.is_open():
        print("pyhula の接続に失敗しました。")
        return

    cap = VideoCapture(api_source)
    
    if not cap.isOpened():
        print("RTPストリームを開けませんでした。")
        return

    video_name = "./img/rtp_record.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    writer = VideoWriter(video_name, fps=fps, frame_size=(int(wt), int(ht)), is_vfr=True)
    recflag = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        if recflag:
            current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            writer.write(frame, custom_msec=current_msec)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recflag = not recflag
    
        cv2.imshow("RTP Stream", frame)
    
    writer.release()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```
