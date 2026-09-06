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

21. [ドローン/RTPキャプチャ](../drone/lecnote_dr01.md)
22. スクリーンキャプチャ（↓）
23. [【旧】Minecraftコントロール(1)](../minecraft/mcbot_01.md)

<hr>

自作ライブラリ `my_libs.capture` 内の `my_cap_screen.py` の画面・ウィンドウ用キャプチャクラス `VideoCapture` を活用し、マルチモニターや領域指定、ウィンドウの自動追従によるデスクトップ映像の取得および録画機能を実装するための解説ドキュメントです。
<br>**`my_libs` を利用するには `mylibspack.7z` を展開し、`my_libs`・`learned_models`・`img` の 3 つのフォルダをソースディレクトリ直下に並列に配置する必要があります。**

<hr>

# 画面・ウィンドウキャプチャライブラリ (`my_cap_screen.py`) の使い方

## 概要

* `./my_libs/capture/my_cap_screen.py` 内の `VideoCapture` および `VideoWriter` クラスを用いて、デスクトップ画面や特定ウィンドウの映像ストリームを取得・録画します。
* OpenCV互換の操作感で、以下のキャプチャ・録画機能を処理します。
* **3種類の入力ソース（モニター番号 / 領域指定 / ウィンドウタイトル検索）の自動判別機能**
* **Windows環境における指定ウィンドウの位置自動追従 (`window_title`)**
* **OpenCV標準と同等の直感的なインターフェース（`read`, `get`, `release`, `isOpened`）**
* **カスタムラップによるシームレスな動画保存 (`VideoWriter`)**

---

## 前提条件

* **【重要】** 以下の外部ライブラリがインストールされていることを確認してください。
```bash
pip install mss pywin32 opencv-python numpy

```


* ライブラリ用スクリプトが `./my_libs/` 配下に配置されていることを想定しています。

---

## :red_square: 画面・ウィンドウキャプチャの概要と特徴

`screen_capture.py` は、OpenCVの `cv2.VideoCapture` と高い互換性を持ちつつ、デスクトップ全体やマルチモニター、特定のアプリケーションウィンドウを切り出してフレームを取得するカスタムライブラリです。

### 主な特徴

1. **OpenCV互換のインターフェース**:
* `cap = VideoCapture(...)` や `cap.read()`, `cap.get(...)`, `cap.release()` などの OpenCV 標準と同等のメソッドを提供します。


2. **多様な入力ソースの自動判別**:
* **モニター番号 (`int`)**: 整数（`0`, `1` など）を渡すと指定モニター全体をキャプチャします。
* **ウィンドウタイトル (`str`)**: アプリケーションのウィンドウタイトル文字列を渡すと、該当ウィンドウを検索してリアルタイムに追従キャプチャします。
* **画面領域指定 (`tuple`)**: `("screen", monitor_index, (left, top, width, height))` の形式で特定エリアのみを切り出します。


3. **ウィンドウ位置の自動追従**:
* ウィンドウタイトルを指定した場合、対象のウィンドウが移動したりリサイズされたりしても、自動で追従してキャプチャ範囲を更新します。



---

## :red_square: 基本的な使い方とサンプル (`screen_viewer.py`)

`VideoCapture` にウィンドウタイトルやモニター番号を渡し、リアルタイムに画面を表示する基本的なサンプルコードです。

### screen_viewer.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from my_libs.screen_capture import VideoCapture

# 入力ソースの指定例:
# 1. 整数 (モニター番号): 0
# 2. 文字列 (ウィンドウタイトル): "メモ帳" または "Google Chrome"
# 3. タプル (画面内領域): ("screen", 0, (100, 100, 800, 600))
source = "メモ帳" 

def main():
    # VideoCapture の初期化
    cap = VideoCapture(source)
    
    if not cap.isOpened():
        print("キャプチャを開始できませんでした。")
        return

    # プロパティの取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Capture Size: {int(wt)}x{int(ht)} / Target FPS: {fps:.1f}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("フレームを取得できませんでした。")
            break

        # 画面に表示
        cv2.imshow("Screen Capture Viewer", frame)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

### VideoCapture メソッドとプロパティ

| メソッド / プロパティ | 内容・説明 |
| --- | --- |
| `VideoCapture(source)` | モニター番号、ウィンドウタイトル、または領域タプルを指定してキャプチャセッションを開始 |
| `cap.read()` | 1フレームを取得。成功フラグ (`bool`) と BGR形式の画像データ (`ndarray`) を返却 |
| `cap.get(cv2.CAP_PROP_POS_MSEC)` | キャプチャ開始からの経過時間（ミリ秒） |
| `cap.get(cv2.CAP_PROP_FPS)` | 実測に基づく現在のフレームレート (FPS) |
| `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | キャプチャ領域の横幅 |
| `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` | キャプチャ領域の縦幅 |
| `cap.get(cv2.CAP_PROP_POS_FRAMES)` | 取得済みフレームの総数 |

---

## :red_square: 画面録画機能 (`screen_recorder.py`)

ラップされた `VideoWriter` を用いて、キャプチャしたデスクトップやウィンドウ映像を動画ファイルとして保存するサンプルコードです。

### screen_recorder.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from my_libs.screen_capture import VideoCapture, VideoWriter

source = 0  # 0番目のモニター全体を対象とする
output_filename = "desktop_record.mp4"

def main():
    recflag = False

    cap = VideoCapture(source)
    if not cap.isOpened():
        print("キャプチャを開けませんでした。")
        return

    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = 30.0  # 録画時の目標FPS

    # 動画書き出しクラスの生成 (ファイル名, コーデック, FPS, (幅, 高さ))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = VideoWriter(output_filename, fourcc, fps, (int(wt), int(ht)))

    print("'r' キーで録画開始/停止, 'q' キーで終了します。")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if recflag:
            # フレームの書き込み
            writer.write(frame)
            # 録画中を示す赤いインジケーターを画面に描画
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recflag = not recflag
            status = "開始" if recflag else "停止"
            print(f"録画を{status}しました。")

        cv2.imshow("Screen Recorder", frame)

    writer.release()
    cv2.destroyAllWindows()
    cap.release()
    print(f"録画ファイルを保存しました: {output_filename}")

if __name__ == '__main__':
    main()

```

<hr>

