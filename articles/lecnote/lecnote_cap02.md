<hr>

**講義ノート・ライブラリ一覧**

<details><summary><b>基礎編（6項目）</b></summary>

1. [環境の設定](../../README.md)
2. [基本概要](../basic/BASIC_00.md)
3. [カメラへのアクセスと動画処理](../basic/BASIC_01.md)
4. [顔と顔パーツの検出](../basic/BASIC_02.md)
5. [顔・手・ポーズ検出](../basic/BASIC_03.md)
6. 2つのベクトルのなす角とベクトル演算（↓）
</details>

<b>➡キャプチャ（3項目）</b>

7. [動画画像処理 (`my_cap_av2.py`)](lecnote_cap01.md)
8. Intel RealSense 画像処理 (`my_rs_cap.py`)（↓）
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](lecnote_dt04.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](lecnote_an04.md)
</details>

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>

自作ライブラリ `my_libs` 内の Intel RealSense 用キャプチャクラス `VideoCapture` を活用し、カラー画像・深度（Depth）画像の一括取得、フィルタリング処理、および `.bag` ファイルの再生・シーク機能を実装するための解説ドキュメントです。

<hr>

# Intel RealSense 画像処理ライブラリ (my_rs_cap.py) の使い方

## 概要

- `./my_libs/video_capture/my_rs_cap.py` 内の `VideoCapture` クラスを用いて Intel RealSense（D400シリーズ等）または `.bag` ファイルから映像ストリームを取得します。
- `pyrealsense2` をベースに、以下の RGB-D キャプチャ機能を統合的に処理します。
  - **カラー画像（BGR）、生深度値（Depth）、可視化用カラーマップ（Colormap）の一括取得**
  - **3つの接続モード（標準カメラ / シリアル番号指定 / `.bag` ファイル再生）**
  - **リアルタイム深度フィルタリング（ノイズ低減・穴埋め処理）**
  - **`.bag` 再生時の高精度タイムスタンプ管理と任意位置へのシーク（`seek`）**

---

## 前提条件

- **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
  - `my_rs_cap.py`: `./my_libs/video_capture/`
- **【重要】** 動作には `pyrealsense2` がインストールされた環境が必要です。
- **【重要】** 再生テストに使用する `.bag` ファイルや保存画像は `./img/` 内に配置されている必要があります。

---

```

## :red_square: my_rs_cap.py (RealSense VideoCapture) の概要と特徴

提供された `VideoCapture` クラスは、OpenCV互換の操作感（`read()`, `get()`, `isOpened()`, `release()` など）を保ちながら、Intel RealSense の強力な RGB-D キャプチャ機能および `.bag` 再生機能を簡潔に扱えるように設計されています。

### 主な特徴

1. **RGB + 深度（Depth）データの一括取得**:
* `cap.read()` を呼び出すだけで、カラー画像（`bgr`）、生の深度値（`depth`）、および可視化用のカラーマップ（`colormap`）の 3 つのデータを辞書形式で同時に取得できます。


2. **3つの接続モード（ライブカメラ・シリアル指定・bag再生）**:
* **標準カメラ接続**: `source=0`（デフォルト）等で接続された RealSense を使用します。
* **シリアル番号指定**: デバイスのシリアル番号（文字列）を指定して特定のカメラをオープンします。
* **`.bag` ファイル再生**: 拡張子が `.bag` のファイルパスを指定すると、過去に録画した RealSense データを再生できます。


3. **高精度タイムスタンプと相対時間化**:
* RealSense 固有のエポックタイムスタンプ（大きな数値）をキャプチャ開始時からの相対ミリ秒（`0.0 ms` 起点）に自動変換し、`cap.get(cv2.CAP_PROP_POS_MSEC)` で取得できます。


4. **リアルタイム深度フィルタリング内蔵**:
* 内部で `temporal_filter`（時系列フィルタ）および `hole_filling_filter`（穴埋めフィルタ）を適用し、ノイズの少ない高品質な深度マップを生成します。


5. **.bag 再生時の時間制御とシーク機能**:
* `.bag` ファイル再生時には非リアルタイム（コマ落ちなし）モードで動作させることができ、`seek(timestamp_ms)` メソッドによる任意のタイムスタンプ（ミリ秒）への移動や `get_total_duration()` による総再生時間の取得が可能です。



---

## :red_square: 基本サンプルコード

### rs_viewer1.py

```python
import cv2
# my_libs/my_rs_cap.py から VideoCapture をインポート
from my_libs.my_rs_cap import VideoCapture

# 0 (デフォルトカメラ)、シリアル番号文字列、または ".bag" ファイルのパスを指定
source = 0 

# main----------------------------------------------------
def main():
    global source

    # RealSense用 VideoCapture インスタンスの生成
    cap = VideoCapture(source=source, width=1280, height=720, fps=30)

    if not cap.isOpened():
        print("エラー: RealSense デバイスまたは .bag ファイルを開けませんでした。")
        return

    # プロパティの取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print(f"Size: {ht} x {wt} / FPS: {fps}")

    while cap.isOpened():
        # 1フレーム分のデータ群を取得
        ret, frames = cap.read(holeFilter=True)
        if not ret:
            print("フレームの取得に失敗したか、.bag ファイルの終端に達しました。")
            break

        # 各種フレームデータの取り出し
        color_img    = frames['bgr']       # カラー画像 (BGR)
        depth_img    = frames['depth']     # 生の深度値配列 (uint16)
        colormap_img = frames['colormap']  # 可視化用カラーマップ画像 (BGR)

        # 現在の経過時間（ミリ秒）を取得
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        # 画面表示
        cv2.imshow("RealSense Color", color_img)
        cv2.imshow("RealSense Depth Colormap", colormap_img)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()

```

### VideoCapture メソッドとプロパティ

| コード | 内容・説明 |
| --- | --- |
| `VideoCapture(source, width, height, fps)` | ストリームまたは `.bag` ファイルを開く。アライメント処理や深度フィルタの初期化も実行 |
| `cap.read(holeFilter=True)` | 成功フラグ (`bool`) と、画像辞書 `{'bgr', 'depth', 'colormap'}` を返却 |
| `cap.get(cv2.CAP_PROP_POS_MSEC)` | キャプチャ開始からの経過時間（ミリ秒） |
| `cap.get(cv2.CAP_PROP_POS_FRAMES)` | 現在のフレーム番号 |
| `cap.get(cv2.CAP_PROP_FPS)` | フレームレート |
| `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | フレーム横幅 |
| `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` | フレーム縦幅 |
| `cap.seek(timestamp_ms)` | `.bag` ファイル読み込み時、指定したミリ秒の位置へシークする |
| `cap.get_total_duration()` | 動画（.bag）の総再生時間（ミリ秒）を取得。ライブ時は現在の経過時間を返す |
| `cap.get_device()` | 内部の `pyrealsense2.device` オブジェクトを取得する |

### :o: 練習

* 上記の [`rs_viewer1.py`](https://www.google.com/search?q=%23rs_viewer1py) のソースコードを VS Code にコピー＆ペーストし、`C:\oit\home\ipbl\rs_viewer1.py` として保存します。
* RealSense カメラを PC に接続し、プログラムを実行してカラー画面と深度カラーマップ画面の両方が表示されることを確認してください。

---

## :red_square: 演習 (`rs_selfie.py`)

* [`rs_viewer1.py`](https://www.google.com/search?q=%23rs_viewer1py) を元にして、特定のキーを押した際にカラー画像と深度カラーマップ画像の両方を保存する `rs_selfie.py` を作成してください。

| キー | 動作内容 |
| --- | --- |
| **q** | プログラムを終了 |
| **s** | カラー画像を `./img/rs_color.jpg`、深度カラーマップを `./img/rs_depth.jpg` として保存 |

* **ヒントコード**:
```python
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
    break
elif key == ord('s'):
    cv2.imwrite("./img/rs_color.jpg", frames['bgr'])
    cv2.imwrite("./img/rs_depth.jpg", frames['colormap'])
    print("Saved RealSense Color and Depth images!")

```



---

## :red_square: `.bag` ファイル再生とシーク機能の使い方

RealSense Viewer等で録画した `.bag` ファイルを読み込んで解析・再生する場合の例です。

### .bag 再生と任意位置へのシークサンプル

```python
import cv2
from my_libs.my_rs_cap import VideoCapture

# .bag ファイルのパスを指定
bag_file = "./img/sample_recording.bag"

def main():
    cap = VideoCapture(source=bag_file)

    if not cap.isOpened():
        print(".bag ファイルが見つかりません。")
        return

    # 総再生時間の取得
    total_ms = cap.get_total_duration()
    print(f"Total Duration: {total_ms / 1000.0:.2f} seconds")

    # 5秒 (5000ms) の位置へシーク
    cap.seek(5000.0)

    while cap.isOpened():
        ret, frames = cap.read()
        if not ret:
            break

        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.putText(frames['bgr'], f"Time: {current_ms/1000.0:.2f} s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("BAG Playback", frames['bgr'])
        cv2.imshow("BAG Depth", frames['colormap'])

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```
