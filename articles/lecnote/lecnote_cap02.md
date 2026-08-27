# Intel RealSense 画像処理ライブラリ (my_rs_cap.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、`pyrealsense2` を用いたカスタム `VideoCapture` クラスを使用して、Intel RealSense（D400シリーズ等）からのカラー画像および深度（Depth）画像を統合的に取得し、高精度なタイムスタンプ管理や `.bag` ファイルの再生・シーク処理を行う方法について解説します。

## 前提条件
- デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
- **【重要】** RealSense用ライブラリファイル（例: `my_rs_cap.py`）が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください。
- `pyrealsense2` などの依存ライブラリがインストールされた Python 環境で実行します。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

  ```

---

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

---

[トップページへ戻る]()
