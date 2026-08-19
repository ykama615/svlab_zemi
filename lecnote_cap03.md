# Orbbec Femto Bolt 画像処理ライブラリ (my_bolt_cap.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、`pyorbbecsdk` を使用したカスタム `VideoCapture` クラスを用いて、Orbbec Femto Bolt などの 3D カメラからカラー画像および深度（Depth）画像を統合的に取得し、リアルタイム処理や `.mkv` ファイルの再生・シーク処理を行う方法について解説します。

## 前提条件
- デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
- **【重要】** Orbbec用ライブラリファイル（例: `my_bolt_cap.py`）が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください。
- `pyorbbecsdk` などの依存ライブラリがインストールされた Python 環境で実行します。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

  ```

---

## :red_square: my_bolt_cap.py (Femto Bolt VideoCapture) の概要と特徴

提供された `VideoCapture` クラスは、OpenCV互換の操作感（`read()`, `get()`, `isOpened()`, `release()` など）を維持しつつ、Orbbec SDK v2 (pyorbbecsdk) の RGB-D キャプチャ機能や録画ファイル再生機能を簡潔に扱えるように設計されています。

### 主な特徴

1. **RGB + 深度（Depth）データの一括取得**:
* `cap.read()` を呼び出すだけで、解像度が合わせ込まれたカラー画像（`bgr`）、生の深度値（`depth`）、および可視化用のカラーマップ（`colormap`）の 3 種類のデータを辞書形式で同時に取得できます。


2. **リアルタイムキャプチャ時のバッファフラッシュ（遅延防止）**:
* ライブカメラモード実行時、処理遅延によって内部バッファに溜まった古いフレームを自動的に掃き出し（ドロップし）、常に最新のフレームを優先して返却することでリアルタイム性を確保します。


3. **.mkv ファイル再生・シーク機能**:
* `source` に `.mkv` ファイルのパスを指定すると録画データの再生モードとなり、`seek(timestamp_ms)` による任意のタイムスタンプ（ミリ秒）へのシークや `get_total_duration()` による総再生時間の取得が可能です。


4. **柔軟な初期化引数と解像度補正**:
* `mode` や `width`, `height` 引数のさまざまな指定パターン（タプル、解像度定数 `HD`, `FHD` など）に対応する引数補正ロジックを内蔵しています。



---

## :red_square: 基本サンプルコード

### bolt_viewer1.py

```python
import cv2
# my_libs/my_bolt_cap.py から VideoCapture をインポート
from my_libs.my_bolt_cap import VideoCapture

# 0 (デフォルトカメラ) または ".mkv" ファイルのパスを指定
source = 0 

# main----------------------------------------------------
def main():
    global source

    # Femto Bolt 用 VideoCapture インスタンスの生成 (デフォルト HD: 1280x720)
    cap = VideoCapture(source=source, mode=VideoCapture.HD, fps=30)

    if not cap.isOpened():
        print("エラー: Femto Bolt デバイスまたは .mkv ファイルを開けませんでした。")
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
            print("カメラが切断されたか、.mkv ファイルの終端に達しました。")
            break

        # フレームが空の場合はスキップ
        if frames is None:
            continue

        # 各種フレームデータの取り出し
        color_img    = frames['bgr']       # カラー画像 (BGR)
        depth_img    = frames['depth']     # 生の深度値配列 (uint16)
        colormap_img = frames['colormap']  # 可視化用カラーマップ画像 (BGR)

        # 現在の経過時間（ミリ秒）を取得
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        # 画面表示
        cv2.imshow("Femto Bolt Color", color_img)
        cv2.imshow("Femto Bolt Depth Colormap", colormap_img)

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
| `VideoCapture(source, mode, fps, width, height)` | ストリームまたは `.mkv` ファイルを開く。プリセット定数 (`NHD`, `HD`, `FHD`) を使用可能 |
| `cap.read(holeFilter=False)` | 成功フラグ (`bool`) と、画像辞書 `{'bgr', 'depth', 'colormap'}` を返却 |
| `cap.get(cv2.CAP_PROP_POS_MSEC)` | キャプチャ開始からの経過時間（ミリ秒） |
| `cap.get(cv2.CAP_PROP_POS_FRAMES)` | 現在のフレーム番号 |
| `cap.get(cv2.CAP_PROP_FPS)` | フレームレート |
| `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | フレーム横幅 |
| `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` | フレーム縦幅 |
| `cap.seek(timestamp_ms)` | `.mkv` ファイル読み込み時、指定したミリ秒の位置へシークする |
| `cap.get_total_duration()` | 動画（.mkv）の総再生時間（ミリ秒）を取得。ライブ時は現在の経過時間を返す |

### :o: 練習

* 上記の [`bolt_viewer1.py`](https://www.google.com/search?q=%23bolt_viewer1py) のソースコードを VS Code にコピー＆ペーストし、`C:\oit\home\ipbl\bolt_viewer1.py` として保存します。
* Femto Bolt カメラを PC に接続し、プログラムを実行してカラー画面と深度カラーマップ画面の両方が表示されることを確認してください。

---

## :red_square: 演習 (`bolt_selfie.py`)

* [`bolt_viewer1.py`](https://www.google.com/search?q=%23bolt_viewer1py) を元にして、特定のキーを押した際にカラー画像と深度カラーマップ画像の両方を保存する `bolt_selfie.py` を作成してください。

| キー | 動作内容 |
| --- | --- |
| **q** | プログラムを終了 |
| **s** | カラー画像を `./img/bolt_color.jpg`、深度カラーマップを `./img/bolt_depth.jpg` として保存 |

* **ヒントコード**:
```python
key = cv2.waitKey(1) & 0xFF
if key == ord('q'):
    break
elif key == ord('s'):
    if frames is not None:
        cv2.imwrite("./img/bolt_color.jpg", frames['bgr'])
        cv2.imwrite("./img/bolt_depth.jpg", frames['colormap'])
        print("Saved Femto Bolt Color and Depth images!")

```



---

## :red_square: `.mkv` ファイル再生とシーク機能の使い方

Orbbec SDK や Orbbec Viewer 等で録画した `.mkv` ファイルを読み込んで再生・解析する場合の例です。

### .mkv 再生と任意位置へのシークサンプル

```python
import cv2
from my_libs.my_bolt_cap import VideoCapture

# .mkv ファイルのパスを指定
mkv_file = "./img/sample_recording.mkv"

def main():
    cap = VideoCapture(source=mkv_file)

    if not cap.isOpened():
        print(".mkv ファイルが見つかりません。")
        return

    # 総再生時間の取得
    total_ms = cap.get_total_duration()
    print(f"Total Duration: {total_ms / 1000.0:.2f} seconds")

    # 3秒 (3000ms) の位置へシーク
    cap.seek(3000.0)

    while cap.isOpened():
        ret, frames = cap.read()
        if not ret:
            break

        if frames is None:
            continue

        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        cv2.putText(frames['bgr'], f"Time: {current_ms/1000.0:.2f} s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("MKV Playback Color", frames['bgr'])
        cv2.imshow("MKV Playback Depth", frames['colormap'])

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

```

```
