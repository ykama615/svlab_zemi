# 動画画像処理ライブラリ (my_cap_av2.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、`my_cap_av2.py` ライブラリを使用して、Webカメラ、動画ファイル、およびHulaドローンからの映像入力を統合的に処理し、高精度なタイムスタンプ管理や動画録画・ログ出力を行う方法を解説します。

## 前提条件
- デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
- **【重要】** `my_cap_av2.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください。
- 作成するPythonプログラム（`.py`）は `C:\oit\home\ipbl` に保存します。読み込み・保存する画像ファイル等は `C:\oit\home\ipbl\img` 内に配置します。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

```

---

## :red_square: my_cap_av2.py の概要と特徴

`my_cap_av2.py` は、OpenCVの `cv2.VideoCapture` と高い互換性を持ちつつ、用途に応じた3つの動作モード（Webカメラ・動画ファイル・Hulaドローン）を自動切り替えして映像フレームを取得するカスタムライブラリです。

### 主な特徴

1. **OpenCV互換のインターフェース**:
* `cap = VideoCapture(...)` や `cap.read()`, `cap.get(...)`, `cap.release()` などの OpenCV 標準と同等のメソッドを提供します。


2. **3つの入力モードを自動判別**:
* **Webカメラモード (`int`)**: 数値（`0` など）を渡すと標準Webカメラを使用します。
* **動画ファイルモード (`str`)**: 動画ファイルのパス文字列を渡すと PyAV (`av`) を使用してデコードし、PTS (Presentation Time Stamp) に基づく正確なタイムスタンプ管理を行います。
* **Hulaドローンモード (`object`)**: `get_image_array` メソッドを持つ SDK オブジェクトを渡すと、RTPストリーム制御や重複フレームのドロップ処理を行います。


3. **高精度なシーク機能と範囲読み出し**:
* ファイルモードにおいて、キーフレームシークと空読み（ロールフォワード）を組み合わせた正確な位置合わせ (`seek`) や、指定した秒数範囲のみを切り出す (`set_range`) 機能を提供します。


4. **CSVログ出力対応の VideoWriter**:
* PyAV を使用した H.264 エンコードに対応し、録画時の理論時間・実経過時間・処理遅延を記録する `.log.csv` を自動生成します。



---

## :red_square: 高精度動画画像処理のサンプル

### video_viewer1.py

```python
import os
# OpenCVのキャプチャ遅延を防ぐ設定
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
# my_libs/my_cap_av2.py から VideoCapture をインポート
from my_libs.my_cap_av2 import VideoCapture

device = 0 # カメラのデバイス番号（動画ファイル名やHulaドローンオブジェクトも指定可能）

# main----------------------------------------------------
def main():
    global device

    # OpenCVの cv2.VideoCapture の代わりに my_cap_av2 の VideoCapture を使用
    cap = VideoCapture(device)
    
    # プロパティの取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 現在のタイムスタンプ位置（ミリ秒）を取得
        current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow("video", frame)

    cv2.destroyAllWindows()
    cap.release()

# run-----------------------------------------------------
if __name__ == '__main__':
    main()

```

### VideoCapture メソッドとプロパティ

* `device` 変数には以下の値を渡すことができます。
* **整数（例: `0`）**: 標準Webカメラ
* **文字列（例: `"./img/movie.mp4"`）**: 動画ファイル
* **Hula SDKインスタンス**: Hulaドローンのカメラ入力



| コード | 内容・説明 |
| --- | --- |
| `VideoCapture(source)` | 入力ソースに応じて内部モード（`camera`, `file`, `hula`）を初期化してストリームを開く |
| `cap.read()` | 1フレームを取得。成功フラグ (`bool`) と BGR形式の画像データ (`ndarray`) を返却 |
| `cap.get(cv2.CAP_PROP_POS_MSEC)` | 経過時間または再生位置（ミリ秒） |
| `cap.get(cv2.CAP_PROP_FPS)` | フレームレート (FPS) |
| `cap.get(cv2.CAP_PROP_FRAME_WIDTH)` | フレームの横幅 |
| `cap.get(cv2.CAP_PROP_FRAME_HEIGHT)` | フレームの縦幅 |
| `cap.get(cv2.CAP_PROP_POS_FRAMES)` | 現在のフレーム番号 |

### :o: 練習

* [`video_viewer1.py`](https://www.google.com/search?q=%23video_viewer1py) のソースコードを VS Code にコピー＆ペーストし、`C:\oit\home\ipbl\video_viewer1.py` として保存します。
* プログラムを実行し、カメラ映像が問題なく表示され、`q` キーで終了することを確認してください。

---

## :red_square: 演習 (`selfie.py`)

* [`video_viewer1.py`](https://www.google.com/search?q=%23video_viewer1py) を元にして、特定のキーを押した際に静止画を保存する `selfie.py` を作成してください。

| キー | 動作内容 |
| --- | --- |
| **q** | プログラムを終了 |
| **s** | 現在の表示フレームを `./img/selfie.jpg` として保存 |

* **ヒントコード**:
```python
key = cv2.waitKey(1)
if key & 0xFF == ord('q'):
    break
elif key & 0xFF == ord('s'):
    cv2.imwrite("./img/selfie.jpg", frame)
    print("Saved selfie.jpg")

```



---

## :red_square: 動画ファイルモード専用機能 (`seek`, `set_range`)

動画ファイル入力時（`mode == "file"`）に利用できる追加機能です。

### 1. 指定秒数へのジャストシーク (`seek`)

直前のキーフレームへ移動した後、目的の時間まで内部でフレームを空読み（ロールフォワード）することで正確な位置へシークします。

```python
# 15.5秒の位置に移動
cap.seek(15.5)

```

### 2. 指定範囲の読み出し (`set_range`)

指定した開始時間（秒）から指定範囲（秒間）のみを読み出す制限を設定します。

```python
# 10.0秒の位置から 5.0秒間 のみ読み出し対象とする
cap.set_range(start_sec=10.0, duration_sec=5.0)

```

---

## :red_square: 付録: 高精度ログ出力機能付き VideoWriter

`my_cap_av2.py` の `VideoWriter` は、H.264 エンコード（`libx264`）での録画と同時に、フレームごとの理論時間・実経過時間・遅延ミリ秒を書き出す `.log.csv` ファイルを生成します。

### video_recorder.py

```python
import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
from my_libs.my_cap_av2 import VideoCapture, VideoWriter

device = 0
video_name = "record.mp4"

def main():
    global device, video_name
    recflag = False

    cap = VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    # 動画書き出しクラスの生成 (ファイル名, FPS, (幅, 高さ), is_vfr)
    writer = VideoWriter(video_name, fps, (int(wt), int(ht)), is_vfr=False)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if recflag:
            # フレームの書き込み (log.csv に自動でタイムスタンプが記録される)
            writer.write(frame)
            # 録画中マークの表示
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            recflag = not recflag # 録画トグル

        cv2.imshow("video", frame)

    writer.release()
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()

```

* `writer.release()` が呼び出されると、保存した動画（例: `record.mp4`）と同じディレクトリにログファイル（例: `record.log.csv`）が作成されます。

---

[トップページへ戻る]()
