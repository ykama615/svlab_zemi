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

7. 動画画像処理 (`my_cap_av2.py`)（↓）
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](lecnote_cap02.md)
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

自作ライブラリ `my_libs` 内の映像キャプチャクラス `VideoCapture` および `VideoWriter` を活用し、Webカメラ・動画ファイル・ドローンからの映像取り込みやタイムスタンプ管理、ログ付き録画を実装するための解説ドキュメントです。

<hr>

# 動画画像処理ライブラリ (my_cap_av2.py) の使い方

## 概要

- `./my_libs/video_capture/my_cap_av2.py` 内の `VideoCapture` クラスを用いて各種ソースから映像ストリームを取得します。
- OpenCV互換の操作感で、以下のキャプチャ・録画機能を処理します。
  - **3つの入力ソース（Webカメラ / 動画ファイル / Hulaドローン）の自動判別機能**
  - **PTSに基づく高精度タイムスタンプ取得とジャストシーク（`seek`）**
  - **指定秒数範囲の読み出し（`set_range`）**
  - **処理遅延ログ（`.log.csv`）を自動生成する H.264 対応 `VideoWriter`**

---

## 前提条件

- **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
  - `my_cap_av2.py`: `./my_libs/video_capture/`
- **【重要】** 読み込み・保存に使用するテスト用画像や動画ファイルは `./img/` 内に配置されている必要があります。

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

