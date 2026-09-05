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
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](lecnote_cap02.md)
9. Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)（↓）
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

自作ライブラリ `my_libs` 内の Orbbec 用キャプチャクラス `VideoCapture` を活用し、カラー画像・深度（Depth）画像の一括取得、バッファ遅延防止処理、および `.mkv` ファイルの再生・シーク機能を実装するための解説ドキュメントです。

<hr>

# Orbbec Femto Bolt 画像処理ライブラリ (my_bolt_cap.py) の使い方

## 概要

* `./my_libs/video_capture/my_bolt_cap.py` 内の `VideoCapture` クラスを用いて Orbbec Femto Bolt などの 3D カメラまたは `.mkv` ファイルから映像ストリームを取得します。
* `pyorbbecsdk` をベースに、以下の RGB-D キャプチャ機能を統合的に処理します。
* **カラー画像（BGR）、生深度値（Depth）、可視化用カラーマップ（Colormap）の一括取得**
* **ライブキャプチャ時のバッファ自動フラッシュ（処理遅延・タイムラグ防止）**
* **解像度補正ロジックおよびプリセット指定（`HD`, `FHD` など）**
* **`.mkv` 再生時の高精度タイムスタンプ管理と任意位置へのシーク（`seek`）**
* **シークバーによる `.mkv` 再生位置の任意移動・連動機能**



---

## 前提条件

* **【重要】** ライブラリ用スクリプトが以下の相対パス配下に配置されていることを確認してください。
* `my_bolt_cap.py`: `./my_libs/video_capture/`


* **【重要】** 動作には `pyorbbecsdk` がインストールされた環境が必要です。
* **【重要】** 再生テストに使用する `.mkv` ファイルや保存画像は `./img/` 内に配置されている必要があります。

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

5. **シークバーによる再生位置コントロール**:

* `.mkv` ファイル再生時に OpenCV ウィンドウへシークバーを設置し、現在の再生フレームと連動させながら自由に早送りや巻き戻しが行えます。

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

* 上記の `bolt_viewer1.py` のソースコードを VS Code にコピー＆ペーストし、作業ディレクトリに保存します。
* Femto Bolt カメラを PC に接続し、プログラムを実行してカラー画面と深度カラーマップ画面の両方が表示されることを確認してください。

---

## :red_square: 演習 (`bolt_selfie.py`)

* `bolt_viewer1.py` を元にして、特定のキーを押した際にカラー画像と深度カラーマップ画像の両方を保存する `bolt_selfie.py` を作成してください。

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

## :red_square: `.mkv` ファイル再生とシークバー機能 (`seek`, `init_seekbar`)

Orbbec SDK や Orbbec Viewer 等で録画した `.mkv` ファイルを読み込んで再生・解析する場合の例です。シークバー（トラックバー）を設置して、再生位置の自由な移動や現在位置への連動を行うことができます。

### .mkv 再生とシークバー連動サンプル

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

    winname = "MKV Playback with Seekbar"
    cv2.namedWindow(winname)
    
    # ウィンドウにシークバー（トラックバー）を設置
    cap.init_seekbar(winname)

    while cap.isOpened():
        ret, frames = cap.read()
        if not ret:
            print(".mkv ファイルの終端に達しました。")
            break

        if frames is None:
            continue

        # ラップされた imshow を使って表示とシークバーの位置連動を行う
        cap.imshow(winname, frames)

        # 'q' キーで終了
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```
