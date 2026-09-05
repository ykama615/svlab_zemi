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

<details><summary><b>キャプチャ（3項目）</b></summary>

7. 動画画像処理 (`my_cap_av2.py`)[lecnote_cap01.md]
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](lecnote_dt04.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](lecnote_an01.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](lecnote_an04.md)
</details>

<b>➡ツール・信号処理（3項目）</b>

18. PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)（↓）
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](lecnote_tl03.md)

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>

自作ライブラリ my_libs.tools 内の高速グラフ描画クラス MyGraph (my_qt_graph.py) を活用し、2D静的グラフ・動的リアルタイムグラフの描画、複数キャンバスの画面分割レイアウト制御、および時系列データの自動リングバッファリングを実装するための解説ドキュメントです。

<hr>

# PyQtGraph 高速グラフ描画ライブラリ (my_qt_graph.py) の使い方

## 目的

* 本ドキュメントでは、PyQtGraph のラッパーライブラリ `my_qt_graph.py`（`MyGraph` クラス）を用いて、Python から高速に 2D グラフ（静的グラフ・動的リアルタイムグラフ・分割画面レイアウト）を描画・表示する方法について解説します。

## 前提条件

* デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
* **【重要】** `my_qt_graph.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください。
* **【重要】** 描画バックエンドに `PyQt5` および `pyqtgraph` を使用するため、環境に正しくインストールされている必要があります。
* ターミナルで以下のコマンドを実行してプログラムを動作させます。
```sh
C:\oit\home\ipbl> python XXX.py

```



---

## :red_square: my_qt_graph.py の概要と特徴

`MyGraph` は、Qt 系の高速描画ライブラリ `pyqtgraph` をシングルトンパターンでラップし、直感的な API（`set_plot_canvas` や `set_curve` 等）でグラフ描画を可能にするクラスです。

### 主な特徴

1. **シングルトン構造管理**:
* `MyGraph.get_instance()` で常に単一のインスタンスを取得・管理します。


2. **自動リングバッファ機能 (`maxdatasize`)**:
* `set_curve()` 等で指定した最大サイズ（既定値: 300）を超えた配列データは自動的に古いものからスライスされ、メモリ圧迫を防ぎます。


3. **タイマー駆動型フレーム更新 (`start_refresh`)**:
* `start_refresh(interval)` を実行すると内部タイマーが働き、バックグラウンドで指定ミリ秒ごとに描画キャンバスを一括更新します。


4. **柔軟なグリッドレイアウト・右 Y 軸対応**:
* `row` / `col` 指定による格子状の画面分割や、`set_right_axis_canvas()` による左右 2 軸グラフの作成に対応します。



---

## :red_square: 静的グラフの描画（一括描画）

事前に用意された全データを `set_curve_data()` で一括挿入し、`refresh()` を 1 回だけ呼んで描画する基本手法です。実験結果の分析や定型レポート出力に適しています。

### qt_static_graph.py

```python
import sys
import numpy as np
from PyQt5 import QtCore

# ライブラリのインポート
from my_libs.my_qt_graph import MyGraph

def main():
    # 1. インスタンス取得とウィンドウサイズ設定
    graph = MyGraph.get_instance()
    graph.set_window_size(800, 500)

    # 2. 描画キャンバスの追加とラベル・範囲設定
    canvas_id = graph.set_plot_canvas(title="Static Sine Wave")
    graph.set_status(
        canvas_id, 
        xrange=[0, 10], 
        yrange=[-1.5, 1.5], 
        xcap=["Time", "s"], 
        ycap=["Amplitude", "V"]
    )

    # 3. 描画用ペンの作成とプロット曲線の登録
    pen = graph.make_pen(color=(0, 120, 255), style=QtCore.Qt.SolidLine, width=2)
    curve_id = graph.set_curve(canvasid=canvas_id, pen=pen, name="Sin Wave")

    # 4. データの一括生成と設定
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    graph.set_curve_data(curve_id, x, y)

    # 5. 手動更新を実行してウィンドウ表示
    graph.refresh()
    graph.show()

if __name__ == '__main__':
    main()

```

---

## :red_square: 動的グラフの描画（リアルタイム描画）

センサー値や解析キーポイント等のストリーミングデータをリアルタイム描画する手法です。`start_refresh()` で自動再描画タイマーを有効にし、`set_curve_data()` でデータを逐次追加します。

### qt_dynamic_graph.py

```python
import numpy as np
from PyQt5 import QtCore
from my_libs.my_qt_graph import MyGraph

def main():
    graph = MyGraph.get_instance()
    graph.set_window_size(800, 500)

    canvas_id = graph.set_plot_canvas(title="Real-time Sensor Data")
    graph.set_status(canvas_id, xrange=[0, 100], yrange=[-2.5, 2.5], xcap=["Step"], ycap=["Value"])

    # 最新 200 件のみ保持する曲線
    pen = graph.make_pen(color=(255, 50, 50), style=QtCore.Qt.SolidLine, width=2)
    curve_id = graph.set_curve(canvasid=canvas_id, maxdatasize=200, pen=pen, name="Signal")

    step_count = 0

    # 逐次データ生成関数 (20ms周期)
    def update_data():
        nonlocal step_count
        step_count += 1
        new_x = step_count
        new_y = np.sin(step_count * 0.1) + np.random.normal(0, 0.15)

        # データ追加（内部で maxdatasize 件に削られる）
        graph.set_curve_data(curve_id, [new_x], [new_y])

        # X軸を最新データに合わせてスクロール
        if step_count > 100:
            graph.set_status(canvas_id, xrange=[step_count - 100, step_count])

    # データ供給用タイマー
    data_timer = QtCore.QTimer()
    data_timer.setInterval(20)
    data_timer.timeout.connect(update_data)
    data_timer.start()

    # 画面自動再描画タイマーの起動 (50ms 周期 = 20 FPS)
    graph.start_refresh(interval=50)

    graph.show()

if __name__ == '__main__':
    main()

```

---

## :red_square: 画面分割と複数ウィンドウ（レイアウト制御）

1 つのウィンドウ内を格子状に分割（`row`, `col` 指定）するか、別ウィンドウ（`figure()`）として独立させることで複雑なグラフ配置を実現します。

### qt_split_graph.py

```python
import numpy as np
from PyQt5 import QtCore
from my_libs.my_qt_graph import MyGraph

def main():
    graph = MyGraph.get_instance()
    graph.set_window_size(800, 600)

    # --- 1. 上段キャンバス (row=0, col=0) ---
    canvas_top = graph.set_plot_canvas(title="Top: Sine Wave", row=0, col=0)
    graph.set_status(canvas_top, xrange=[0, 10], yrange=[-1.5, 1.5], xcap=["Time", "s"], ycap=["Amp"])
    
    pen_blue = graph.make_pen(color=(0, 120, 255), style=QtCore.Qt.SolidLine, width=2)
    curve_top = graph.set_curve(canvasid=canvas_top, pen=pen_blue, name="Sin")

    # --- 2. 下段キャンバス (row=1, col=0) ---
    canvas_bottom = graph.set_plot_canvas(title="Bottom: Cosine Wave", row=1, col=0)
    graph.set_status(canvas_bottom, xrange=[0, 10], yrange=[-1.5, 1.5], xcap=["Time", "s"], ycap=["Amp"])

    pen_red = graph.make_pen(color=(255, 50, 50), style=QtCore.Qt.SolidLine, width=2)
    curve_bottom = graph.set_curve(canvasid=canvas_bottom, pen=pen_red, name="Cos")

    # データセット
    x = np.linspace(0, 10, 300)
    graph.set_curve_data(curve_top, x, np.sin(x))
    graph.set_curve_data(curve_bottom, x, np.cos(x))

    graph.refresh()
    graph.show()

if __name__ == '__main__':
    main()

```

---

## :red_square: MyGraph の主なメソッド一覧

### ウィンドウ・キャンバス制御

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `get_instance()` | なし | `MyGraph` | シングルトンインスタンスを取得 |
| `set_window_size(w, h)` | `w`: 幅, `h`: 高さ | なし | 現在のウィンドウサイズを設定 |
| `figure(num)` | `num`: ウィンドウ番号 | `int` | 指定番号のウィンドウを選択、`None` の場合は新規生成 |
| `set_plot_canvas(title, col, row)` | `title`: タイトル, `col`: 列, `row`: 行 | `canvas_id` | 指定位置にグラフ描画領域（キャンバス）を作成 |
| `set_right_axis_canvas(canvasid)` | `canvasid`: 対象キャンバス | `right_id` | 指定キャンバスに右 Y 軸専用の ViewBox を連動追加 |
| `set_status(canvasid, ...)` | `xrange`, `yrange`, `xcap`, `ycap` | なし | 軸範囲（`[min, max]`）や軸ラベル（`["名前", "単位"]`）、Y軸反転を一括設定 |

### データ曲線・ポイント・補助線操作

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `set_curve(canvasid, maxdatasize, pen, name)` | キャンバスID, 最大保持数, ペン設定, 凡例名 | `curve_id` | 折れ線グラフ要素をキャンバスに追加 |
| `set_curve_data(curveid, xlist, ylist)` | 曲線ID, Xリスト, Yリスト | なし | データ配列を追加（`maxdatasize` を上限に自動スライス） |
| `set_step(canvasid, ...)` | キャンバスID, 最大保持数, ペン設定, 凡例名 | `step_id` | ステップ状（階段状）グラフ要素を追加 |
| `set_step_data(curveid, xlist, ylist)` | ステップID, Xリスト, Yリスト | なし | ステップグラフ用データを追加 |
| `set_point(canvasid, symbol, incolor, ...)` | キャンバスID, シンボル種類, 色, サイズ | `point_id` | 散布図プロット（点表示）要素を追加 |
| `set_point_data(pointid, xlist, ylist)` | ポイントID, Xリスト, Yリスト | なし | 散布図用データを追加 |
| `set_trg_line(canvasid, posX, pen)` | キャンバスID, X位置, ペン | なし | 指定 X 座標に垂直なトリガーライン（無限直線）を追加 |
| `set_horizontal_line(canvasid, posY, pen)` | キャンバスID, Y位置, ペン | なし | 指定 Y 座標に水平な基準ライン（無限直線）を追加 |
| `set_gridline(canvasid, space_x, space_y)` | キャンバスID, X間隔, Y間隔 | なし | 指定キャンバスに破線のグリッド線を表示 |
| `make_pen(color, style, width)` | RGB/Color, スタイル, 太さ | `QPen` | PyQtGraph 用の描画ペンオブジェクトを生成 |

### 画面表示・更新・保存制御

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `start_refresh(interval)` | `interval`: 更新間隔 (ms) | なし | 自動再描画タイマーを起動 |
| `refresh()` | なし | なし | キャンバス全体の手動一括再描画を実行 |
| `stop_refresh()` | なし | なし | 自動再描画タイマーを停止 |
| `destroy_refresh()` | なし | なし | 自動再描画タイマーを破棄（メモリ解放） |
| `show()` | なし | なし | Qt イベントループを実行して GUI を表示 |
| `save_plot_window(filename, num)` | `filename`: ファイル名, `num`: ウィンドウ番号 | なし | 指定ウィンドウのプロット画面をPNG画像として保存 |
| `destroy_window(num)` | `num`: ウィンドウ番号 | なし | 指定ウィンドウを破棄 |

---

## :red_square: 演習 (`qt_graph_exercise.py`)

`qt_split_graph.py` をベースに、**左右 2 分割（1 行 2 列）** のウィンドウを作成し、左側にランダムノイズ信号の動的グラフ、右側にその移動平均の動的グラフを表示するプログラムを作成してください。

### 解答サンプルコード (`qt_graph_exercise.py`)

```python
import numpy as np
from PyQt5 import QtCore
from my_libs.my_qt_graph import MyGraph

def main():
    # 1. シングルトンインスタンスの取得とウィンドウサイズ設定
    graph = MyGraph.get_instance()
    graph.set_window_size(1000, 500)

    # 2. 左右2分割（1行2列）のキャンバスを作成
    canvas_left  = graph.set_plot_canvas(title="Raw Signal (Random Noise)", row=0, col=0)
    canvas_right = graph.set_plot_canvas(title="Moving Average Signal", row=0, col=1)

    # 軸の範囲やラベルを設定
    graph.set_status(canvas_left, xrange=[0, 100], yrange=[-3, 3], xcap=["Step"], ycap=["Value"])
    graph.set_status(canvas_right, xrange=[0, 100], yrange=[-3, 3], xcap=["Step"], ycap=["Average"])

    # 3. 描画用ペンの作成と曲線の登録（最大データ保持数は200）
    pen_left  = graph.make_pen(color=(100, 100, 255), style=QtCore.Qt.SolidLine, width=2)
    pen_right = graph.make_pen(color=(255, 100, 100), style=QtCore.Qt.SolidLine, width=2)
    
    curve_left  = graph.set_curve(canvasid=canvas_left, maxdatasize=200, pen=pen_left, name="Raw")
    curve_right = graph.set_curve(canvasid=canvas_right, maxdatasize=200, pen=pen_right, name="Moving Avg")

    # データの蓄積用リスト（移動平均を計算するために過去の値を保持する）
    raw_data_buffer = []
    step_count = 0

    # 4. 定期的に呼ばれるデータ更新関数
    def update_data():
        nonlocal step_count
        step_count += 1
        
        # 新しいランダムノイズの生成
        new_y = np.random.normal(0, 1.0)
        raw_data_buffer.append(new_y)
        
        # 移動平均の計算（直近10個の平均）
        window_size = 10
        recent_data = raw_data_buffer[-window_size:]
        moving_avg = np.mean(recent_data)

        # 左右それぞれの曲線にデータを追加
        graph.set_curve_data(curve_left, [step_count], [new_y])
        graph.set_curve_data(curve_right, [step_count], [moving_avg])

        # X軸を最新データに合わせてスクロールさせる
        if step_count > 100:
            graph.set_status(canvas_left, xrange=[step_count - 100, step_count])
            graph.set_status(canvas_right, xrange=[step_count - 100, step_count])

    # 5. データ供給用タイマーの設定 (20ms周期)
    data_timer = QtCore.QTimer()
    data_timer.setInterval(20)
    data_timer.timeout.connect(update_data)
    data_timer.start()

    # 6. 画面の自動再描画タイマーの起動 (50ms 周期)
    graph.start_refresh(interval=50)

    # 7. 画面を表示
    graph.show()

if __name__ == '__main__':
    main()

```
