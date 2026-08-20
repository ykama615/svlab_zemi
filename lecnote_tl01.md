# PyQtGraph 高速グラフ描画ライブラリ (my_qt_graph.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、PyQtGraph のラッパーライブラリ `my_qt_graph.py`（`MyGraph` クラス）を用いて、Python から高速に 2D グラフ（静的グラフ・動的リアルタイムグラフ・分割画面レイアウト）を描画・表示する方法について解説します[cite: 10]。

## 前提条件
- デスクトップ上の `ipbl26_start` を実行して VSCode を起動します。ターミナルウィンドウに表示されるカレントディレクトリが `C:\oit\home\ipbl` であることを確認してください。
- **【重要】** `my_qt_graph.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に配置されていることを確認してください[cite: 10]。
- **【重要】** 描画バックエンドに `PyQt5` および `pyqtgraph` を使用するため、環境に正しくインストールされている必要があります[cite: 10]。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
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
    graph = MyGraph.get_instance()[cite: 10]
    graph.set_window_size(800, 500)[cite: 10]

    # 2. 描画キャンバスの追加とラベル・範囲設定
    canvas_id = graph.set_plot_canvas(title="Static Sine Wave")[cite: 10]
    graph.set_status(
        canvas_id, 
        xrange=[0, 10], 
        yrange=[-1.5, 1.5], 
        xcap=["Time", "s"], 
        ycap=["Amplitude", "V"]
    )[cite: 10]

    # 3. 描画用ペンの作成とプロット曲線の登録
    pen = graph.make_pen(color=(0, 120, 255), style=QtCore.Qt.SolidLine, width=2)[cite: 10]
    curve_id = graph.set_curve(canvasid=canvas_id, pen=pen, name="Sin Wave")[cite: 10]

    # 4. データの一括生成と設定
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    graph.set_curve_data(curve_id, x, y)[cite: 10]

    # 5. 手動更新を実行してウィンドウ表示
    graph.refresh()[cite: 10]
    graph.show()[cite: 10]

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
    graph = MyGraph.get_instance()[cite: 10]
    graph.set_window_size(800, 500)[cite: 10]

    canvas_id = graph.set_plot_canvas(title="Real-time Sensor Data")[cite: 10]
    graph.set_status(canvas_id, xrange=[0, 100], yrange=[-2.5, 2.5], xcap=["Step"], ycap=["Value"])[cite: 10]

    # 最新 200 件のみ保持する曲線
    pen = graph.make_pen(color=(255, 50, 50), style=QtCore.Qt.SolidLine, width=2)[cite: 10]
    curve_id = graph.set_curve(canvasid=canvas_id, maxdatasize=200, pen=pen, name="Signal")[cite: 10]

    step_count = 0

    # 逐次データ生成関数 (20ms周期)
    def update_data():
        nonlocal step_count
        step_count += 1
        new_x = step_count
        new_y = np.sin(step_count * 0.1) + np.random.normal(0, 0.15)

        # データ追加（内部で maxdatasize 件に削られる）
        graph.set_curve_data(curve_id, [new_x], [new_y])[cite: 10]

        # X軸を最新データに合わせてスクロール
        if step_count > 100:
            graph.set_status(canvas_id, xrange=[step_count - 100, step_count])[cite: 10]

    # データ供給用タイマー
    data_timer = QtCore.QTimer()
    data_timer.setInterval(20)
    data_timer.timeout.connect(update_data)
    data_timer.start()

    # 画面自動再描画タイマーの起動 (50ms 周期 = 20 FPS)
    graph.start_refresh(interval=50)[cite: 10]

    graph.show()[cite: 10]

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
    graph = MyGraph.get_instance()[cite: 10]
    graph.set_window_size(800, 600)[cite: 10]

    # --- 1. 上段キャンバス (row=0, col=0) ---
    canvas_top = graph.set_plot_canvas(title="Top: Sine Wave", row=0, col=0)[cite: 10]
    graph.set_status(canvas_top, xrange=[0, 10], yrange=[-1.5, 1.5], xcap=["Time", "s"], ycap=["Amp"])[cite: 10]
    
    pen_blue = graph.make_pen(color=(0, 120, 255), style=QtCore.Qt.SolidLine, width=2)[cite: 10]
    curve_top = graph.set_curve(canvasid=canvas_top, pen=pen_blue, name="Sin")[cite: 10]

    # --- 2. 下段キャンバス (row=1, col=0) ---
    canvas_bottom = graph.set_plot_canvas(title="Bottom: Cosine Wave", row=1, col=0)[cite: 10]
    graph.set_status(canvas_bottom, xrange=[0, 10], yrange=[-1.5, 1.5], xcap=["Time", "s"], ycap=["Amp"])[cite: 10]

    pen_red = graph.make_pen(color=(255, 50, 50), style=QtCore.Qt.SolidLine, width=2)[cite: 10]
    curve_bottom = graph.set_curve(canvasid=canvas_bottom, pen=pen_red, name="Cos")[cite: 10]

    # データセット
    x = np.linspace(0, 10, 300)
    graph.set_curve_data(curve_top, x, np.sin(x))[cite: 10]
    graph.set_curve_data(curve_bottom, x, np.cos(x))[cite: 10]

    graph.refresh()[cite: 10]
    graph.show()[cite: 10]

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
| `set_status(canvasid, ...)` | `xrange`, `yrange`, `xcap`, `ycap` | なし | 軸範囲（`[min, max]`）や軸ラベル（`["名前", "単位"]`）を設定 |

### データ曲線・ポイント操作

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `set_curve(canvasid, maxdatasize, pen, name)` | キャンバスID, 最大保持数, ペン設定, 凡例名 | `curve_id` | 折れ線グラフ要素をキャンバスに追加 |
| `set_curve_data(curveid, xlist, ylist)` | 曲線ID, Xリスト, Yリスト | なし | データ配列を追加（`maxdatasize` を上限に自動スライス） |
| `set_step(canvasid, ...)` | キャンバスID, 最大保持数, ペン設定, 凡例名 | `step_id` | ステップ状（階段状）グラフ要素を追加 |
| `set_point(canvasid, symbol, incolor, ...)` | キャンバスID, シンボル種類, 色, サイズ | `point_id` | 散布図プロット（点表示）要素を追加 |
| `make_pen(color, style, width)` | RGB/Color, スタイル, 太さ | `QPen` | PyQtGraph 用の描画ペンオブジェクトを生成 |

### 画面表示・更新制御

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `start_refresh(interval)` | `interval`: 更新間隔 (ms) | なし | 自動再描画タイマーを起動 |
| `refresh()` | なし | なし | キャンバス全体の手動一括再描画を実行 |
| `stop_refresh()` | なし | なし | 自動再描画タイマーを停止 |
| `show()` | なし | なし | Qt イベントループを実行して GUI を表示 |

---

## :red_square: 演習 (`qt_graph_exercise.py`)

* [`qt_split_graph.py`](https://www.google.com/search?q=%23qt_split_graphpy) をベースに、**左右 2 分割（1 行 2 列）** のウィンドウを作成し、左側にランダムノイズ信号の動的グラフ、右側にその移動平均の動的グラフを表示するプログラムを作成してください。


* **ヒントコード**:
```python
# 左右2分割 (1行2列) の設定
canvas_left  = graph.set_plot_canvas(title="Raw Signal", row=0, col=0)
canvas_right = graph.set_plot_canvas(title="Moving Average", row=0, col=1)

# それぞれのキャンバスに曲線を追加
curve_left  = graph.set_curve(canvasid=canvas_left, maxdatasize=200)
curve_right = graph.set_curve(canvasid=canvas_right, maxdatasize=200)

```



---

[トップページへ戻る]()
