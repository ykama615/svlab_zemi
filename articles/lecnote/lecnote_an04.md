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

<b>➡生体・動作解析（4項目）</b>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](lecnote_an03.md)
17. 非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)（↓）
    
<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>
自作ライブラリ `my_libs.analysis` 内の非接触脈波解析クラス `MyAnalysisrPPG` を活用し、顔ランドマークから両頬の関心領域（ROI）の抽出、平均 RGB 値の算出、および POS 法を用いた非接触脈波（rPPG）信号の推定を実装するための解説ドキュメントです。
<hr>

# 非接触脈波・rPPG（POS法）信号抽出ライブラリ (my_analysis_rppg.py) の使い方

## 目的

* 本ドキュメントでは、顔ランドマークから皮膚（両頬）の関心領域（ROI）を特定して RGB 信号を抽出するとともに、POS（Plane-Orthogonal-to-Skin）アルゴリズムを用いて皮膚の色調変化から脈波信号を推定する `MyAnalysisrPPG` クラスの動作原理と利用方法について解説します。

## 前提条件

* **【重要】** `my_analysis_rppg.py` がライブラリフォルダー（例: `my_libs/analysis/`）に配置されていることを確認してください。
* **【重要】** 内部処理に `numpy` を使用します。
* ターミナルで以下のコマンドを実行してプログラムを動作させます。
```sh
C:\oit\home\ipbl> python XXX.py

```



---

## :red_square: 幾何学的アルゴリズムと動作原理

`MyAnalysisrPPG` は、顔のランドマーク座標（68 点データ等）から鼻や輪郭の影響を受けにくい両頬の平地領域を抽出し、その平均 RGB 時間変化から皮膚表面の血流変化に由来する脈波（rPPG）信号を算出するモジュールです。

```
[ カメラ画像 ] + [ 68点 顔ランドマーク ]
        │
        ├─► 1. 頬 ROI 領域算出 (extract_cheek_rois) ──► 目尻・小鼻座標からジッターフリーな両頬 ROI を切り出し
        │                                                   │
        ├─► 2. 平均 RGB 抽出 (extract_roi_by_rect) ────────┼ 画像枠内クリップ処理 + BGR->RGB 変換 + 平均値算出
        │                                                   │
        └─► 3. POS 脈波演算 (compute_pos_signal) ◄────────┘ RGB 時系列から直交射影・標準偏差比により脈波信号を再構築

```

### 1. ジッター除去を伴う両頬 ROI の自動切り出し

* 顔ランドマークの「小鼻」と「目尻」の中間点を中心として、輪郭や鼻の影を回避する安全圏に左右の頬 ROI を配置します。
* ランドマーク座標の微小な揺れによるノイズ（自己生成ジッター）を遮断するため、ROI の一辺サイズを 4 ピクセル単位（`max(16, ...)`）で丸めて固定切り出しを行います。

### 2. クリップ処理と平均 RGB 値の変換

* 切り出し範囲がカメラ画像の画角外へ食み出ないよう、座標を `[0, 幅/高さ]` の範囲に制約（クリップ）します。
* OpenCV の標準的な BGR 配列を RGB 配列に並べ替え（`::-1`）、ROI 内の全ピクセルの色平均（`np.mean`）を取得します。

### 3. POS（Plane-Orthogonal-to-Skin）アルゴリズム

* 取得した RGB 時系列信号 $C$ を平均値で正規化（$C_n$）した後、分光特性に基づく 2 つの直交色空間成分 $X, Y$ を定義します。

$$\begin{aligned} X &= G_n - B_n \\ Y &= -2R_n + G_n + B_n \end{aligned}$$

* 色調の標準偏差比 $\alpha = \frac{\sigma(X)}{\sigma(Y)}$ を重みとして合成（$S = X + \alpha Y$）することで、体動や照明変化の影響を無効化したクリアな脈波時系列信号を生成します。

---

## :red_square: 基本的な使い方とサンプルコード

### 1. 顔ランドマークからの両頬 RGB 取得および POS 脈波信号の計算 (`rppg_analysis_sample.py`)

```python
import numpy as np
from my_libs.analysis.my_analysis_rppg import MyAnalysisrPPG

def main():
    # 30 fps 設定でインスタンス化
    rppg_analyzer = MyAnalysisrPPG(fps=30.0)

    # ダミーのカメラフレーム (480x640, BGR 3チャンネル)
    dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # 68点顔ランドマークのダミーデータ（配列インデックスに対応する [x, y]）
    dummy_landmarks = {i: (300 + (i % 5) * 10, 200 + (i // 5) * 10) for i in range(68)}
    # 主要な参照点をダミー調整（目・小鼻付近）
    dummy_landmarks[31] = (290, 220)  # 右小鼻
    dummy_landmarks[35] = (330, 220)  # 左小鼻
    dummy_landmarks[36] = (250, 180)  # 右目尻
    dummy_landmarks[41] = (280, 190)  # 右目下
    dummy_landmarks[45] = (370, 180)  # 左目尻
    dummy_landmarks[46] = (340, 190)  # 左目下

    # 1. 両頬の ROI 矩形および平均 RGB 値の取得
    cheeks = rppg_analyzer.extract_cheek_rois(dummy_frame, dummy_landmarks)
    print(f"右頬 RGB: {cheeks['right_rgb']}, 矩形: {cheeks['right_rect']}")
    print(f"左頬 RGB: {cheeks['left_rgb']}, 矩形: {cheeks['left_rect']}")

    # 2. 時系列 RGB データからの POS 脈波信号の抽出（例: 60フレーム分のデータ）
    dummy_rgb_history = np.random.normal(loc=120.0, scale=2.0, size=(60, 3))

    # 脈波時系列全体を計算
    pos_signal = rppg_analyzer.compute_pos_signal(dummy_rgb_history)
    # 最新の生値 1 点を取得
    latest_pos = rppg_analyzer.get_latest_pos_value(dummy_rgb_history)

    print(f"算出された脈波データ長: {len(pos_signal)}")
    print(f"最新の脈波値: {latest_pos:.4f}")

if __name__ == '__main__':
    main()

```

---

## :red_square: API リファレンス

### MyAnalysisrPPG クラス

| メソッド | 主要引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__(fps=30.0)` | `fps`: フレームレート | なし | 1フレームあたりのミリ秒時間幅（`dt_msec`）を初期化。 |
| `extract_roi_by_rect(frame, x, y, w, h)` | `frame`: 画像フレーム<br><br><br>`x, y, w, h`: 切り出し領域 | `tuple` | 指定範囲を画角内に安全クリップし、平均 RGB 値（`rgb_mean`）と補正後の有効矩形（`actual_rect`）を返却。 |
| `extract_cheek_rois(frame, landmarks)` | `frame`: 画像フレーム<br><br><br>`landmarks`: 68点顔座標 | `dict` | 両目・小鼻の幾何位置からノイズの少ない左右頬の ROI 座標および平均 RGB 値をまとめて返却。 |
| `compute_pos_signal(input_rgb_array)` | `input_rgb_array`: 時系列 RGB 配列 `(N, 3)` | `np.ndarray` | POS（Plane-Orthogonal-to-Skin）演算を適用し、長さ $N$ の脈波時系列信号配列を算出。 |
| `get_latest_pos_value(input_rgb_array)` | `input_rgb_array`: 時系列 RGB 配列 `(N, 3)` | `float` | POS 演算を行った結果の最新（末尾 1 点）の脈波値を抽出して返却。 |

---

## :red_square: 演習 (`analysis_rppg_exercise.py`)

1. カメラ映像から `extract_cheek_rois` を用いてフレームごとに左右頬の平均 RGB 値を取得し、左右の平均値を算出して長さを一定に保つリングバッファ（例: 90 フレーム分）に追加してください。
2. バッファに溜まった RGB アレイに対して `compute_pos_signal` を実行し、返された 1 次元配列に対して `numpy` や `scipy` を用いて高速フーリエ変換（FFT）を行い、心拍数（bpm）に対応するピーク周波数を算出するプログラムを作成してください。
