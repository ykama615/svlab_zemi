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
16. 呼吸信号抽出 (`my_analysis_respiration.py`)（↓）
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](lecnote_an04.md)

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>

自作ライブラリ `my_libs.analysis` 内の呼吸信号処理クラス `MyAnalysisRespiration` を活用し、顔領域と Depth 画像から胸部・腹部の ROI 抽出、生デプス平均値の算出、および呼吸運動の検知を実装するための解説ドキュメントです。

<hr>

# 呼吸信号（胸部・腹部デプス平均値）抽出ライブラリ (my_analysis_respiration.py) の使い方

## 目的

* 本ドキュメントでは、顔領域座標（`face_region`）と Depth（奥行き）画像データから、胸部および腹部の関心領域（ROI）を推定し、その瞬間の生のデプス平均値を抽出する `MyAnalysisRespiration` クラスの動作原理と利用方法について解説します。

## 前提条件

* **【重要】** `my_analysis_respiration.py` がライブラリフォルダー（例: `my_libs/analysis/`）に配置されていることを確認してください。
* **【重要】** 内部処理に `numpy` を使用します。
* ターミナルで以下のコマンドを実行してプログラムを動作させます。
```sh
C:\oit\home\ipbl> python XXX.py

```



---

## :red_square: 幾何学的アルゴリズムと動作原理

`MyAnalysisRespiration` は、カメラから取得した顔領域の位置関係を基準に、体幹（胸部・腹部）の 2D ROI 領域を空間的に算出し、対応する Depth マップから体表の変位（呼吸運動）の指標となる生デプス平均値を抽出するモジュールです。

```
[ 顔領域 (fx, fy, fw, fh) ] + [ Depth マップ ]
        │
        ├─► 1. ROI 領域推定 (extract_respiration_rois) ────► 顔サイズ比（下1.5倍/2.8倍）から胸部・腹部 ROI を算出
        │                                                    │
        ├─► 2. 境界領域判定 (_region_is_inside) ──────────────＋ 画面サイズ (wt, ht) 内に収まるかチェック
        │                                                    │
        └─► 3. 生デプス抽出 (get_raw_respiration_values) ◄────┘ ROI 内の有効値 (depth > 0) の平均値を返却

```

### 1. 顔サイズ比に基づく胸部・腹部 ROI の推定

* 人体の幾何的比例関係に基づき、顔の位置 `(fx, fy)` およびサイズ `(fw, fh)` から以下のように胸部および腹部の 2D バウンディングボックスを決定します。
* **胸部 ROI**: 顔の下方向 $1.5 \times fh$ の位置に配置（高さ: $fh / 3$）
* **腹部 ROI**: 顔の下方向 $2.8 \times fh$ の位置に配置（高さ: $fh / 3$）

### 2. 画面領域内チェック（境界保護）

* 計算された ROI の座標・幅・高さが、カメラ解像度（デフォルト: $640 \times 480$）の範囲内に完全に収まっているかを判定します。
* 顔が画面下端に寄りすぎて ROI が画面外へ食み出る場合は、安全のため ROI を `None` に設定し、領域外アクセスによる例外を防止します。

### 3. 有効デプス値の抽出と平均値計算

* 指定された ROI 内の Depth 配列から、信頼できる有効データ（`depth > 0`）のみを抽出して算術平均値を算出します。
* データバッファや時系列フィルタ処理を行わない「ステートレス設計」とすることで、純粋な生数値の集計処理を行います。

---

## :red_square: 基本的な使い方とサンプルコード

### 1. 顔領域と Depth 画像からの呼吸 raw データの抽出 (`respiration_sample.py`)

```python
import numpy as np
from my_libs.analysis.my_analysis_respiration import MyAnalysisRespiration

def main():
    # 640x480 解像度でインスタンス化
    resp_analyzer = MyAnalysisRespiration(wt=640, ht=480)

    # ダミーの Depth 画像データ (640x480, 奥行き約1000mm)
    dummy_depth_map = np.full((480, 640), 1000, dtype=np.uint16)

    # 検出された顔領域 (x, y, w, h)
    face_region = [200, 50, 100, 120]

    # 1. 胸部・腹部 ROI 座標の決定
    chest_roi, abs_roi = resp_analyzer.extract_respiration_rois(face_region)
    print(f"胸部 ROI: {chest_roi}")
    print(f"腹部 ROI: {abs_roi}")

    # 2. その瞬間の生のデプス平均値を取得
    if chest_roi and abs_roi:
        raw_chest, raw_abs = resp_analyzer.get_raw_respiration_values(
            dummy_depth_map, chest_roi, abs_roi
        )
        print(f"胸部デプス生値: {raw_chest:.2f} mm")
        print(f"腹部デプス生値: {raw_abs:.2f} mm")

if __name__ == '__main__':
    main()

```

---

## :red_square: API リファレンス

### MyAnalysisRespiration クラス

| メソッド | 主要引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__(wt=640, ht=480)` | `wt`: 画面幅<br><br><br>`ht`: 画面高さ | なし | 画面解像度を設定して初期化。 |
| `extract_respiration_rois(face_region)` | `face_region`: 顔位置 `[x, y, w, h]` | `tuple` | 顔領域から胸部 (`chest_roi`) および腹部 (`abs_roi`) の 2D 座標 `[x, y, w, h]` を決定。 |
| `get_raw_respiration_values(depth_map, chest_roi, abs_roi)` | `depth_map`: Depth画像<br><br><br>`chest_roi`: 胸部ROI<br><br><br>`abs_roi`: 腹部ROI | `tuple` | 各 ROI 内の有効デプス値（`> 0`）の平均値 `(raw_chest, raw_abs)` を返却。 |

---

## :red_square: 演習 (`analysis_respiration_exercise.py`)

1. ループ内で連続して得られる Depth フレームに対し、`get_raw_respiration_values` で取得した `raw_chest`（胸部デプス値）を配列へ記録してください。
2. 直近 30 フレームの `raw_chest` の変動幅（`max - min`）を計算し、`15.0 mm` 以上の変位があった場合に「呼吸（体動）を検知」と表示するプログラムを作成してください。
