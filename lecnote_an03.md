# 呼吸信号（胸部・腹部デプス平均値）抽出ライブラリ (my_analysis_respiration.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、顔領域座標（`face_region`）と Depth（奥行き）画像データから、胸部および腹部の関心領域（ROI）を推定し、その瞬間の生のデプス平均値を抽出する `MyAnalysisRespiration` クラスの動作原理と利用方法について解説します[cite: 7]。

## 前提条件
- **【重要】** `my_analysis_respiration.py` がライブラリフォルダー（例: `my_libs/`）に配置されていることを確認してください[cite: 7]。
- **【重要】** 内部処理に `numpy` を使用します[cite: 7]。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
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
       │                                                     │
       ├─► 2. 境界領域判定 (_region_is_inside) ───────────────┼ 画面サイズ (wt, ht) 内に収まるかチェック
       │                                                     │
       └─► 3. 生デプス抽出 (get_raw_respiration_values) ◄─────┘ ROI 内の有効値 (depth > 0) の平均値を返却

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

### 1. 顔領域と Depth 画像からの呼吸 raw データの抽出 (respiration_sample.py)

```python
import numpy as np
from my_libs.my_analysis_respiration import MyAnalysisRespiration

def main():
    # 640x480 解像度でインスタンス化
    resp_analyzer = MyAnalysisRespiration(wt=640, ht=480)[cite: 7]

    # ダミーの Depth 画像データ (640x480, 奥行き約1000mm)
    dummy_depth_map = np.full((480, 640), 1000, dtype=np.uint16)

    # 検出された顔領域 (x, y, w, h)
    face_region = [200, 50, 100, 120]

    # 1. 胸部・腹部 ROI 座標の決定
    chest_roi, abs_roi = resp_analyzer.extract_respiration_rois(face_region)[cite: 7]
    print(f"胸部 ROI: {chest_roi}")[cite: 7]
    print(f"腹部 ROI: {abs_roi}")[cite: 7]

    # 2. その瞬間の生のデプス平均値を取得
    if chest_roi and abs_roi:
        raw_chest, raw_abs = resp_analyzer.get_raw_respiration_values(
            dummy_depth_map, chest_roi, abs_roi
        )[cite: 7]
        print(f"胸部デプス生値: {raw_chest:.2f} mm")[cite: 7]
        print(f"腹部デプス生値: {raw_abs:.2f} mm")[cite: 7]

if __name__ == '__main__':
    main()

```

---

## :red_square: API リファレンス

### MyAnalysisRespiration クラス

| メソッド | 主要引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__(wt=640, ht=480)` | `wt`: 画面幅<br><br>`ht`: 画面高さ | なし | 画面解像度を設定して初期化。 |
| `extract_respiration_rois(face_region)` | `face_region`: 顔位置 `[x, y, w, h]` | `tuple` | 顔領域から胸部 (`chest_roi`) および腹部 (`abs_roi`) の 2D 座標 `[x, y, w, h]` を決定。 |
| `get_raw_respiration_values(depth_map, chest_roi, abs_roi)` | `depth_map`: Depth画像<br><br>`chest_roi`: 胸部ROI<br><br>`abs_roi`: 腹部ROI | `tuple` | 各 ROI 内の有効デプス値（`> 0`）の平均値 `(raw_chest, raw_abs)` を返却。 |

---

## :red_square: 演習 (`analysis_respiration_exercise.py`)

1. ループ内で連続して得られる Depth フレームに対し、`get_raw_respiration_values` で取得した `raw_chest`（胸部デプス値）を配列へ記録してください。


2. 直近 30 フレームの `raw_chest` の変動幅（`max - min`）を計算し、`15.0 mm` 以上の変位があった場合に「呼吸（体動）を検知」と表示するプログラムを作成してください。



---

[トップページへ戻る]()
