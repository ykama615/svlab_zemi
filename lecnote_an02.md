# 3D身体姿勢・背骨・移動量解析ライブラリ (my_analysis_body.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、2D 姿勢ランドマーク（MediaPipe Pose など）から身体の重心・背骨（Neck/Centroid/Hip）の算出、上半身・下半身・首の姿勢屈曲角の計測、および各部位のフレーム間移動量を計測する `MyAnalysisBody` クラスの動作原理と利用方法について解説します。

## 前提条件
- **【重要】** `my_analysis_body.py` がライブラリフォルダー（例: `my_libs/`）に配置されていることを確認してください。
- **【重要】** 内部処理に `numpy` および Python 標準ライブラリの `math`, `collections.deque` を使用します。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

  ```

---

## :red_square: 幾何学的アルゴリズムと動作原理

`MyAnalysisBody` は、左右の肩座標および腰座標から身体のフレーム構造（背骨ライン）を定義し、姿勢の歪みや運動移動量を幾何学的に解析するモジュールです。

```
[ 2D 姿勢 ランドマーク (肩・腰) ]
       │
       ├─► 1. 重心計算 (get_body_centroid) ───► 全体重心・上半身重心・下半身重心・法線ベクトルを算出
       │                                           │
       ├─► 2. 背骨算出 (get_backbone) ◄─────────────┴ 最小二乗法により Neck点 / 身体中心 / Hip点を特定
       │                                           │
       ├─► 3. 姿勢角計算 (get_posture_angles) ◄─────┘ 基準角からの Lower / Upper / Neck 屈曲角を算出
       │
       └─► 4. 移動量検知 (get_body_landmark_moving) ──► deque 履歴を用いたフレーム間のユークリッド移動量算出

```

### 1. 身体重心および方向ベクトルの算出

* 左右の肩（必須）と左右の腰（任意）の平均値から身体の重心（`body_centroid`）を計算します。
* 肩の方向ベクトル（$n_s$）と腰の方向ベクトル（$n_h$）を合成・正規化し、身体の向き（`cv`）および上半身・下半身のローカル重心（`upper_centroid`, `lower_centroid`）を割り出します。

### 2. 最小二乗法を用いた背骨ライン（Neck / Center / Hip）の再構築

* 上半身重心（$c_u$）を通り、身体の向きベクトル（$cv$）に垂直な直線と、左右の肩を結ぶ直線との交点を最小二乗法（`np.linalg.lstsq`）により算出し、首の付け根（`neck`）の位置を推定します。
* 同様の計算を腰ライン（$c_l$）に対しても適用し、腰の中心（`hip_c`）の位置を交点として算出します。

### 3. 関節屈曲角の補正と算出

* 腰または肩の水平線（基準線 `base_angle`）を基準軸（90度）として設定します。
* 「腰中心→身体中心」（`lower_angle`）、「身体中心→首」（`upper_angle`）、「首→頭部中心」（`neck_angle`）の相対的な傾き角（Euler 度数）を順次算出し、身体の猫背や前傾姿勢を定量化します。

### 4. 時系列 Deque による移動検知

* 指定されたフレーム履歴長（`maxlen`）を持つ `deque` 構造を用いて各関節の座標履歴を管理します。
* 最も古いフレーム座標と最新座標の差分から、方向符号付きの移動量（`hypot(dx, dy)`）を計算します。

---

## :red_square: 基本的な使い方とサンプルコード

### 1. 姿勢ランドマークからの背骨・姿勢角および移動量の推定 (body_analysis_sample.py)

```python
import numpy as np
from my_libs.my_analysis_body import MyAnalysisBody

def main():
    body_analyzer = MyAnalysisBody()

    # 1. 姿勢ランドマーク（左右の肩・腰）の登録
    sholder_l = [400, 200]
    sholder_r = [240, 200]
    hip_l     = [380, 400]
    hip_r     = [260, 400]
    
    body_analyzer.set_pose_landmarks(
        sholder_l=sholder_l,
        sholder_r=sholder_r,
        hip_l=hip_l,
        hip_r=hip_r
    )

    # 2. 背骨ライン (Neck, Body Center, Hip) の算出
    backbone = body_analyzer.get_backbone()
    if backbone:
        neck, center, hip = backbone
        print(f"首位置: {neck}, 身体中心: {center}, 腰位置: {hip}")

    # 3. 姿勢屈曲角の計算 (頭部中心座標を渡す場合)
    head_center = [320, 120]
    angles = body_analyzer.get_posture_angles(head_center=head_center)
    if angles:
        print(f"下半身角度: {angles['lower_angle']}°")
        print(f"上半身角度: {angles['upper_angle']}°")
        print(f"首屈曲角度: {angles['neck_angle']}°")

    # 4. フレーム間の移動量検知 (履歴数 5 フレーム)
    sholder_pair = [sholder_r, sholder_l]
    mv_hp, mv_sh, mv_bc = body_analyzer.get_body_landmark_moving(
        head_center=head_center,
        sholder=sholder_pair,
        body_center=center,
        maxlen=5
    )
    print(f"頭部移動量: {mv_hp}, 身体中心移動量: {mv_bc}")

if __name__ == '__main__':
    main()

```

---

## :red_square: API リファレンス

### MyAnalysisBody クラス

| メソッド | 主要引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `set_pose_landmarks(sholder_l, sholder_r, hip_l, hip_r)` | `sholder_l/r`: 左右肩座標<br><br>`hip_l/r`: 左右腰座標 | なし | 左右の肩座標および腰座標（`[x, y]`）を登録。 |
| `get_body_centroid()` | なし | `tuple` / `None` | 全体重心、上半身重心、下半身重心、合成方向ベクトル、肩ベクトルの 5 要素を返却。 |
| `get_backbone()` | なし | `tuple` / `None` | 最小二乗法に基づき算出した背骨ライン（`neck`, `body_center`, `hip_center`）の 3 点座標を返却。 |
| `get_posture_angles(head_center=None)` | `head_center`: 顔の中心座標 `[x, y]` | `dict` / `None` | 基準角に対する `lower_angle`（下半身）, `upper_angle`（上半身）, `neck_angle`（首）の屈曲角を返却。 |
| `get_body_landmark_moving(head_center, sholder, body_center, maxlen)` | `head_center`: 頭部座標<br><br>`sholder`: `[右肩, 左肩]`<br><br>`body_center`: 重心<br><br>`maxlen`: 履歴フレーム数 | `tuple` | 時系列 deque を保持し、頭部 (`mv_hp`)・両肩 (`mv_sh`)・身体中心 (`mv_bc`) のフレーム間移動量を算出。 |

---

## :red_square: 演習 (`analysis_body_exercise.py`)

1. カメラ画像から取得した姿勢ランドマークに対し、`get_posture_angles` を呼び出して `upper_angle`（上半身の傾き角）を取得してください。
2. 前傾姿勢になり `upper_angle` の絶対値が `20` 度を超えた場合に、「姿勢を正してください」と画面またはターミナルに警告を表示するプログラムを作成してください。

---

[トップページへ戻る]()
