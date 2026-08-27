# 3D頭部姿勢・視線・顔正面化解析ライブラリ (my_analysis_head.py) の使い方

[トップページへ戻る](../../README.md)

---

## 目的
- 本ドキュメントでは、2D ランドマーク（dlib / MediaPipe）から 3D 頭部姿勢角（Pitch/Yaw/Roll）の推定、瞳孔の 3D 空間マッピング、視線ベクトルの抽出、およびデバッグ用の顔正面化再構成描画を行う `MyAnalysisHead` クラスの動作原理と利用方法について解説します。

## 前提条件
- **【重要】** `my_analysis_head.py` がライブラリフォルダー（例: `my_libs/`）に配置されていることを確認してください。
- **【重要】** 内部処理に `opencv-python` (`cv2`) および `numpy` を使用します。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

  ```

---

## :red_square: 幾何学的アルゴリズムと動作原理

`MyAnalysisHead` は、2D 画像上の特徴点から 3D 空間上の頭部位置・姿勢・視線方向を幾何学的に復元・解析するモジュールです。

```
[ 2D ランドマーク (dlib 68点) ]
       │
       ├─► 1. 姿勢推定 (solvePnP) ─────► 回転行列 R / 平行移動 Vector t を算出
       │                                      │
       ├─► 2. 3D 瞳孔マッピング ◄──────────────┤ Z奥行き補正 ＆ 逆回転 R^T で顔ローカルへ変換
       │                                      │
       ├─► 3. 視線ベクトル抽出 ◄───────────────┤ 目の中心からの正規化オフセット計算
       │                                      │
       └─► 4. 正面顔再構成 (デバッグ) ◄─────────┘ 68点全域を 3D 復元して回転キャンセルの上描画

```

### 1. PnP (Perspective-n-Point) による姿勢推定

* 標準的な 13 点の 3D 形状モデル（`model_pts`）と 2D 画像上の対応点を用いて、OpenCV の `cv2.solvePnP`（EPnP アルゴリズム）を実行します。
* これにより、カメラ座標系に対する頭部の回転行列 $R$（`rmat`）と平行移動ベクトル $t$（`tvec`）を算出し、Euler 角（`pitch`, `yaw`, `roll`）へ変換します。

### 2. Z 奥行きを考慮した 3D 瞳孔マッピング

* 2D 瞳孔座標を 3D カメラ空間へ逆投影する際、モデルの回転 $R$ を適用した左右の目頭・目尻の $Z$ 座標平均値（`z_lpoint`, `z_rpoint`）を用いて奥行きを補正します。
* さらに、頭部中心からのオフセットに回転行列の転置（$R^T$＝逆回転）を適用することで、顔の傾きに依存しない「顔ローカル座標系での瞳孔位置」（`re_left`, `re_right`）を復元します。

### 3. 正規化視線ベクトルの算出

* 顔ローカル空間で復元された 3D 瞳孔位置と、目頭・目尻の中点（目の中心）の差分を算出します。
* 目の幅でスケーリング（正規化）を行うことで、顔がどの向きを向いていても正確な水平・垂直の視線度数（`horizontal`, `vertical`）を取得します。

### 4. 正面顔 3D 再構成（デバッグ描画）

* 68 点の標準顔モデル（`master_68`）を生成・スケーリングし、各頂点の奥行き $Z$ 座標を推定します。
* 画像上の全ランドマークに逆回転 $R^T$ をかけることで「顔が正面を向いた状態」の 3D 座標を再構築し、2D キャンバス上に描画（シアン色＝正面化顔、マゼンタ色＝元の顔）します。

---

## :red_square: 基本的な使い方とサンプルコード

### 1. 2D ランドマークからの頭部姿勢と視線の推定 (head_pose_sample.py)

```python
import cv2
import numpy as np
from my_libs.my_analysis_head import MyAnalysisHead

def main():
    img_w, img_h = 640, 480
    head_analyzer = MyAnalysisHead()

    # ダミーの dlib 68 点ランドマーク (画面中央付近に顔があると仮定)
    dummy_landmarks = np.zeros((68, 2), dtype=np.int32)
    for idx in head_analyzer.stable_idx:
        dummy_landmarks[idx] = [320 + idx, 240 + idx]

    # 1. 耳の中点（頭部中心）の登録
    head_analyzer.set_ears(ear_l=[400, 240], ear_r=[240, 240])

    # 2. 頭部姿勢 (Pitch, Yaw, Roll) の計算
    pose_res = head_analyzer.get_head_pose(dummy_landmarks, img_w, img_h)

    if pose_res:
        print(f"Pitch: {pose_res['pitch']}°, Yaw: {pose_res['yaw']}°, Roll: {pose_res['roll']}°")

        # 3. 瞳孔の 3D 位置の計算
        dummy_iris_l = [[360, 220]]  # 左目瞳孔 (2D)
        dummy_iris_r = [[280, 220]]  # 右目瞳孔 (2D)
        iris_3d = head_analyzer.get_iris_3d_positions(dummy_iris_l, dummy_iris_r, pose_res)

        if iris_3d:
            # 4. 視線ベクトルの取得
            gaze_left = head_analyzer.get_gaze_vector(iris_3d["re_left"], pose_res, side="left")
            print("左目 視線角:", gaze_left["angles"])

            # 5. デバッグ用正面顔の再構成描画
            head_center = head_analyzer.get_head_center()
            head_analyzer.draw_front_face_with_iris(dummy_landmarks, pose_res, iris_3d, head_center)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()

```

---

## :red_square: API リファレンス

### MyAnalysisHead クラス

| メソッド | 主要引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `set_ears(ear_l, ear_r)` | `ear_l`: 左耳座標<br><br>`ear_r`: 右耳座標 | なし | 左右の耳座標から頭部中心座標 (`head_center`) を自動計算して保持。 |
| `set_head_center(center_pt)` | `center_pt`: `[x, y]` | なし | 頭部中心座標を直接指定して保持。 |
| `get_head_center()` | なし | `list` / `None` | 現在保持している頭部中心座標 (`[x, y]`) を取得。 |
| `set_mp_matrix(matrix)` | `matrix`: 4x4 行列 | なし | MediaPipe の 4x4 姿勢変換行列をセット（セット時は PnP より優先使用）。 |
| `get_head_pose(dlib_landmark, img_w, img_h, head_center=None)` | `dlib_landmark`: 68点座標<br><br>`img_w`, `img_h`: 解像度 | `dict` | 頭部姿勢（`pitch`, `yaw`, `roll`, `rmat`, `tvec`, `cam_mtx` 等）を解く。 |
| `get_iris_3d_positions(lpoint_list, rpoint_list, pose_res)` | `lpoint_list`, `rpoint_list`: 2D 瞳孔座標<br><br>`pose_res`: 姿勢結果 | `dict` | 瞳孔の 3D カメラ空間座標（`left`, `right`）および顔ローカル復元座標（`re_left`, `re_right`）を計算。 |
| `get_gaze_vector(iris_3d_points, pose_res, side="left")` | `iris_3d_points`: 3D 瞳孔座標<br><br>`pose_res`: 姿勢結果<br><br>`side`: `'left'` / `'right'` | `dict` | 3D 視線ベクトルおよび水平・垂直の視線度数（`horizontal`, `vertical`）を取得。 |
| `get_eye_aspect_ratio(dlib_landmark, pose_res, side="left")` | `dlib_landmark`: 68点座標<br><br>`pose_res`: 姿勢結果<br><br>`side`: `'left'` / `'right'` | `float` | 姿勢回転の影響を取り除いた正確な EAR（目の開き具合）を算出。 |
| `draw_front_face_with_iris(face_landmarks, pose_res, iris_res, head_center)` | `face_landmarks`: 顔座標<br><br>`pose_res`: 姿勢結果<br><br>`iris_res`: 瞳孔3D結果<br><br>`head_center`: 頭部中心 | なし | **【デバッグ用】** 傾いた顔と正面化した 3D 再構成顔を OpenCV ウィンドウ（"Normalized Front Face"）で比較描画。 |

---

## :red_square: 演習 (`analysis_head_exercise.py`)

1. カメラ画像から取得した顔データに対し、`get_head_pose` を呼び出して `yaw`（首の横振り角度）を取得してください。
2. `yaw` の値が `+15` 度以上または `-15` 度以下になった場合に、「脇見注意」とターミナルに表示する警告ロジックを作成してください。

---

[トップページへ戻る]()
