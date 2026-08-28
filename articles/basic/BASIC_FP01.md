<hr>

**講義ノート・ライブラリ一覧**

<b>➡基礎編</b>
1. [環境の設定](../../README.md)
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. 2つのベクトルのなす角とベクトル演算（↓）

<details><summary><b>キャプチャ（3項目）</b></summary>

7. [動画画像処理 (`my_cap_av2.py`)](../lecnote/lecnote_cap01.md)
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](../lecnote/lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](../lecnote/lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](../lecnote/lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](../lecnote/lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](../lecnote/lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](../lecnote/lecnote_dt04.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](../lecnote/lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](../lecnote/lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](../lecnote/lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](../lecnote/lecnote_an04.md)
</details>

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](../lecnote/lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](../lecnote/lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](../lecnote/lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](../minecraft/mcbot_01.md)
</details>

<hr>

NumPy を用いた 2 つのベクトルのなす角の計算手順，浮動小数点誤差対策，サンプルの解説ドキュメントです．

<hr>

# 2つのベクトルのなす角とベクトル演算

## 2つのベクトルのなす角

* 内積とベクトルの大きさを使って余弦（$\cos \theta$）を求め，その後逆三角関数（$\arccos$）で角度を求めます．

$$cos \theta = {\vec{v_1} \cdot \vec{v_2} \over \vert{}\vec{v_1}\vert{}\vert{}\vec{v_2}\vert{}}$$

* **使用する関数**
* ベクトルの大きさ: `numpy.linalg.norm`
* 内積: `numpy.inner`
* 逆コサイン: `numpy.arccos`
* 弧度法（rad）から度数法（deg）への変換: `numpy.rad2deg`
* 値の範囲制限（誤差対策）: `numpy.clip`
* ※ 逆三角関数は `math` パッケージにも実装されています（`acos`, `asin`, `atan`）



---

### 注意点・補足

1. **丸め誤差対策**
浮動小数点数の計算誤差により，`cos_theta` の値が極稀に `-1.0` 未満や `1.0` を超える値（例: `1.0000000000000002`）になることがあります．これをそのまま `np.arccos` に渡すと `NaN`（非数）が発生するため，`np.clip(..., -1.0, 1.0)` で範囲内に収めるのが安全です．
2. **ゼロベクトルの防止**
長さが 0 のベクトル（ゼロベクトル）が入力された場合のゼロ除算エラーを防ぐため，事前にベクトルの大きさをチェックします．

---

### サンプルコード

```python
# -*- coding: utf-8 -*-
import numpy as np

def calcAngle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)

    # ゼロ除算の防止
    if v1_n == 0 or v2_n == 0:
        return 0.0

    # 浮動小数点誤差対策：-1.0 〜 1.0 の範囲に収める
    cos_theta = np.clip(np.inner(v1, v2) / (v1_n * v2_n), -1.0, 1.0)

    return np.rad2deg(np.arccos(cos_theta))

def main():
    v1 = np.array([1, 1, 1])
    v2 = np.array([1, 1, 0])

    print(calcAngle(v1, v2))

    v1 = np.array([3, 1])
    v2 = np.array([4, 5])

    print(calcAngle(v1, v2))

if __name__ == '__main__':
    main()

```

### 実行結果

```sh
% python deg_sample.py
35.26438968275466
32.905242922987895

```
