# 雑記01

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
