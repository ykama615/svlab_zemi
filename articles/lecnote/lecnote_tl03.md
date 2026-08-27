# デジタル信号処理ライブラリ (my_digital_filter.py) の使い方

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、波形データのフィルタリング、3次スプライン補間、移動平均、ピーク検出、トレンド除去機能を提供する `myDigitalFilter` クラス（`my_digital_filter.py`）の利用方法について解説します[cite: 9]。

## 前提条件
- **【重要】** `my_digital_filter.py` がライブラリフォルダー（例: `my_libs/`）に配置されていることを確認してください[cite: 9]。
- 内部で数値計算・信号処理を行うため、`numpy` および `scipy` がインストールされている必要があります[cite: 9]。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

```

---

## :red_square: myDigitalFilter の概要と特徴

`myDigitalFilter` は、`scipy.signal` や `scipy.interpolate` などの高度な信号処理モジュールを静的メソッド（`@staticmethod`）として集約し、直感的な記述で波形整形や解析を行えるようにしたクラスです。

### 主な特徴

1. **位相遅れのない周波数フィルタ (`frequency_filter`)**:
* バタワースデジタルフィルタを双方向（`filtfilt`）で適用するため、信号の位相（時間的なズレ）を保ったまま高周波ノイズを除去できます。




2. **不等間隔データの等間隔補間 (`complement`)**:
* サンプリング周期が変動するセンサデータを 3 次スプライン補間し、指定したサンプリング周波数 `fs` の均一なデータに再構築します。




3. **ストリーミング向け平滑化処理**:
* 配列全体に対する単純移動平均（`moving_average`）に加え、逐次更新用の指数移動平均（`exponential_moving_average`）や最新点のみの高速移動平均抽出（`extract_latest_smoothed_value`）を備えています。




4. **前処理・特徴抽出機能**:
* 基線変動（漂流トレンド）を取り除く `remove_trend` や、波形の極大値を抽出する `find_peaks` を容易に呼び出せます。





---

## :red_square: 基本的な使い方とサンプルコード

### 1. ノイズ除去とトレンド除去 (filter_sample.py)

高周波ノイズが含まれる波形からノイズを取り除き、直線的な傾き（トレンド）を除去する例です。

```python
import numpy as np
from my_libs.my_digital_filter import myDigitalFilter

def main():
    fs = 100.0  # サンプリング周波数 [Hz]
    t = np.linspace(0, 2.0, int(2.0 * fs))

    # 正弦波 + トレンド成分 + ノイズ
    raw_signal = np.sin(2 * np.pi * 3 * t) + 0.5 * t + 0.3 * np.random.normal(size=len(t))

    # 1. ローパスフィルタリング (カットオフ周波数 5Hz)
    filtered = myDigitalFilter.frequency_filter(raw_signal, fs=fs, fc=5.0, btype='low')[cite: 9]

    # 2. 直線トレンドの除去（ドリフト成分の除去）
    detrended = myDigitalFilter.remove_trend(filtered, type='linear')[cite: 9]

    # 3. 単純移動平均 (窓幅 5)
    ma_signal = myDigitalFilter.moving_average(detrended, window_size=5)[cite: 9]

    print("処理完了 - 元データ数:", len(raw_signal), "移動平均データ数:", len(ma_signal))

if __name__ == '__main__':
    main()

```

---

### 2. データ補間とピーク検出 (peaks_sample.py)

時間間隔が不均一なデータを一定のサンプリング周期に補間し、極大値（ピーク）の位置を検出する例です。

```python
import numpy as np
from my_libs.my_digital_filter import myDigitalFilter

def main():
    # 時間間隔がバラバラなサンプリングデータ
    time = np.array([0.0, 0.12, 0.25, 0.38, 0.51, 0.65, 0.78, 0.90, 1.0])
    data = np.array([0.1, 0.5,  0.9,  0.4,  0.1,  0.8,  0.3,  0.1,  0.0])

    # 1. 3次スプライン補間 (20Hz の等時間間隔に再構築)
    ntime, ndata = myDigitalFilter.complement(time, data, fs=20.0)[cite: 9]

    # 2. ピーク（極大値）の検出 (高さ 0.5 以上の頂点を抽出)
    peaks, properties = myDigitalFilter.find_peaks(ndata, height=0.5)[cite: 9]

    print("補間後のデータ点数:", len(ndata))
    print("検出されたピーク位置(インデックス):", peaks)[cite: 9]
    print("ピーク位置の時間[s]:", ntime[peaks])

if __name__ == '__main__':
    main()

```

---

## :red_square: APIリファレンス

### myDigitalFilter クラス (静的クラス)

| メソッド | 主要引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `frequency_filter(data, fs, fc, btype='low')` | `data`: 入力配列<br><br>`fs`: サンプリング周波数[Hz]<br><br>`fc`: カットオフ周波数[Hz]<br><br>`btype`: `'low'`, `'high'`, `'band'` | `np.ndarray` | バタワース 2 次フィルタをゼロ位相（双方向）で適用。 |
| `complement(time, data, fs)` | `time`: 時間配列<br><br>`data`: データ配列<br><br>`fs`: 目標サンプリング周波数[Hz] | `(ntime, ndata)` | 3 次スプライン補間により均一なサンプリング間隔のデータへ変換。 |
| `moving_average(data, window_size, mode='valid')` | `data`: 入力配列<br><br>`window_size`: 窓幅 | `np.ndarray` | 畳み込みによる単純移動平均を計算。 |
| `exponential_moving_average(prv, value, alpha)` | `prv`: 前回値<br><br>`value`: 現在値<br><br>`alpha`: 平滑化係数 ($0 < \alpha < 1$) | `float` | 単一値に対する指数移動平均 (EMA) を算出。 |
| `find_peaks(y, width=None, height=None, prominence=None)` | `y`: データ配列<br><br>`height`: 最小高さ<br><br>`prominence`: 突出度 | `(peaks, properties)` | 波形から極大値（ピーク）のインデックスと属性情報を取得。 |
| `remove_trend(data, type='linear')` | `data`: 入力配列<br><br>`type`: `'linear'` / `'constant'` | `np.ndarray` | データから線形または定数のトレンド（傾き・バイアス）を除去。 |
| `extract_latest_smoothed_value(data, window_size=5)` | `data`: 入力配列<br><br>`window_size`: 窓幅 | `float` | 配列末尾（最新データ）の窓幅分から移動平均値 1 点を高速取得。 |

---

## :red_square: 演習 (`digital_filter_exercise.py`)

1. ノイズが含まれた波形データに対して `frequency_filter`（ローパス）を適用し、さらに `remove_trend` でドリフト成分を除去する処理を作成してください。


2. `MyCSVWriter` を使用して、フィルタ処理前と処理後の比較データを CSV ファイルに保存してください[cite: 6, 9]。

---

[トップページへ戻る]()
