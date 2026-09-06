<hr>

**講義ノート・ライブラリ一覧**

<details><summary><b>基礎編（6項目）</b></summary>
  
1. [環境の設定](../../README.md)
2. [基本概要](../basic/BASIC_00.md)
3. [カメラへのアクセスと動画処理](../basic/BASIC_01.md)
4. [顔と顔パーツの検出](../basic/BASIC_02.md)
5. [顔・手・ポーズ検出](../basic/BASIC_03.md)
6. [2つのベクトルのなす角とベクトル演算](../basic/BASIC_FP01.md)
</details>

<details><summary><b>キャプチャ（3項目）</b></summary>

7. [動画画像処理 (`my_cap_av2.py`)](lecnote_cap01.md)
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](lecnote_dt04.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](lecnote_an01.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](lecnote_an04.md)
</details>

<b>➡ツール・信号処理（3項目）</b>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](lecnote_tl01.md)
19. CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)（↓）
20. [デジタル信号処理 (`my_digital_filter.py`)](lecnote_tl03.md)

<details><summary><b>その他（3項目）</b></summary>

21. [ドローン/RTPキャプチャ](../drone/lecnote_dr01.md)
22. [スクリーンキャプチャ](../minecraft/lecnote_mc01.md)
23. [【旧】Minecraftコントロール(1)](../minecraft/mcbot_01.md)    
</details>

<hr>

自作ライブラリ `my_libs` 内のファイルパス・日時操作クラス `myUtil` および CSV入出力クラス `MyCSVWriter / MyCSVReader (my_csv.py / my_util.py)` を活用し、パス・日時文字列の取得、重複を回避する自動連番付きCSV書き込み、およびデータ型変換を伴うCSV読み込みを実装するための解説ドキュメントです。

<hr>

# CSVファイル入出力とファイルパス・日時操作ライブラリ (my_csv.py / my_util.py)

## 目的
- 本ドキュメントでは、パス情報や日時文字列の取得を行う `myUtil` モジュールと、CSVデータの書き込み・読み込みを行う `MyCSVWriter` / `MyCSVReader` の利用方法を解説します。

## 前提条件
- **【重要】** `my_csv.py` および `my_util.py` が `C:\oit\home\ipbl\my_libs` フォルダー内に正しく配置されていることを確認してください。
- **【重要】** 標準ライブラリの `csv`、`math`、`pathlib`、`datetime` が利用可能な環境であることを確認してください。

---

## :red_square: my_util および my_csv の概要と特徴

ファイルパスの操作やタイムスタンプの自動生成を行うユーティリティ（`myUtil`）と、安全・高速なCSV入出力を管理するクラス（`MyCSVWriter` / `MyCSVReader`）について解説します。

### 主な特徴

1. **パス操作とタイムスタンプ生成 (`myUtil`)**:
* ファイル名、拡張子、親ディレクトリの抽出や、`YYYY-MM-DD_HHMM` 形式の現在日時文字列を静的メソッドで簡単に取得できます。


2. **重複防止の自動インクリメント (`MyCSVWriter`)**:
* 同名のファイルが既に存在する場合、自動で連番（`_1`, `_2`...）を付与して既存データを保護します。また、`keep_open=True` により高速な連続書き込みが可能です。


3. **柔軟な列データ抽出 (`MyCSVReader`)**:
* CSVを読み込み、指定したデータ型（`float` / `int`）への変換や `NaN` 処理を行いながら、列ごとのリストや辞書形式としてデータを復元します。



---

## :red_square: ユーティリティとCSV書き込み・読み込みの基本実装

`myUtil` で日時入りのファイル名を生成し、`MyCSVWriter` でデータを書き込んだ後、`MyCSVReader` で読み込む一連の流れです。

### csv_util_basic.py

```python
from my_libs.my_util import myUtil
from my_libs.my_csv import MyCSVWriter, MyCSVReader

def main():
    # 1. ユーティリティを使った日時スタンプ付きファイル名の生成
    date_str = myUtil.get_date_time_str()
    filename = f"data/sample_{date_str}.csv"

    # 2. CSVライタの初期化 (keep_open=True で高速モード)
    writer = MyCSVWriter(filename, keep_open=True)

    # 3. ヘッダーとデータの書き込み
    writer.write(["Index", "Value_A", "Value_B"])
    for i in range(5):
        writer.write([i, i * 1.5, i * 10])

    writer.close()
    print(f"保存完了: {writer.filename}")

    # 4. CSVリーダによるデータの読み込み
    cols = MyCSVReader.read(writer.filename, dtype=float)
    print("読み込んだ列データ (Value_A):", cols[1])

if __name__ == '__main__':
    main()

```

---

## :red_square: 主なメソッド一覧

### myUtil クラス (ファイルパス・日時ユーティリティ)

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `get_filename(path_str)` | `path_str`: ファイルパス文字列 | `str` | 拡張子を除いたファイル名（ステム）を取得する |
| `get_suffix(path_str)` | `path_str`: ファイルパス文字列 | `str` | 拡張子を取得する |
| `get_parent_path(path_str)` | `path_str`: ファイルパス文字列 | `Path` | 親ディレクトリのパスを取得する |
| `get_date_time_str()` | なし | `str` | 現在の日時を `YYYY-MM-DD_HHMM` 形式の文字列として取得する |

### MyCSVWriter クラス (CSV書き込み)

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `__init__(filename, keep_open=False)` | `filename`: 保存先パス<br><br>`keep_open`: 開放維持フラグ | なし | 同名ファイルがある場合は自動で連番を付与して初期化する |
| `write(vlist)` | `vlist`: 1行分のデータリスト | なし | 指定された行データをCSVファイルに書き込む |
| `close()` | なし | なし | 開いているファイルを安全に閉じる |

### MyCSVReader クラス (CSV読み込み)

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `read(filename, id_list, dtype)` | `filename`: 読込先パス<br><br>`id_list`: 辞書キーのリスト<br><br>`dtype`: 変換型 (`float`/`int`) | `list` / `dict` | CSVを読み込み、列ごとのリスト、または指定時は辞書型で返す |

---

## :red_square: 演習 (`csv_util_exercise.py`)

`myUtil.get_date_time_str()` を使って `log_YYYY-MM-DD_HHMM.csv` というファイル名を自動生成し、`MyCSVWriter` で任意の 3行以上のデータを書き込んだ後、保存したファイルを `MyCSVReader` で読み込んでコンソールに表示するプログラムを作成してください。

### 解答サンプルコード (`csv_util_exercise.py`)

```python
from my_libs.my_util import myUtil
from my_libs.my_csv import MyCSVWriter, MyCSVReader

def main():
    # 1. ユーティリティで動的なファイル名を生成
    date_str = myUtil.get_date_time_str()
    filename = f"log_{date_str}.csv"

    # 2. MyCSVWriter を用いたデータの書き込み
    writer = MyCSVWriter(filename, keep_open=False)
    
    writer.write(["Step", "Sensor_X", "Sensor_Y"])
    writer.write([1, 12.5, 98.2])
    writer.write([2, 13.1, 97.5])
    writer.write([3, 11.9, 99.0])
    
    print(f"ファイル '{writer.filename}' への書き込みが完了しました。")

    # 3. MyCSVReader を用いたデータの読み込みと確認
    data_dict = MyCSVReader.read(
        writer.filename, 
        id_list=["step", "x", "y"], 
        dtype=float
    )
    
    print("--- 読み込み結果 (辞書形式) ---")
    for key, values in data_dict.items():
        print(f"{key}: {values}")

if __name__ == '__main__':
    main()

```
