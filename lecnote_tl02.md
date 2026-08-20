# CSVファイル入出力とファイルパス・日時操作ライブラリ (my_csv.py / my_util.py)

[トップページへ戻る](../README.md)

---

## 目的
- 本ドキュメントでは、パス情報や日時文字列の取得を行う `myUtil` モジュールと、CSVデータの書き込み・読み込みを行う `MyCSVWriter` / `MyCSVReader` の利用方法を解説します[cite: 6, 7, 8]。

## 前提条件
- **【重要】** `my_csv.py` および `my_util.py` が同一ディレクトリまたはライブラリフォルダー（例: `my_libs/`）に配置されていることを確認してください[cite: 6, 8]。
- ターミナルで以下のコマンドを実行してプログラムを動作させます。
  ```sh
  C:\oit\home\ipbl> python XXX.py

```

---

## :red_square: ユーティリティモジュール (myUtil)

`myUtil` は、ファイルパスの整形や実験データ保存時に便利な日時スタンプ文字列を取得するための静的メソッド群です。

### 主な機能

* **パス構成要素の取得**: 渡されたパス文字列から拡張子無しのファイル名、拡張子、親ディレクトリパスを抽出します。


* **日時スタンプの生成**: 現刻の年・月・日・時・分を `YYYY-MM-DD_hhmm` 形式の文字列として取得します（ファイル名への付与に最適）。



### myUtil の単体実行サンプル

```python
from my_libs.my_util import myUtil

def main():
    path_str = "logs/sensor/data.csv"

    # ファイルパスの分解
    print("ファイル名:", myUtil.get_filename(path_str))    # -> data[cite: 8]
    print("拡張子:    ", myUtil.get_suffix(path_str))      # -> .csv[cite: 8]
    print("親パス:    ", myUtil.get_parent_path(path_str)) # -> logs/sensor[cite: 8]

    # 現在日時の取得
    time_str = myUtil.get_date_time_str()
    print("現在日時:  ", time_str)                       # -> 2026-08-20_1900 等[cite: 8]

if __name__ == '__main__':
    main()

```

---

## :red_square: CSVへの書き込み（MyCSVWriter）

`myUtil` を内部で利用して既存ファイルとの重複を回避し、安全または高速に CSV データを書き込みます。

### 主な特徴

1. **自動ナンバリング機能**: `myUtil` でパスを分解し、同名ファイルが存在する場合 `filename_1.csv` のように連番を自動付与して上書きを防ぎます。


2. **書き込みモード選択**: 逐次閉じる安全モード（`keep_open=False`）と、リソースを保持する高速モード（`keep_open=True`）に対応します。



### my_csv_write_sample.py

```python
from my_libs.my_util import myUtil
from my_libs.my_csv import MyCSVWriter

def main():
    # 日時スタンプ付きのファイル名を生成
    date_str = myUtil.get_date_time_str()[cite: 8]
    filename = f"data/experiment_{date_str}.csv"

    # ライタの初期化 (keep_open=True で高速モード)
    writer = MyCSVWriter(filename, keep_open=True)[cite: 6]

    # ヘッダーとデータの追加
    writer.write(["Time", "Val_1", "Val_2"])[cite: 6]
    for i in range(5):
        writer.write([i * 0.1, i * 10, i * 100])[cite: 6]

    writer.close()[cite: 6]
    print(f"保存完了: {writer.filename}")[cite: 6]

if __name__ == '__main__':
    main()

```

---

## :red_square: CSVからの読み込み（MyCSVReader）

保存された CSV ファイルを読み込み、列データ（`column`）単位でリスト化します。

### my_csv_read_sample.py

```python
from my_libs.my_csv import MyCSVReader

def main():
    target_file = "data/experiment_log.csv"

    # 1. 2次元リストとして列別に読み込み
    cols = MyCSVReader.read(target_file, dtype=float)[cite: 7]
    print("Time列:", cols[0])[cite: 7]

    # 2. 辞書型（キー指定）で読み込み
    data_dict = MyCSVReader.read(target_file, id_list=["time", "val1", "val2"], dtype=float)[cite: 7]
    print("Val_1列:", data_dict["val1"])[cite: 7]

if __name__ == '__main__':
    main()

```

---

## :red_square: APIリファレンス

### myUtil クラス (静的クラス)

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `get_filename(path_str)` | `path_str`: ファイルパス | `str` | 拡張子を除いたファイル名を取得。 |
| `get_suffix(path_str)` | `path_str`: ファイルパス | `str` | 拡張子（例: `.csv`）を取得。 |
| `get_parent_path(path_str)` | `path_str`: ファイルパス | `Path` | 親ディレクトリのパスオブジェクトを取得。 |
| `get_date_time_str()` | なし | `str` | 現刻を `YYYY-MM-DD_hhmm` 形式の文字列で取得。 |

### MyCSVWriter クラス

| メソッド | 引数 | 説明 |
| --- | --- | --- |
| `__init__(filename, keep_open=False)` | `filename`: 保存パス<br><br>`keep_open`: ファイル開放維持 | インスタンス作成。`myUtil` を参照してファイル重複時に番号を自動付与。 |
| `write(vlist)` | `vlist`: データ配列 | 1 行分の配列データを追加書き込み。 |
| `close()` | なし | ファイルを安全に閉じる（`keep_open=True` 時に使用）。 |

### MyCSVReader クラス

| メソッド | 引数 | 戻り値 | 説明 |
| --- | --- | --- | --- |
| `read(filename, id_list=[], dtype=float)` | `filename`: 読込パス<br><br>`id_list`: 列用キー<br><br>`dtype`: 変換型 | `list` / `dict` | CSVを読み込み、列要素を抽出・型変換して返却。 |

---

## :red_square: 演習 (`csv_util_exercise.py`)

1. `myUtil.get_date_time_str()` を使い、`log_YYYY-MM-DD_hhmm.csv` というファイル名を自動生成して `MyCSVWriter` で保存してください。


2. 保存したファイルを `MyCSVReader` で読み込み、取得したデータ要素数を表示する処理を作成してください。



---

[トップページへ戻る]()
