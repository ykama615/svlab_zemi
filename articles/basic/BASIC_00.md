<hr>

**講義ノート・ライブラリ一覧**

<b>基礎編</b>
1. [環境の設定](../../README.md)
2. 基本概要（↓）
3. [カメラへのアクセスと動画処理](articles/basic/BASIC_01.md)
4. [顔と顔パーツの検出](articles/basic/BASIC_02.md)
5. [顔・手・ポーズ検出](articles/basic/BASIC_03.md)
6. [2つのベクトルのなす角とベクトル演算](articles/basic/BASIC_FP01.md)

<details><summary><b>検出・推定（4項目）</b></summary>

7. [MediaPipe統合処理 (`my_mediapipe_n.py`)](articles/lecnote/lecnote_dt01.md)
8. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](articles/lecnote/lecnote_dt02.md)
9. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](articles/lecnote/lecnote_dt03.md)
10. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](articles/lecnote/lecnote_dt04.md)
</details>

<details><summary><b>キャプチャ（3項目）</b></summary>

11. [動画画像処理 (`my_cap_av2.py`)](articles/lecnote/lecnote_cap01.md)
12. [Intel RealSense 画像処理 (`my_rs_cap.py`)](articles/lecnote/lecnote_cap02.md)
13. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](articles/lecnote/lecnote_cap03.md)
</details>

<details><summary><b>生体・動作解析（4項目）</b></summary>

14. [3D頭部姿勢・視線・顔正面化 (`my_analysis_head.py`)](articles/lecnote/lecnote_an01.md)
15. [3D身体姿勢・背骨・移動量 (`my_analysis_body.py`)](articles/lecnote/lecnote_an02.md)
16. [呼吸信号抽出 (`my_analysis_respiration.py`)](articles/lecnote/lecnote_an03.md)
17. [非接触脈波・rPPG信号抽出 (`my_analysis_rppg.py`)](articles/lecnote/lecnote_an04.md)
</details>

<details><summary><b>ツール・信号処理（3項目）</b></summary>

18. [PyQtGraph 高速グラフ描画 (`my_qt_graph.py`)](articles/lecnote/lecnote_tl01.md)
19. [CSV入出力・ファイルパス操作 (`my_csv.py` / `my_util.py`)](articles/lecnote/lecnote_tl02.md)
20. [デジタル信号処理 (`my_digital_filter.py`)](articles/lecnote/lecnote_tl03.md)
</details>

<details><summary><b>その他（1項目）</b></summary>

21. [Minecraftコントロール(1)](articles/minecraft/mcbot_01.md)
</details>

<hr>

Pythonの基本実行モードや文法基礎（クラス・データ構造・大域変数等）から，標準ライブラリ deque を用いた高速なデータ操作までを実例付きでまとめた解説ドキュメントです．<br>
よりしっかり学習したい人は，[「Python ゼロからはじめるプログラミング」](https://mitani.cs.tsukuba.ac.jp/book_support/python/) のPDFなどを参考にしてください．

<hr>

# Pythonの実行方法

## 1. 対話（インタラクティブ）モード
入出力環境（PowerShellやターミナル）を起動後に `python` と入力し，入力プロンプト（`>>>`）の後ろにコマンドを入力していきます．
変数や条件分岐，繰り返し処理などもファイルに記述することなく対話的に実行できます．終了時は `exit()` コマンドまたは `exit(0)` を入力します．

```sh
% python
Python 3.X.X (tags/v3.X.X:...) [MSC v.XXXX 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> print("Hello python")
Hello python
>>> for i in range(5): # range(5)は0～4までのリストを生成
...    print(i)  # print文の前にスペース(インデント)が必要
... # 空のEnterでfor文のブロックを抜ける
0
1
2
3
4
>>> exit(0) #対話モードの終了
%

```

## 2. スクリプトモード (1)

`.py` ファイルにコマンドをまとめて記述し，`python` コマンドを使ってプログラムを実行します．プログラムはファイルの上から順に実行されます．

```python
# script1.py
print("Hello python")

def func_hoge(): # 関数hoge
    print("hoge hoge")

def func_fuga(): # 関数fuga
    print("fuga fuga")

func_hoge()

```

```sh
% python script1.py
Hello python
hoge hoge

```

## 3. スクリプトモード (2)

その `.py` ファイル（モジュール）が「スクリプトとして直接実行された場合にのみ実行する処理」をまとめる `if __name__ == '__main__':` 条件文を記述します．
他のファイルから `import` された場合，このブロック内部の処理は無視されます．

```python
# script2.py
def func_hoge(): # 関数hoge
    print("hoge hoge")

def func_fuga(): # 関数fuga
    print("fuga fuga")

if __name__ == '__main__': # スクリプトとして実行された場合にのみ実行
    print("Hello python")
    func_fuga()
    func_hoge()

```

```sh
% python script2.py
Hello python
fuga fuga
hoge hoge

```

---

# 関数の引数と戻り値

Python の関数では以下の柔軟な呼び出し・戻り値の受け取りが可能です．

* **仮引数に初期値を設定**: 実引数未指定時のデフォルト値を指定
* **キーワード引数**: 仮引数名を使って実引数を指定し，引数の順序を無視
* **複数の戻り値**: タプルやリストとして複数の値を同時に返却
* **アンダースコア `_` による受け取りスキップ**: 不要な戻り値を破棄

```python
# script3.py
def func_msg(num, str="hoge"):
    for i in range(num):
        print(str)

def func_sum(st, ed):
    sum = 0
    for i in range(st, ed + 1):
        sum = sum + i
    return [sum, st, ed] # []を付けない場合はタプルで返却

if __name__ == '__main__':
    func_msg(3)
    all = func_sum(1, 5)
    sm, _, e = func_sum(ed=3, st=1) # キーワード指定＆不用な戻り値を _ でスキップ
    print(all)
    print(sm, e)

```

```sh
% python script3.py
hoge        # func_msgの出力
hoge
hoge
[15, 1, 5]  # func_sumの戻り値をリストで受け取る
6 3         # func_sumの第1と第3戻り値を変数で受け取る

```

---

# クラス

```python
# script4.py
class Twice:
    cnum = 10
    
    def __init__(self):
        print("constructor")

    def twice(self):
        print(self.cnum * 2)

    def setnum(self, num):
        self.cnum = num

if __name__ == '__main__':
    tw = Twice()  
    tw.twice()
    tw.setnum(15)
    tw.twice()

```

```sh
% python script4.py
constructor
20
30

```

---

# 大域変数 (global)

```python
# script5.py
def twice():
    global gnum
    print(gnum * 2)

def main():
    global gnum
    gnum = 10 # gnumはここで定義される
    twice()

if __name__ == '__main__':
    global gnum
    # print(gnum) -> NameError: name 'gnum' is not defined (main実行前のため未定義)
    main()
    print(gnum)

```

```sh
% python script5.py
20
10

```

---

# リスト，タプル，辞書，集合

```python
# script6.py
# -*- coding: utf-8 -*-  

# リスト（要素の書き換えが可能な順序付き配列）
lst = ['子', '丑', '寅', '卯']

# タプル（イミュータブルで要素の書き換えができない配列）
tpl = ('子', '丑', '寅', '卯')

# 辞書（キーを使って要素にアクセスする構造）
dic = {'十二支': ['子', '丑', '寅', '卯'], '十二月': ['睦月', '如月', '弥生']}

# 集合（重複のないデータの集まり・インデックスなし）
stt = {'1月', '2月', '3月'}

print(tpl[0], lst[0], dic['十二支'][0])

lst.append('猫')
dic['十二支'].append('猫')

print(tpl, lst, dic['十二支'])
print(stt)

# 辞書の統合・追加
dic_a = {'十二月': ['睦月', '如月', '弥生']}
dic_b = {'十二刻': ['子', '丑', '寅', '卯']}
dic.update(**dic_a, **dic_b) # 複数の辞書を追加したい場合は ** を付ける

print('辞書追加', dic)

```

```sh
% python script6.py
子 子 子
('子', '丑', '寅', '卯') ['子', '丑', '寅', '卯', '猫'] ['子', '丑', '寅', '卯', '猫']
{'2月', '1月', '3月'}
辞書追加 {'十二支': ['子', '丑', '寅', '卯', '猫'], '十二月': ['睦月', '如月', '弥生'], '十二刻': ['子', '丑', '寅', '卯']}

```

---

# for文とリスト内包表記

```python
# script7.py
lst = ['子', '丑', '寅', '卯']
tpl = ('子', '丑', '寅', '卯')
dic = {'十二支': ['子', '丑', '寅', '卯'], '十二月': ['睦月', '如月', '弥生']}

for i in range(len(lst)):
    print(lst[i])

# リスト内包表記
tmp = [lst[i] for i in range(len(lst))]
print(tmp)

for elm in enumerate(tpl):
    print(elm)

for i, elm in enumerate(tpl):
    print(i, ": ", elm)

for kw in dic.keys():
    print(kw)

for val in dic.values():
    print(val)

for itm in dic.items():
    print(itm[0], itm[1])

```

```sh
% python script7.py  
子
丑
寅
卯
['子', '丑', '寅', '卯']
(0, '子')
(1, '丑')
(2, '寅')
(3, '卯')
0 :  子
1 :  丑
2 :  寅
3 :  卯
十二支
十二月
['子', '丑', '寅', '卯']
['睦月', '如月', '弥生']
十二支 ['子', '丑', '寅', '卯']
十二月 ['睦月', '如月', '弥生']

```

---

# モジュールの import とパッケージ

`.py` のファイル名はモジュール名として扱われ，`import` によってモジュールの読み込みが可能です．
`.py` ファイルをフォルダにまとめ，`__init__.py` ファイルを入れることでパッケージ（ライブラリ）として扱うことができます．

```python
# mul_module.py
def twice(num):
    print(num * 2)

```

```python
# script8.py
# -*- coding: utf-8 -*-  
import mul_module

mul_module.twice(10)

```

```sh
% python script8.py
20

```

`__init__.py` の記述例:

```python
from モジュール名 import *

```

---

# collections モジュールの deque (デック)

リストより高速かつ高機能な両端キュー構造として `collections.deque` があります．
数値だけでなく，配列（リスト）や画像データなど様々なデータを格納でき，両端への要素の追加・取り出しが高速です．

```python
# deque_sample.py
import numpy as np
from collections import deque # dequeの利用に必要

def main():
    queue = deque()
    for num in range(10, 21, 2):
        queue.append(num) # 末尾に追加
    print(queue)

    for num in range(11, 20, 2):
        queue.appendleft(num) # 先頭（左）に挿入
    print(queue)

    queue.insert(5, 0) # 添字5の位置に0を挿入
    print(queue)

    queue.reverse() # 要素を逆順にする
    print(queue)

    print(queue.pop()) # 末尾からpop
    print(queue)

    print(queue.popleft()) # 先頭からpop
    print(queue)

    for num in queue: # すべての要素を参照
        print(num, queue)

if __name__ == '__main__':
    main()

```

```sh
% python deque_sample.py
deque([10, 12, 14, 16, 18, 20])
deque([19, 17, 15, 13, 11, 10, 12, 14, 16, 18, 20])
deque([19, 17, 15, 13, 11, 0, 10, 12, 14, 16, 18, 20])
deque([20, 18, 16, 14, 12, 10, 0, 11, 13, 15, 17, 19])
19
deque([20, 18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
20
deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
18 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
16 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
14 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
12 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
10 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
0 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
11 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
13 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
15 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])
17 deque([18, 16, 14, 12, 10, 0, 11, 13, 15, 17])

```

* **複数要素の連結 (`extend` / `extendleft`)**:
`extend()` は末尾に連結し，`extendleft()` は順に先頭へ追加します．

```python
a = deque([1, 2, 3])
b = [4, 5, 6]
a.extend(b)
print(a)

```

```sh
deque([1, 2, 3, 4, 5, 6])

```

* **最大サイズ指定 (`maxlen`)**:
`maxlen` を指定すると，容量を超えた場合に古い要素が自動的に反対側から押し出されます．

```python
a = deque([1, 2, 3], maxlen=5)
b = [4, 5, 6]
a.extend(b)
print(a)
a.extendleft(b)
print(a)

```

```sh
deque([2, 3, 4, 5, 6], maxlen=5)
deque([6, 5, 4, 2, 3], maxlen=5)

```

* **スライス操作 (`:`) について**:
`deque` 自体はスライス (`queue[5:11]`) に対応していません．型のキャスト（`np.array`），`itertools.islice`，またはリスト内包表記を使用します．

```python
from collections import deque
import numpy as np
import itertools

queue = deque()
for num in range(21):
    queue.append(num)
print(queue)

# 1. 型のキャストを利用
print(np.array(queue))

# 2. itertools.isliceを利用
print(list(itertools.islice(queue, 5, 11)))

# 3. リスト内包表記を利用
print([queue[i] for i in range(5, 11)])

```

```sh
deque([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
[5, 6, 7, 8, 9, 10]
[5, 6, 7, 8, 9, 10]
[5, 6, 7, 8, 9, 10]

```
