<hr>

1. [環境の設定](README.md)
2. 基本概要（↓）
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)

<hr>


# pythonの実行方法
 1. 対話（インタラクティブ）モード<br>
  入出力環境（PowerShell）を起動後にpythonと入力し，入力プロンプト（>>>）の後ろにコマンドを入力していきます．<br>
  変数や条件分岐，繰り返し処理などを使用したものもファイルなどに記述することなく対話的に実行することができます．<br>
  終了時はexitコマンドを入力します．
  ```sh
  % python
  Python 3.X.X (tags/v3.X.X:...) [MSC v.XXXX 64 bit (AMD64)] on win32
  Type "help", "copyright", "credits" or "license" for more information.
  >>> print("Hello python")
  Hello python
  >>> for i in range(5): # range(5)は0～4までのリストを生成
  ...   print(i)  # print文の前にスペースが必要
  ... # 空のEnterでfor文のブロックを抜ける
  0
  1
  2
  3
  4
  >>> exit(0) #対話モードの終了
  %
  ```
 
 2. スクリプトモード(1)<br>
  .pyファイルにコマンドをまとめて記述し，pythonコマンドを使ってプログラムを実行します．<br>
  プログラムはファイルの上から順に実行されます．
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
 
 3. スクリプトモード(2)<br>
  その.pyファイル（モジュールと呼ばれます）が「スクリプトとして実行された場合にのみ実行する処理」をまとめるif文を記述します．<br>
  if文のブロックには，呼び出す関数や実行する処理の手順をまとめて記述します．他のファイルからimportされた場合，if文のブロックは無視されます．
  ```python
  # script2.py
  def func_hoge(): # 関数hoge
    print("hoge hoge")
  
  def func_fuga(): # 関数fuga
    print("fuga fuga")
  
  if __name__=='__main__': # スクリプトとして実行された場合にのみ実行する処理をまとめるif文
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
  
# 関数の引数と戻り値
 関数は，引数では，
 
  - 仮引数に初期値を設定して，実引数未指定で利用したり（実引数の指定も可能）
  - 仮引数名を使って実引数を指定することで引数の順序を無視したり
 
 でき，戻り値では，
 
  - 複数の戻り値を指定してリストで受け取ったり，
  - 戻り値の受け取りをそれぞれ変数で受け取ったり（不要な戻り値は_で飛ばすことも可能）
 
 できます．
 
  ```python
  # script3.py
  def func_msg(num, str="hoge"):
    for i in range(num):
        print(str)

  def func_sum(st, ed):
    sum = 0
    for i in range(st, ed+1):
      sum = sum + i

    return [sum, st, ed] # []を付けない場合タプルになる

  if __name__=='__main__':
    func_msg(3)
    all = func_sum(1,5)
    sm, _, e = func_sum(ed=3,st=1)
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

# クラス
  ```python
  # script4.py

  class Twice:
    cnum = 10
    def __init__(self):
      print("constructor")

    def twice(self):
      print(self.cnum*2)

    def setnum(self, num):
      self.cnum = num

  if __name__=='__main__':
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

# 大域変数
  ```python
  # script5.py

  def twice():
    global gnum
    print(gnum*2)

  def main():
    global gnum
    gnum = 10 # gnumはここで定義されている
    twice()

  if __name__=='__main__':
      global gnum
      # print(gnum)-> NameError: name 'gnum' is not defined
      main()
      print(gnum)
  ```
  ```sh
  % python script5.py
  20
  10
  ```

# List，Tuple，辞書
  ```python
  # script6.py
  # -*- coding: utf-8 -*-  

  #所謂配列リスト
  lst = ['子','丑','寅','卯']

  #タプルはイミュータブルで要素の書き換えができない[1]
  tpl = ('子','丑','寅','卯')

  #辞書はキーを使って要素にアクセス（キーはイミュータブル）[2]
  dic = {'十二支': ['子','丑','寅','卯'],'十二月':['睦月', '如月', '弥生']}

  #集合はただのデータの並び（インデックスなし）
  stt = {'1月', '2月', '3月'}

  #print(dic[0][0]) [2]辞書にインデックスはない
  print(tpl[0], lst[0], dic['十二支'][0])

  #tpl.append('猫') [1]要素の書き換えはできない
  lst.append('猫')
  dic['十二支'].append('猫')

  print(tpl, lst, dic['十二支'])
  print(stt)
  
  #辞書の追加
  dic_a = {'十二月':['睦月', '如月', '弥生']}
  dic_b = {'十二刻':['子','丑','寅','卯']}
  dic.update(**dic_a, **dic_b) # 複数の辞書を追加したい場合は ** を付ける

  print('辞書追加',dic)
  ```
  ```sh
  % python script6.py
  子 子 子
  ('子', '丑', '寅', '卯') ['子', '丑', '寅', '卯', '猫'] ['子', '丑', '寅', '卯', '猫']
  {'2月', '1月', '3月'}
  辞書追加 {'十二支': ['子', '丑', '寅', '卯', '猫'], '十二月': ['睦月', '如月', '弥生'], '十二刻': ['子', '丑', '寅', '卯']}
  ```

# for文，内包標記
  ```python
  # script7.py

  lst = ['子','丑','寅','卯']
  tpl = ('子','丑','寅','卯')
  dic = {'十二支': ['子','丑','寅','卯'], '十二月':['睦月', '如月', '弥生']}

  for i in range(len(lst)):
    print(lst[i])

  tmp = [lst[i] for i in range(len(lst))] #内包標記
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

  # モジュールのimportとパッケージ
  .pyのファイル名はモジュール名として扱われ，importによってモジュールの読み込みが可能です．<br>
  .pyファイルをフォルダにまとめ，\_\_init\_\_.pyファイルを入れることでパッケージ（ライブラリ？）として扱うことができます．
  ```python
  # mul_module.py
  def twice(num):
  print(num*2)
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
  \_\_init\_\_.pyの例は，
  ```python
  from モジュール名 import *
  ```

  # collection（コンテナデータ型）モジュールの deque
 リストより高速，高機能なデータ構造として，「deque(デック)」（double-ended queue）があります．<br>
 数値だけでなく，配列(リスト)や画像など様々なデータを格納でき，格納と取り出しが非常に高速です．下記のサンプルを実行して結果を確認しましょう．<br>
 ```python
 # deque_sample.py
 import numpy as np
 from collections import deque # dequeの利用に必要

 def main():
   queue = deque()
   for num in range(10, 21, 2):
     queue.append(num) #末尾に追加
   print(queue)

   for num in range(11, 20, 2):
     queue.appendleft(num) #先頭に挿入（左に追加）
   print(queue)

   queue.insert(5, 0) #添字4と5の間に0を挿入
   print(queue)

   queue.reverse() #要素を逆順にする
   print(queue)

   print(queue.pop()) #末尾からpop
   print(queue)

   print(queue.popleft()) #先頭からpop
   print(queue)

   for num in queue: #すべての要素を参照
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

 - 要素を1つ先頭に追加する appendleft() ，末尾に追加する append() がありますが，それ以外に複数の要素を末尾に連結する extend() ， extendleft もあります

  ```python
  a = deque([1,2,3])
  b = [4,5,6] # b=deque([4,5,6])でも可
  a.extend(b)
  print(a)
  ```
  ```sh
  deque([1,2,3,4,5])
  ```

 - dequeではキューの最大サイズを指定することができ，その場合その数を超えて要素を格納すると，あふれた分は反対側から押し出されます

  ```python
  a = deque([1,2,3], maxlen=5)
  b = [4,5,6]
  a.extend(b)
  print(a)
  a.extendleft(b)
  print(a)
  ```
  ```sh
  deque([2, 3, 4, 5, 6], maxlen=5)
  deque([6, 5, 4, 2, 3], maxlen=5)
  ```
