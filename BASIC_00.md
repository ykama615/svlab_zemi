<hr>

1. [環境の設定](README.md)
2. 基本概要（↓）
3. [カメラ，顔・手・ポーズ検出](BASIC_01.md)

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
  # -*- coding: utf-8 -*-
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
  # -*- coding: utf-8 -*-  
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
  # -*- coding: utf-8 -*-  
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
  # -*- coding: utf-8 -*-  

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
  # -*- coding: utf-8 -*-  

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
  # -*- coding: utf-8 -*-  

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
  # -*- coding: utf-8 -*-  
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
