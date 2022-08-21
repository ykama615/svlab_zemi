# pythonの実行方法
 1. 対話（インタラクティブ）モード<br>
  入出力環境（PowerShell）を起動後にpythonと入力し，入力プロンプト（>>>）の後ろにコマンドを入力していく．<br>
  変数や条件分岐，繰り返し処理などを使用したものもファイルなどに記述することなく対話的に実行することができる．<br>
  終了時はexitコマンドを入力する．
  ```sh
  % python
  Python 3.X.X (tags/v3.X.X:...) [MSC v.XXXX 64 bit (AMD64)] on win32
  Type "help", "copyright", "credits" or "license" for more information.
  >>> print("Hello python")
  Hello python
  >>> for i in range(5):
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
  .pyファイルにコマンドをまとめて記述し，pythonコマンドを使ってプログラムを実行する．<br>
  プログラムはファイルの上から順に実行される．
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
  その.pyファイル（モジュールと呼ぶ）がスクリプトとして実行された場合にのみ実行する処理をまとめるif文を記述する．<br>
  if文のブロックには，呼び出す関数や実行する処理の手順をまとめて記述する．他のファイルからimportされた場合，if文のブロックは無視される．
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
  ・仮引数に初期値を設定して，実引数未指定で利用したり（実引数の指定も可能）
  ・仮引数名を使って実引数を指定することで引数の順序を無視したり
 でき，戻り値では，
  ・複数の戻り値を指定してリストで受け取ったり，
  ・戻り値の受け取りをそれぞれ変数で受け取ったり（不要な戻り値は_で飛ばすことも可能）
 できる
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

    return sum, st, ed

  if __name__=='__main__':
    func_msg(3)
    all = func_sum(1,5)
    sm, _, e = func_sum(ed=3,st=1)
    print(all)
    print(sm, e)
  ```
  ```sh
  hoge        # func_msgの出力
  hoge
  hoge
  (15, 1, 5)  # func_sumの戻り値をリストで受け取る
  6 3         # func_sumの第1と第3戻り値を変数で受け取る
  ```
