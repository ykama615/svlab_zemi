# pythonの実行方法
 1. 対話（インタラクティブ）モード<br>
  入出力環境（PowerShell）を起動後にpythonと入力し，入力プロンプト（>>>）の後ろにコマンドを入力していく．<br>
  変数や条件分岐，繰り返し処理などを使用したものもファイルなどに記述することなく対話的に実行することができる．<br>
  終了時はexitコマンドを入力する．
  ```python
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
  ```python
  % python script1.py
  Hello python
  hoge hoge
  ```
 3. スクリプトモード(2)<br>
  その.pyファイルがスクリプトとして実行された場合にのみ実行する処理をまとめるif文を記述する．<br>
  if文のブロックには，呼び出す関数や実行する処理の手順をまとめて記述する．
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
  ```python
  % python script2.py
  Hello python
  fuga fuga
  hoge hoge
  ```
