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
 2. スクリプトモード<br>
  .pyファイルにコマンドをまとめて記述し，プログラムを実行する，
  ```python
  ## script.py
  # -*- coding: utf-8 -*-
  print("Hello python")
  
  def func_hoge():
    print("hoge hoge")
  
  def func_fuga():
    print("fuga fuga")
  
  func_hoge()
  ```
  出力結果
  ```python
  % python script.py
  Hello python
  hoge hoge
  ```
  
