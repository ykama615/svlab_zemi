<hr>

1. 環境の設定（↓）
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. [応用※未整理](BASIC_04.md)

<hr>

# 環境のインストール

 - 以下は配布環境のインストール方法です（2023r2）

1. exeファイルを展開します
    - C:\oit\py23r2\以下に，python，VS Code，ソースコード用フォルダが展開されます．
 
 
2. 環境の起動
    - デスクトップにあるpy23_startのショートカットまたは C:\oit\py23r2\py23_start.bat をダブルクリックして起動します．
 
 
3. ディレクトリ構造
    - C:\oit\py23\SourceCode\以下のディレクトリ構造は次の通りです．新しい.pyファイルはSourceCodeフォルダに追加します．
      ```
      +[SourceCode]             <== ワーキングディレクトリ ("C:\oit\oitpy22\SourceCode")
      |
      |-+[.vscode]              <== （編集不要）ワークスペース設定のフォルダ
      | |-launch.json           <== （編集不要）ローンチ用jsonファイル
      |
      |-+[image]                <== 画像用フォルダ
      | |-swan.jpg
      | +[standard]             <== 標準画像用フォルダ
      |   |-+[mono]             <== グレースケール画像用フォルダ
      |   | |-- 
      |   |
      |   |--
      |
      |-+[samples]              <== サンプルファイル
      | |--
      |
      ```

4. 実行ファイルの作り方とターミナルの起動
    - VS Codeの[エクスプローラー]メニューの[SOURCECODE]フォルダの右に示される，新規ファイルの追加，新規フォルダの追加でファイルやフォルダの追加を行います．<br>
        ![fig001](./fig001.png)
    - ターミナルが開いていない場合，[ターミナル]メニューから[新しいターミナル]を選択してターミナルを起動します．<br>
        ![fig002](./fig002.png)

5. サンプルプログラムの実行
    - 新規ファイルとして[sample_basic.py]を作成し，以下のコードを入力してみましょう．
      ```python
      # sample_basic.py
      sum = 0
      for i in range(10):
        sum = sum + i
        print(str(i) + ":" + str(sum))
      if sum <= 30 :
        print("sum is under 30")
      elif sum <= 50 :
        print("sum is between 30 and 50")
      else:
        print("sum is over 50")
      ```
    - [実行] 以下のコマンドを入力するか，VS Code右上の再生ボタンを押してプログラムを実行します．
      ```sh
      python sample_basic.py
      ```
      
    - 以下の実行結果が出力されれば成功です．
      ```sh
      C:\oit\py23r2\SourceCode> python sample_basic.py
      0:0
      1:1
      2:3
      3:6
      4:10
      5:15
      6:21
      7:28
      8:36
      9:45
      sum is between 30 and 50
      ```

6. OpenCVライブラリのサンプルの実行
    - 新規ファイルとして[sample_cv2.py]を作成し，以下のコードを入力して実行してみましょう．
      ```python
      # sample_cv2.py
      import cv2
      img = cv2.imread('./image/standard/Mandrill.bmp') # read image file
      if img is None: # maybe Path is wrong
          print("image file is not opened.")
          exit(1)
      bimg = cv2.GaussianBlur(img, (51,51), 5) # gaussian filter (size=(51,51),sigma=5)
      cv2.imshow('img',img)
      cv2.imshow('blur img',bimg)
      cv2.waitKey(0) # pause until press any key
      cv2.destroyAllWindows # close all cv2's windows
      ```

    - 実行結果は以下の通りで， 2つ目のウィンドウは[Mandrill.bmp]をブラー（平滑化）した結果となります．<br>
        ![fig003](./fig003.png)
