# 環境のインストール
1. exeファイルを展開します
    - C:\oit\oitpy22\以下に，python，VS Code，ソースコード用フォルダが展開されます．
 
 
2. 環境の起動
    - デスクトップにあるOITpy22のショートカットまたは C:\oit\oitpy22\OITpy22.bat をダブルクリックして起動します．
 
 
3. ディレクトリ構造
    - C:\oit\oitpy22\SourceCode\以下のディレクトリ構造は次の通りです．新しい.pyファイルはSourceCodeフォルダに追加します．
      ```
      +[SourceCode]             <== ワーキングディレクトリ ("C:\oit\oitpy22\SourceCode")
      |
      |-+[mylibs]               <== 独自ライブラリ
      | |-+[myCapture]          <== 動画キャプチャ用のライブラリ関数
      | | |--
      | |
      | |-+[myPhysiology]       <== 生体信号関連のライブラリ関数
      | | |-+[learned_model]
      | | | |-+[haarcascades]   <== haarcascades用の学習済みファイル
      | | | | |--
      | | | |
      | | | |--lbfmodel.yaml    <== lbf顔パーツ検出用の学習済みファイル
      | | | |--README.md        <== 
      | | | |--shape_predictor_68_face_landmarks.dat  <== dlib顔パーツ検出用の学習済みファイル
      | | | |--
      | | |
      | | |--
      | |
      | |--
      |
      |--test_faceDetect.py     <== mylibsを使った顔検出サンプルpy
      |--test_myCapture.py      <== mylibsを使った動画キャプチャサンプルpy
      |--test_myMediapipe.py    <== mylibsを使ったmediapipeのサンプルpy
      |--                       <== プログラムを追加していく場所
      |
      ```
2. 実行ファイルの作り方とターミナルの起動
    - VS Codeの[エクスプローラー]メニューの[SOURCECODE]フォルダの右に示される

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
- It is O.K., if it is executed as follows.
  ```sh
  C:\\...\code> python sample_basic.py
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
