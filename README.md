<hr>

1. 環境の設定（↓）
2. [基本概要](BASIC_00.md)
3. [カメラへのアクセスと動画処理](BASIC_01.md)
4. [顔と顔パーツの検出](BASIC_02.md)
5. [顔・手・ポーズ検出](BASIC_03.md)
6. [各種クラス・応用](BASIC_04.md)

<hr>

# 環境のインストール

 - 以下は配布環境のインストール方法です

0. 指定されたリンクから `pyXX.exe` をダウンロードします．
    > **Note**
    > - パスワードが必要な場合，大学のアカウントでのアクセスが必要な場合があるので注意しましょう
1. `py25.exe` ファイルを実行します.
    > **Note**
    > - 以下のような警告画面が表示された場合, `詳細情報` をクリックした後，`実行` をクリックしてください. <br>
    > <image src="../image/inst00.png" width="40%" height="40%"><br>
    > <image src="../image/inst01.png" width="40%" height="40%"><br>
2. `次へ(N)` をクリックします.<br>
  <image src="../image/inst02.png" width="40%" height="40%"><br>
  > インストール過程(1)...<br>
  > <image src="../image/inst03.png" width="40%" height="40%"><br>
  > インストール過程(2)...<br>
  > <image src="../image/inst04.png" width="40%" height="40%"><br>

3. このインストーラでは， `C:\oit\py25` に python実行環境（Python3.X + VSCode) をインストールし，ソースファイル ディレクトリとして `C:\oit\py25\source` と，デスクトップ上に実行のためのショートカット（下図）を設定します．.<br>
  <image src="../image/icon.png" width="10%" height="10%">

> **Note**
> Creating a link on the Desktop often fails. In that case, please run "C:\oit\py25en\py25en_start.bat" directly. It is possible to create the link manually, but DO NOT move anything in the `py25en` folder!)

#### Installed folder structure
- This environment is installed to "C:\oit\py25en\" and its inside is included the following.
  - **C:\oit\py25en\source**: the working directory for saving the source code (Directory "py25en" NEED NOT touch)
  - **C:\oit\py25en\\_tmp_**: NEED NOT touch
  - **C:\oit\py25en\VSCode**: NEED NOT touch, Visual Studio Code
  - **C:\oit\py24\WPy64-312101**: NEED NOT touch, Python3.12.10amd64 (WPy64-312101)
  - **C:\oit\py24\py25en_start.bat**: bat file to start this environment up 

### :o:Checkpoint(Start the environment 1)
- Start the environment from "py25en_start" icon on the Desktop (or C:\oit\py25en\py25en_start.bat).
- **If the following warning pops up...**
  - **CHECK** the "Trust the authors..." box out
  - CLICK the **"YES"** button <br>
    <image src="../image/trust_vsws.png" width="50%" height="50%">

### :o:Checkpoint(Start the environment 2)
- **If the location of the EXPLORER does not be the `souce` folder(SOURCE), you have to open the `C:\oit\py25en\sorce\` from the [File]-[Open Folder] menu.** <br>
  <image src="../image/vsws_explorer.png" width="50%" height="50%">
- **If the terminal window has not shown, please open it from the [Terminal]-[New Terminal] menu.** <br>
  <image src="../image/vsws_tmenu.png" width="50%" height="50%">
- Please confirm Python modules by inputting the `pip list` command in the terminal window.<br>
  <image src="../image/vsws_piplist.png" width="50%" height="50%">

### :o:Checkpoint(Start the environment 3)
- **When you select the `.py` file in the Explorer window, if the status bar shows `Select Python Interpreter` ...** <br>
  <image src="../image/vs_setting01.png" width="50%" height="50%">
- **you need to set the path to `python.exe` (`C:\oit\py25en\WPy64-312101\python\python.exe`) as shown below.** <br>
  <image src="../image/vs_setting02.png" width="50%" height="50%"><br>
  <image src="../image/vs_setting03.png" width="50%" height="50%">
- **To check whether the interpreter is set correctly, use the status bar or the command palette (Select Interpreter).**
  <image src="../image/vs_setting04.png" width="50%" height="50%"><br>
  <image src="../image/vs_setting05.png" width="50%" height="50%">

### :o:Checkpoint(Run python code with VSCode)
- Please confirm how to execute the sample Python code with VSCode.
  - Open the "sample1.py" file with Double Click in [source] folder of the explorer menu.<br>
    <image src="../image/vs_sample1.png" width="100%" height="100%">
  - Open the terminal window if it has not appeared.<br>
    <br>
    > **Note** The current Working directory shown in the terminal window has to be the same as the file's location to execute. <br>
    > **Note** You have to change the directory using the 'cd' command, in case the current directory shown in the terminal window is different from the source code directory. <br>
    <br>
  - Please confirm that the Python code can execute in the terminal window.
    ```sh
    C:\oit\py25en\source> python sample1.py
    ```
    <br>
    
    > **Note** The program is executable with the run button, but **we suggest executing with the command line**. <br>
    > <image src="../image/vs_sample2.png"><br>
    <br>

  - The following are running results successfully.<br>
    <image src="../image/vs_sample3.png"><br>

### :o: Practice
- Give it a try to run the ”hello_opencv.py”.
  - It is the sample of reading and showing an image file with the cv2 library.
  - The window is closed if any button is pressed.
- Give it a try to run the "show_video.py"
  - Create a new file" named "show_video.py"<br>
    <image src="../image/create_newfile.png" width="50%" height="50%"><br>
  - The following code is the sample of capturing from the camera and showing frames with the cv2 library.
    - Please copy & paste this code to "show_video.py".
    - The window is closed if \'q\' button is pressed.
    ```
    import cv2
    
    dev = 0
    
    def main():
        cap = cv2.VideoCapture(dev)
        ht = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        wt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(ht," x ", wt)
    
        while cap.isOpened():
            ret, frame = cap.read()
    
            if ret==False or cv2.waitKey(1) == ord('q'):
                break
    
            cv2.imshow("video", frame)
        
        cap.release()
        cv2.destroyAllWindows()
    
    if __name__=='__main__':
        main()
    ```
  
  > **Note** The latest usage of the Mediapipe is able to be learned in another section.

  <br>



1. exeファイルを展開します
    - C:\oit\py25\以下に，python，VS Codeが，<!-- C:\oit\home\以下にpy24フォルダ（ソースコード用フォルダ）-->が展開されます．
 
 
2. 環境の起動
    - デスクトップにあるpy25_startのショートカットまたは C:\oit\py25\py25_start.bat をダブルクリックして起動します．
 
 
3. ソースコードディレクトリ構造
    - C:\oit\py25\source\以下のディレクトリ構造は次の通りです．新しい.pyファイルはsourceフォルダに追加します．
      ```
      +[source]           <== ワーキングディレクトリ ("C:\oit\py25")
      |
      |-+[img]            <== 画像用フォルダ
      | |-+[standard]     <== 標準画像用フォルダ
      |   |-+[mono]       <== グレースケール画像用フォルダ
      |   | |-(files)
      |   |-(files)
      |-+[learned_models] <== 学習済み物体・人検出ファイル格納フォルダ
      | |-+[haarcascades]
      | | |-(files)
      | |-+[mediapipe]
      | | |-(files)
      | |-(files)
      |-sample1.py
      |-sample2.py
      |-sample3.py
      |-sample4.py
      |-sample5.py
      |-sample6.py
      |-sample7.py
      |-sample8.py
      |-(files)
      ```

4. 実行ファイルの作り方とターミナルの起動
    - VS Codeの[エクスプローラー]メニューの[SOURCE]フォルダの右に示される，新規ファイルの追加，新規フォルダの追加でファイルやフォルダの追加を行います．＊画像中の[SOURCECODE]を[SOURCE]に読み替えてください<br>
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
    - [実行] 以下のコマンドを入力するか，VS Code右上の再生ボタンを押してプログラムを実行します．<br>
      ※カレントディレクトリに注意
      ```sh
      python sample_basic.py
      ```
      
    - 以下の実行結果が出力されれば成功です．
      ```sh
      C:\oit\py25\source> python sample_basic.py
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
      img = cv2.imread('./img/standard/Mandrill.bmp') # read image file
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
