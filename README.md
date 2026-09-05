<hr>

**講義ノート・ライブラリ一覧**

<b>➡基礎編</b>
1. 環境の設定（↓）
2. [基本概要](articles/basic/BASIC_00.md)
3. [カメラへのアクセスと動画処理](articles/basic/BASIC_01.md)
4. [顔と顔パーツの検出](articles/basic/BASIC_02.md)
5. [顔・手・ポーズ検出](articles/basic/BASIC_03.md)
6. [2つのベクトルのなす角とベクトル演算](articles/basic/BASIC_FP01.md)

<details><summary><b>キャプチャ（3項目）</b></summary>

7. [動画画像処理 (`my_cap_av2.py`)](articles/lecnote/lecnote_cap01.md)
8. [Intel RealSense 画像処理 (`my_rs_cap.py`)](articles/lecnote/lecnote_cap02.md)
9. [Orbbec Femto Bolt 画像処理 (`my_bolt_cap.py`)](articles/lecnote/lecnote_cap03.md)
</details>

<details><summary><b>検出・推定（4項目）</b></summary>

10. [MediaPipe統合処理 (`my_mediapipe_n.py`)](articles/lecnote/lecnote_dt01.md)
11. [OpenMMLab 顔検出・キーポイント抽出 (`my_mmface.py`)](articles/lecnote/lecnote_dt02.md)
12. [OpenMMLab 統合姿勢推定 (`my_mmpose.py`)](articles/lecnote/lecnote_dt03.md)
13. [dlib 顔検出・68点ランドマーク抽出 (`my_dlib.py`)](articles/lecnote/lecnote_dt04.md)
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

# Python・VSCode 画像処理環境 セットアップガイド

## 概要

* このページでは、画像処理実習に向けた Python および VSCode の環境構築手順を説明します。
* ポータブル版を使用するため、PCのレジストリなど環境を汚すことはありません。

> **メモ** 環境が不要になった場合は、フォルダごと削除するだけで完全にアンインストールできます。

### :green_square: インストールされる環境の詳細

* Python 3.12.10 (WinPython)
* 含まれるパッケージ
* numpy == 1.26.4
* mediapipe == 0.10.33
* opencv-python == 3.4.18.65


* ※用意されたポータブル環境を使用しない場合は、上記の要件を満たす仮想環境をご自身で構築してください。


* Visual Studio Code 1.113.0 (ポータブル版)

## 前提条件

* Windows 10 または 11
* 内蔵カメラまたはUSBカメラ
* アンチウイルスソフトの無効化（またはアンインストール）、および Windows セキュリティの「スマート アプリ コントロール」をオフにする
* セキュリティソフト等によって、インストーラーやバッチファイルが削除されてしまう場合があります。

## インストーラーを使用した Python・VSCode のセットアップ

### :green_square: 環境のインストール手順

* 指定されたフォルダにアクセスし、フォルダ内にある `README_py26.pdf` などの指示に従って環境をインストールしてください。
* [フォルダへのリンク](https://oskit-my.sharepoint.com/:f:/g/personal/yoshiyuki_kamakura_oit_ac_jp/IgCmGGWyRvidTIrSIwzoav-gAWJziWVS6J4E8Qa3WLxZ6wE?e=Xbl8Cj)

> **メモ**
> * フォルダへのアクセスにはパスワードが必要です（パスワードは別途通知されます）。
> * アンチウイルスソフトが有効な場合、実行ファイルやバッチファイルが正常に動作しないことがあります。
> 
> 

### :green_square: インストール後のフォルダ構造

* フォルダ構造は以下の通りです。
* **C:\oit\home\python**: ソースコードを保存する作業ディレクトリ
* **C:\oit\py26\**: **※このフォルダ内を変更・編集しないでください**
* **C:\oit\py26\py26_start.bat**: 環境を起動するためのバッチファイル


> **メモ**
> * デスクトップにショートカットが作成されています。
> * ショートカットが見つからない場合は、バッチファイル（**py26_start.bat**）から直接起動してください。
> 
> 



### :o: チェックポイント（環境の起動 1）

* デスクトップ上のアイコン（または `C:\oit\py26\py26_start.bat`）から環境を起動します。
* **次のような警告画面が表示された場合：**
* 「信頼する...」のチェックボックスに **チェックを入れます**
* **「はい (YES)」** ボタンをクリックします 






### :o: チェックポイント（環境の起動 2）

* **VSCode のエクスプローラーに `python` フォルダが開いていない場合は、[ファイル (File)] - [フォルダーを開く (Open Folder)] から `C:\oit\home\python\` を開いてください。** 



* **ターミナル画面が表示されていない場合は、[ターミナル (Terminal)] - [新しいターミナル (New Terminal)] メニューから開いてください。** 




### :o: チェックポイント（環境の起動 3）

* **エクスプローラーで `.py` ファイルを選択したとき：**
* **設定が正しい場合：** ステータスバーにPythonのバージョンが正しく表示されます。

* **設定が正しくない（または未設定の）場合：** ステータスバーのPythonバージョン（または `Python インタープリターの選択`）をクリックし、**「インタープリターを選択」** -> **「参照... (Browse...)」** の順にクリックして、以下の `python.exe` を選択してください。
> **パス:** `"C:\oit\py26\WPy64-312101\python\python.exe"`
> 





## :red_square: 動作確認（実践）

* カメラ映像を表示するサンプルプログラム `show_video.py` を動かしてみましょう。
* 「`show_video.py`」という名前の新しいファイルを作成します。






* 以下のコードは、カメラから映像をキャプチャし、OpenCV ライブラリを使用してフレームを表示するサンプルです。
* コードをコピーして `show_video.py` に貼り付けてください。
* キーボードの `q` キーを押すとウィンドウが閉じます。


```python
import cv2

dev = 0

def main():
    cap = cv2.VideoCapture(dev)
    ht = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    wt = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print(ht, " x ", wt)

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False or cv2.waitKey(1) == ord('q'):
            break

        imshow_name = "video"
        cv2.imshow(imshow_name, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

```


* ターミナルウィンドウで Python コードが実行できることを確認します。
```sh
C:\oit\home\python> python show_video.py

```
