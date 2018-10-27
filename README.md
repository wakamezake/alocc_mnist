# ディープラーニングによる異常検知手法ALOCCを実装した
[Qiitaに書いた記事](https://qiita.com/kzkadc/items/334c3d85c2acab38f105)のコードです。

## 準備
ChainerとOpenCVを使います。  
インストール：
```bash
$ sudo pip install chainer opencv-python
```

## 実行
```bash
$ python train.py 設定ファイル 出力フォルダ [-g GPUID]
```

例：  
```bash
$ python train.py setting.json result
```
