# 読解解答可能性データセットへの適用

## 分類問題としてとらえる

読解解答可能性データセットをこれに当てはめる。

まずは0,1,2,3,4,5のラベルの識別課題として解く。結果は、きっとつらい。
回帰問題にすればまだましだろう。すくなくとも解ける・解けそう・解けそうにないが判断できれば。

データは、あるqid (question,answer)に対して passage (documents)が複数ある。

すべてをシャッフルしてtrain/test splitするのでなく、
qidでシャッフルして分割する。

question,answerがたがいに紛れ込まないため。

ただし、questionは異なるが同じanswerがいくつかのqaついにあるかは未確認。

```sh
data.score.value_counts()
0    20407
1     8550
5     8020
4     7119
3     6278
2     6277
Name: score, dtype: int64
```

ラベルの分布。粗い数字。

```sh
epoch: None     loss: 28.441026         accuracy: 0.1853        precision: 0.1735       recall: 0.1754  f1: 0.1379
epoch: 0        loss: 20.648578         accuracy: 0.4613        precision: 0.2990       recall: 0.3459  f1: 0.2908
epoch: 1        loss: 20.198934         accuracy: 0.4726        precision: 0.3757       recall: 0.3916  f1: 0.3601                                                                                          
epoch: 2        loss: 21.159628         accuracy: 0.4782        precision: 0.3528       recall: 0.3753  f1: 0.3468
epoch: 3        loss: 23.695038         accuracy: 0.4775        precision: 0.3920       recall: 0.3919  f1: 0.3889
epoch: 4        loss: 27.322239         accuracy: 0.4709        precision: 0.3762       recall: 0.3870  f1: 0.3788
epoch: 5        loss: 29.946543         accuracy: 0.4630        precision: 0.3891       recall: 0.3880  f1: 0.3883
epoch: 6        loss: 33.786729         accuracy: 0.4585        precision: 0.3940       recall: 0.3920  f1: 0.3910
epoch: 7        loss: 36.977728         accuracy: 0.4521        precision: 0.3995       recall: 0.3902  f1: 0.3922
epoch: 8        loss: 39.986943         accuracy: 0.4557        precision: 0.4030       recall: 0.3917  f1: 0.3935
epoch: 9        loss: 43.296225         accuracy: 0.4540        precision: 0.3931       recall: 0.3869  f1: 0.3881
epoch: 10       loss: 46.055529         accuracy: 0.4491        precision: 0.3949       recall: 0.3921  f1: 0.3903
epoch: 11       loss: 47.627651         accuracy: 0.4505        precision: 0.3980       recall: 0.3842  f1: 0.3862
```

F1-bestで、これ。
epoch: 8 loss: 39.986943 accuracy: 0.4557 precision: 0.4030 recall: 0.3917 f1: 0.3935


---
2023/02/06

sonoisa/sentence-bert-base-ja-mean-tokens-v2 で。

```sh
morioka@legion:~/bert-classification-tutorial$ poetry run python src_rcqa/train.py --model_name sonoisa/sentence-bert-base-ja-mean-tokens-v2
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at sonoisa/sentence-bert-base-ja-mean-tokens-v2 and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
epoch: None     loss: 29.579372         accuracy: 0.1356        precision: 0.1530       recall: 0.1730  f1: 0.0897
epoch: 0        loss: 20.683091         accuracy: 0.4752        precision: 0.3393       recall: 0.3505  f1: 0.3073
epoch: 1        loss: 20.234028         accuracy: 0.4826        precision: 0.3663       recall: 0.3819  f1: 0.3614
epoch: 2        loss: 20.674121         accuracy: 0.4793        precision: 0.3894       recall: 0.3851  f1: 0.3826
epoch: 3        loss: 23.099997         accuracy: 0.4580        precision: 0.3873       recall: 0.3835  f1: 0.3829
epoch: 4        loss: 27.103876         accuracy: 0.4721        precision: 0.3868       recall: 0.3857  f1: 0.3852
epoch: 5        loss: 29.898725         accuracy: 0.4561        precision: 0.3920       recall: 0.3893  f1: 0.3883
epoch: 6        loss: 32.663677         accuracy: 0.4611        precision: 0.4061       recall: 0.3932  f1: 0.3967
epoch: 7        loss: 37.900856         accuracy: 0.4477        precision: 0.4089       recall: 0.4007  f1: 0.3992
epoch: 8        loss: 40.981920         accuracy: 0.4498        precision: 0.4006       recall: 0.3889  f1: 0.3888
epoch: 9        loss: 44.520389         accuracy: 0.4409        precision: 0.4035       recall: 0.3873  f1: 0.3893
epoch: 10       loss: 45.469061         accuracy: 0.4528        precision: 0.3998       recall: 0.3916  f1: 0.3928
epoch: 11       loss: 49.025310         accuracy: 0.4475        precision: 0.4011       recall: 0.3856  f1: 0.3880
```

F1-bestで、これ。
epoch: 7        loss: 37.900856         accuracy: 0.4477        precision: 0.4089       recall: 0.4007  f1: 0.3992


----

2023-02-16

- evaluateごとにラベル出力する `--output_gold_and_pred`
- 学習後にF1最良モデルを出力する `--save_best_model`
  
``sh
$ bash src_rcqa/download.sh
$ poetry run python src_rcqa/prepare.py
$ poetry run python src_rcqa/train.py --save_best_model --output_gold_and_pred --epochs 12
epoch: None     loss: 28.441026         accuracy: 0.1853        precision: 0.1735       recall: 0.1754  f1: 0.1379
epoch: 0        loss: 20.648578         accuracy: 0.4613        precision: 0.2990       recall: 0.3459  f1: 0.2908
epoch: 1        loss: 20.198934         accuracy: 0.4726        precision: 0.3757       recall: 0.3916  f1: 0.3601
epoch: 2        loss: 21.137527         accuracy: 0.4782        precision: 0.3530       recall: 0.3761  f1: 0.3494
epoch: 3        loss: 23.753740         accuracy: 0.4730        precision: 0.3908       recall: 0.3903  f1: 0.3878
epoch: 4        loss: 27.274354         accuracy: 0.4728        precision: 0.3770       recall: 0.3879  f1: 0.3794
epoch: 5        loss: 31.159773         accuracy: 0.4632        precision: 0.3896       recall: 0.3885  f1: 0.3887
epoch: 6        loss: 35.091379         accuracy: 0.4597        precision: 0.3962       recall: 0.3957  f1: 0.3939
epoch: 7        loss: 38.528027         accuracy: 0.4585        precision: 0.4058       recall: 0.3959  f1: 0.3983
epoch: 8        loss: 42.428249         accuracy: 0.4615        precision: 0.4129       recall: 0.4019  f1: 0.4013
epoch: 9        loss: 46.191750         accuracy: 0.4664        precision: 0.4072       recall: 0.4001  f1: 0.4019
epoch: 10       loss: 49.087271         accuracy: 0.4601        precision: 0.4033       recall: 0.3987  f1: 0.3991
epoch: 11       loss: 50.833403         accuracy: 0.4606        precision: 0.4078       recall: 0.3998  f1: 0.4014
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [4:56:39<00:00, 1483.26s/it]
$ 
``

## 回帰問題としてとらえる

2023-02-15

文埋め込みから解答可能性スコアを予測する。
埋め込みモデルと回帰モデルは別にする。
埋め込みにはSentenceBERT、回帰モデルにはLightGBMを初手に用いる。

```bash
# qid 毎に tran-valid-test splitして document毎に展開　分類問題と同じ
poetry run python ./src_rcqa/prepare.py 
# X=<question, answer>, <pasasge>それぞれの埋め込みと y=解答可能性スコアを抽出
poetry run python ./src_rcqa/prepare_emb.py 
```

-------

# データのクリーニング

本来であれば先にすべきこと

- 解答が似ているものがないか?   test と trainの間に
- 設問が似ているものがないか?   test と trainの間に
- パッセージが似ているものがないか?   test と trainの間に
- 設問と回答のペアが似ているものがないか?    test と trainの間に

確認方法：
    - SentenceBERTでの埋め込みの近いものを眺める
    - elasticsearch (BM25)で似ているものを眺める
    - データセット作成時にケアされているものと信じる

http://www.cl.ecei.tohoku.ac.jp/rcqa/
http://www.cl.ecei.tohoku.ac.jp/publications/2018/master_thesis_m-suzuki.pdf
https://github.com/onikazu/ArticleSum/issues/61
https://www.anlp.jp/proceedings/annual_meeting/2018/pdf_dir/C4-5.pdf