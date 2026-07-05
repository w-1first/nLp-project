# 下面是我的作业的代码结构

```
nlp_project/
  README.md
  requirements.txt
  run_all.py                  #运行所有模型
  further_experiments.py      #进一步实验（学习率 / Pooling / Dropout）
  add_bert_metrics.py         #BERT per-class指标
  add_trad_metrics.py         #ML per-class指标

  data/
    emotion.jsonl            #数据集（20,000样本，6类情感）

  src/
    utils.py                 #工具函数
    data_analysis.py         #数据分析与可视化
    download_embeddings.py   #GloVe词向量下载

    train_traditional.py     #TF-IDF+LR/SVM/NB
    train_cnn.py             #CNN
    train_rnn.py             #RNN
    train_lstm.py            #LSTM/BiLSTM/Attention
    train_transformer.py     #Transformer Encoder
    train_bert.py            #BERT（冻结/全量微调）

    error_analysis.py        #错误分析+混淆矩阵

  results/
    all_metrics.csv
    per_class_metrics.csv
    traditional_metrics.csv
    lr_comparison.csv
    pooling_comparison.csv
    dropout_comparison.csv

    figures/                 #运行生成的所有的图表结果
    结果/                    #控制台的信息复制成txt保存的

  models/                    #模型权重（.pth）
```

---

#数据来源

```
data/emotion.jsonl
```


---

#分类

- joy (33.8%)
- sadness (29.0%)
- anger (13.5%)
- fear (11.9%)
- love (8.2%)
- surprise (3.6%)
