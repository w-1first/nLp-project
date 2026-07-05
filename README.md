下面是作业的代码结构
nlp_project/
README.md
requirements.txt
run_all.py                  # 运行所有模型
further_experiments.py      # 进一步实验（学习率/Pooling/Dropout）
add_bert_metrics.py         # BERT 模型per-class指标计算
add_trad_metrics.py         # 传统 ML per-class指标计算
data/
   emotion.jsonl           # 数据集（20,000样本，6类情感）
   src/
    utils.py                # 工具函数（数据加载/指标/训练/早停）
    data_analysis.py        # 数据分析与可视化
    download_embeddings.py  # GloVe词向量下载与加载
    train_traditional.py    # TF-IDF+LR/SVM/NB
    train_cnn.py            # CNN
    train_rnn.py            # RNN
    train_lstm.py           # LSTM/BiLSTM/BiLSTM+Attention
    train_transformer.py    # Transformer Encoder（从零搭建）
    train_bert.py           # BERT 微调（冻结/全量）
    error_analysis.py       # 混淆矩阵与错误案例分析
    results/
    all_metrics.csv         # 所有模型对比总表
    per_class_metrics.csv   # 各类别 Precision/Recall/F1
    traditional_metrics.csv # 传统 ML 结果
    lr_comparison.csv       # 学习率对比实验
    pooling_comparison.csv  # Pooling 方法对比实验
    dropout_comparison.csv  # Dropout 对比实验
    figures/                # 混淆矩阵/训练曲线/对比图
    结果txt
    models/                  # 模型权重文件（.pth）



数据集
data/emotion.jsonl`，每行一个 JSON 对象：


6 类情感
joy (33.8%)
sadness (29.0%)
anger (13.5%)
fear (11.9%)
love (8.2%)
surprise (3.6%)
