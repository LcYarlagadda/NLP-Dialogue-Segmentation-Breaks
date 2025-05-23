# 582Final
Group member: Ming Zhu, Lakshmi Chandrika Yarlagadda, Irfan Tekdir, Yichao Chen, Sai Chandana Priya

# Project Overview
In this paper, we delve into the task of classifying conversation segment breaks using various Natural Language Processing (NLP) models. We leverage the rich textual data within the GUIDE dataset to identify these transitions. We have worked with a couple of baseline models alongside some advanced models like SpanBERT and RoBERTa to assess their effectiveness in dialogue segmentation. We further experiment with optimization techniques to refine model performance. This analysis gives some insights for the future advancements in dialogue understanding and the development of more sophisticated conversation analysis systems.

# Test
Simply run our best model: ```python SpanBERT.py```

# Explaintion 
The utils folder contains the codes we used to manipulate the training data.

The experiment folder contains the codes we used to conduct experiments.

For our Best result we use SpanBERT for model, and use "user + intent + text" about data feature, dataset: doubled train, and here is our result:

lr: 2e-5 batch size: 32 Epoch: 20 optimizer: AdamW
|----|precision|recall|f1-score|support|
|----|----|----|----|----|
|0|0.90|0.90|0.90|895|
|1|0.78|0.78|0.78|395|
|accuracy|----|----|0.87|1290|
|macro avg|0.84|0.84|0.84|1290|
|weighted avg|0.87|0.87|0.87|1290|