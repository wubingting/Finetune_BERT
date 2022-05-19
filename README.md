# Finetune_BERT

Memory usage problem occurs when using the whole training dataset.

## With very small samples

limit the categories of dataset to [ "alt.atheism", "talk.religion.misc", "comp.graphics"]. The size of dataset is:
![image](https://user-images.githubusercontent.com/49680463/169280045-6a1c16a9-7b35-443a-afe3-605e90d1391a.png)

Use GPU with 85.9G MEM (about 13% shared by other people). The memory usage problem didn't occur. 
In this case, the usage of GPU MEM is usually 5-30G, maximal about 60G. 
![image](https://user-images.githubusercontent.com/49680463/169281157-e69e965f-6222-4434-8e40-9e0fffa821c2.png)

![image](https://user-images.githubusercontent.com/49680463/169281181-9d26d960-4b16-437a-b237-4dae91d89488.png)

## With whole training dataset
problems occur
Several possible solutions have been tried:
1. reduce the batch_size to 1
2. release GPU memory cache
3. remove differentiable variables
