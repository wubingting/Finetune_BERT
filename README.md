# Finetune_BERT

The code is trained and evaluated for very small samples of training dataset (**BERT_Hierarchical.ipynb**) and the whole training dataset(**BERT_Hierarchical_Large.py**). 

The code is to fine-tune BERT by computing the pooled result from the output of each segment of the long document. Details see the .ipynb file.

Memory usage problem occurs when using the whole training dataset (**BERT_Hierarchical_Large.py**).

## File structure
To run the code from terminal, **BERT_Hierarchical.ipynb** is first converted to **BERT_Hierarchical.py** file and then run on GPU, results are in **output_small_sample.txt**.

The accuracy changes for the five epochs are shown in **BERT_Hierarchical_Model.png**.
Because of the small samples the accuracy increases very slowly.

Code to train the large dataset (already modified with possible solutions like set batch_size to 1): **BERT_Hierarchical_large.ipynb**

Details of memory message problem when using the whole training dataset: **message.txt**

## With very small samples (BERT_Hierarchical.ipynb)

limit the categories of dataset to [ "alt.atheism", "talk.religion.misc", "comp.graphics"]. The size of dataset is:
![image](https://user-images.githubusercontent.com/49680463/169280045-6a1c16a9-7b35-443a-afe3-605e90d1391a.png)

Use GPU with 85.9G MEM (about 13% shared by other people). The memory usage problem didn't occur. 
In this case, the usage of GPU MEM is usually 5-30G, maximal about 60G. 
![image](https://user-images.githubusercontent.com/49680463/169284400-1cd421f2-8440-480a-abd1-f937ba986dc2.png)

![image](https://user-images.githubusercontent.com/49680463/169281181-9d26d960-4b16-437a-b237-4dae91d89488.png)

## With whole training dataset (BERT_Hierarchical_large.ipynb). Memory usage problems occurs. For more details, see error message.txt. 
![image](https://user-images.githubusercontent.com/49680463/169285661-5a3142aa-e1a5-4f38-a184-b2c8dba3b02d.png)

Several possible solutions have been tried together:
1. reduce the batch_size to 1
2. release GPU memory cache after every epoch
torch.cuda.empty_cache()
3. remove differentiable variables. 
In the training and evaluation phase, losses.append(float(loss.item())) instead of loss.item().

but the problem still occurs
![image](https://user-images.githubusercontent.com/49680463/169288024-b93de1f4-9351-4ab7-bb1a-046c1e4a7e3f.png)
