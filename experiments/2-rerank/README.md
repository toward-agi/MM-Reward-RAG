# 备忘录

这个文件夹实现的主要是在这四个数据集的hard-negatives上，使用reranker进行推理/测试的代码；

思路：
1. 选择top-k的hard-negatives用来测试，这个不用太大，要不然做不完，inference太多了；
2. 每个数据集实现一个策略，然后统一调不同的reranker的接口，得到rerank的结果，再通过一个统一的eval function测试得到结果就好了；
