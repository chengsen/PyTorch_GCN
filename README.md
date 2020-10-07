# Graph Convolutional Networks in PyTorch

PyTorch 1.6 and Python 3.7 implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

Tested on the cora/pubmed/citeseer data set, the code on this repository can achieve the effect of the paper.

## Benchmark

| dataset       | Citeseea | Cora | Pubmed |
|---------------|----------|------|--------|
| GCN(official) | 70.3     | 81.5 | 79.0   |
| This repo.    | 70.7     | 81.2 | 79.2   |

NOTE: The result of the experiment is to repeat the run 10 times, and then take the average of accuracy.

## Requirements
* PyTorch==1.6.0
* Python==3.7
* dgl==0.5.2 
* scipy==1.5.2 
* numpy==1.19.1 
* networkx==2.5

## Usage
```python
python train.py --dataset cora
```

## References
[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](http://xxx.itp.ac.cn/pdf/1609.02907.pdf)
