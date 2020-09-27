# Graph Convolutional Networks in PyTorch

PyTorch implementation of Graph Convolutional Networks (GCNs) for semi-supervised classification [1].

Tested on the cora/pubmed/citeseer data set, the code on this repository can achieve the effect of the paper.

## Requirements
* dgl==0.5.2 
* scipy==1.5.2 
* numpy==1.19.1 
* PyTorch==1.6.0 
* networkx==2.5 s

## Usage
python train.py --dataset cora

## References
[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](http://xxx.itp.ac.cn/pdf/1609.02907.pdf)
