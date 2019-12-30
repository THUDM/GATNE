# GATNE

### [Project](https://sites.google.com/view/gatne) | [Arxiv](https://arxiv.org/abs/1905.01669)

Representation Learning for Attributed Multiplex Heterogeneous Network.

[Yukuo Cen](https://sites.google.com/view/yukuocen), Xu Zou, Jianwei Zhang, [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jingren Zhou](http://www.cs.columbia.edu/~jrzhou/), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/)

Accepted to KDD 2019 Research Track!

## Prerequisites

- Python 3
- TensorFlow >= 1.8 (or PyTorch)

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/GATNE
cd GATNE
```

Please install dependencies by

```bash
pip install -r requirements.txt
```

### Dataset

These datasets are sampled from the original datasets.

- Amazon contains 10,166 nodes and 148,865 edges. [Source](http://jmcauley.ucsd.edu/data/amazon)
- Twitter contains 10,000 nodes and 331,899 edges. [Source](https://snap.stanford.edu/data/higgs-twitter.html)
- YouTube contains 2,000 nodes and 1,310,617 edges. [Source](http://socialcomputing.asu.edu/datasets/YouTube)
- Alibaba contains 6,163 nodes and 17,865 edges.

### Training

#### Training on the existing datasets

You can use `./scripts/run_example.sh` or `python src/main.py --input data/example` or `python src/main_pytorch.py --input data/example` to train GATNE-T model on the example data. (If you share the server with others or you want to use the specific GPU(s), you may need to set `CUDA_VISIBLE_DEVICES`.) 

If you want to train on the Amazon dataset, you can run `python src/main.py --input data/amazon` or `python src/main.py --input data/amazon --features data/amazon/feature.txt` to train GATNE-T model or GATNE-I model, respectively. 

You can use the following commands to train GATNE-T on Twitter and YouTube datasets: `python src/main.py --input data/twitter --eval-type 1` or `python src/main.py --input data/youtube`. We only evaluate the edges of the first edge type on Twitter dataset as the number of edges of other edge types is too small.

As Twitter and YouTube datasets do not have node attributes, you can generate heuristic features for them, such as DeepWalk embeddings. Then you can train GATNE-I model on these two datasets by adding the `--features` argument.

#### Training on your own datasets

If you want to train GATNE-T/I on your own dataset, you should prepare the following three(or four) files:
- train.txt: Each line represents an edge, which contains three tokens `<edge_type> <node1> <node2>` where each token can be either a number or a string.
- valid.txt: Each line represents an edge or a non-edge, which contains four tokens `<edge_type> <node1> <node2> <label>`, where `<label>` is either 1 or 0 denoting an edge or a non-edge
- test.txt: the same format with valid.txt
- feature.txt (optional): First line contains two number `<num> <dim>` representing the number of nodes and the feature dimension size. From the second line, each line describes the features of a node, i.e., `<node> <f_1> <f_2> ... <f_dim>`.

If your dataset contains several node types and you want to use meta-path based random walk, you should also provide an additional file as follows:
- node_type.txt: Each line contains two tokens `<node> <node_type>`, where `<node_type>` should be consistent with the meta-path schema in the training command, i.e., `--schema node_type_1-node_type_2-...-node_type_k-node_type_1`. (Note that the first node type in the schema should equals to the last node type.)


If you have ANY difficulties to get things working in the above steps, feel free to open an issue. You can expect a reply within 24 hours.

## Cite

Please cite our paper if you find this code useful for your research:

```
@inproceedings{cen2019representation,
  title = {Representation Learning for Attributed Multiplex Heterogeneous Network},
  author = {Cen, Yukuo and Zou, Xu and Zhang, Jianwei and Yang, Hongxia and Zhou, Jingren and Tang, Jie},
  booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year = {2019},
  pages = {1358--1368},
  publisher = {ACM},
}
```