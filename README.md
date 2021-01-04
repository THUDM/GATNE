# GATNE

### [Project](https://sites.google.com/view/gatne) | [Arxiv](https://arxiv.org/abs/1905.01669)

Representation Learning for Attributed Multiplex Heterogeneous Network.

[Yukuo Cen](https://sites.google.com/view/yukuocen), Xu Zou, Jianwei Zhang, [Hongxia Yang](https://sites.google.com/site/hystatistics/home), [Jingren Zhou](http://www.cs.columbia.edu/~jrzhou/), [Jie Tang](http://keg.cs.tsinghua.edu.cn/jietang/)

Accepted to KDD 2019 Research Track!

## ❗ News

Recent Updates (Nov. 2020):
- Use multiprocessing to speedup the random walk procedure (by `--num-workers`)
- Support saving/loading walk file (by `--walk-file`)
- The PyTorch version now supports node features (by `--features`)

Some Tips:
- The PyTorch version may not reproduce the results (especially on the Twitter dataset). Please use the original TensorFlow version (`src/main.py`) for reproducing the paper results.
- Running on large-scale datasets needs to set a larger value for `batch-size` to speedup training (e.g., several hundred or thousand).
- If **out of memory (OOM)** occurs, you may need to decrease the values of `dimensions` and `att-dim`.

Our GATNE models have been implemented by many popular graph toolkits:
- Deep Graph Library ([DGL](https://github.com/dmlc/dgl)): see https://github.com/dmlc/dgl/tree/master/examples/pytorch/GATNE-T 
- Paddle Graph Learning ([PGL](https://github.com/PaddlePaddle/PGL)): see https://github.com/PaddlePaddle/PGL/tree/main/examples/GATNE
- [CogDL](https://github.com/THUDM/cogdl): see https://github.com/THUDM/cogdl/blob/master/cogdl/models/emb/gatne.py

Some recent papers have listed GATNE models as a strong baseline:
- [Deep Adversarial Completion for Sparse Heterogeneous Information Network Embedding](https://dl.acm.org/doi/pdf/10.1145/3366423.3380134) (WWW'20)
- [Decoupled Graph Convolution Network for Inferring Substitutable and Complementary Items](https://dl.acm.org/doi/pdf/10.1145/3340531.3412695) (CIKM'20)
- [Graph Attention Networks over Edge Content-Based Channels](https://dl.acm.org/doi/pdf/10.1145/3394486.3403233) (KDD'20)
- [Temporal heterogeneous interaction graph embedding for next-item recommendation](http://shichuan.org/doc/84.pdf) (PKDD'20)
- [Link Inference via Heterogeneous Multi-view Graph Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-59410-7_48) (DASFAA 2020)
- [Multi-View Collaborative Network Embedding](https://arxiv.org/pdf/2005.08189.pdf) (Arxiv, May 2020)

Please let me know if your toolkit includes GATNE models or your paper uses GATNE models as baselines. 

## Prerequisites

- Python 3
- TensorFlow >= 1.8 or PyTorch

## Getting Started

### Installation

Clone this repo.

```bash
git clone https://github.com/THUDM/GATNE
cd GATNE
```

Please first install TensorFlow or PyTorch, and then install other dependencies by

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