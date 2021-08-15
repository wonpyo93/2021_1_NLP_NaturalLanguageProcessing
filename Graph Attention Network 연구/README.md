# GAT Structures

** PREREQUISITE 으로 읽어야 할 논문들 **

GCN

[http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf](http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf)

GAT

[https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

HAN

[https://arxiv.org/pdf/1903.07293.pdf](https://arxiv.org/pdf/1903.07293.pdf)

GTN

[https://papers.nips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf](https://papers.nips.cc/paper/2019/file/9d63484abb477c97640154d40595a3bb-Paper.pdf)

AdaEdge

[https://arxiv.org/pdf/1909.03211.pdf](https://arxiv.org/pdf/1909.03211.pdf)

Regularized GAT

[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054363](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9054363)

논문 Short Reviews

ICLR 2020 - ADSF

[https://openreview.net/pdf/d628a352c692f2480ce51c29e1b67b3609a3ffe6.pdf](https://openreview.net/pdf/d628a352c692f2480ce51c29e1b67b3609a3ffe6.pdf)

"structure attention을 반영
기존 attention은 1-hop만 봐서 structure 정보를 온전히 담을 수 없다는 limit이 존재
adaptive structural fingerprints
node 한개를 가지고 k-hop 까지의 모든 주변 노드를 합쳐서 한개의 subgraph로 둘 때.. 결국 k=1보다 k=2인 ADSF 스터럭쳐 정보를 심어서 좋은 점수를 얻을 수 있었다.. FingerPrint. structural attention
wp: ADSF는 두 가지의 attention scores를 사용한다. content-based 인 e와 structure-based인 s.
그런데 여기서 'fingerprints'라고 부르는 local receptive field는 k-hop neighborhood로 제한한다고 하는데, 이 논문에서 이 k를 2로 둔다.
다시 말하면 Figure 3에서처럼 보여진 저런 그림들은 사실상 무의미하고 Figure 4가 좀더 정확한건데,
사실 2hop만 고려하는게 큰 의미가 있나 싶다.
내가 저번주에 아무리 식을 끼워넣어도 structure한 value들이 너무 influence가 작아서 문제가 있었는데,
이 논문은 그냥 두 개를 더해서 평균내는 식으로 최종 attention을 구하는 데에 있어서 영향을 크게 준 것 같다.
(물론 이건 아직 코드를 제대로 분석하지 않은 상태에서의 추론일 뿐이다)"

GRAPH-BERT: Only Attention is Needed for Learning Graph Representations

[https://arxiv.org/pdf/2001.05140.pdf](https://arxiv.org/pdf/2001.05140.pdf)

Graph 와 Bert를 결합.
Linkless Subgraph Batching
Node Input Embedding
Graph-Transformer Based Encoder
Node Classification OR Graph Clustering"

GNN Exponentially Lose Expressive Power for Node Classification

이 논문은 그냥 레이어를 쌓을수록 안좋다는 내용, 그리고 GCN limited 내용이라 읽다 그만둠

NGAT4Rec: Neighbor-Aware Graph Attention Network For Recommendation

[https://arxiv.org/pdf/2010.12256.pdf](https://arxiv.org/pdf/2010.12256.pdf)

핵심 내용: 기존의 GAT는 중심 노드와 그 주변 노드들과의 관계만 생각을 했었다, 하지만 이젠 주변 노드들끼리의 관계도 생각을 해줘야 한다. attention coefficient e1과 e2를 봐도 만드는 과정이 중심노드와 1번 주변노드의 관련도를 e1라 보고 e2도 마찬가지인데 이 값을 구할 때 그 주변노드끼리도 봐줘야한다는 말

Graph Neural Network for Tag Ranking in Tag-enhanced Video Recommendation

[https://dl.acm.org/doi/pdf/10.1145/3340531.3416021](https://dl.acm.org/doi/pdf/10.1145/3340531.3416021)

Video, Tag, Media, User라는 노드를 정의를 한 후 neighbor-similarity도 구하고 graphSAGE 를 인용해서 HFIN (Heterogeneous Field Interaction Network)을 구한다는데 너무 데이터셋을 이리저리 손을 많이 봐서 포기

Bilinear Graph Neural Network with Neighbor Interactions

[https://pdfs.semanticscholar.org/4d39/df7d5e2cc81318eda44caec51eb2558e98cc.pdf](https://pdfs.semanticscholar.org/4d39/df7d5e2cc81318eda44caec51eb2558e98cc.pdf)

"BGNN = AGG + BA
AGG = Neighbor feature를 recursive하게 aggregate
BA = Bilinear Aggregator = Neighbor의 Interaction을 나타내기 위함 = 모든 Neighbor의 hW를 쌍으로 element-wise product를 낸 값
BGNN = 저걸 Multi-Layer로 여러번 하는데 대신 최대한 recursive 안하게 보이기 위해 무언가를 하는데, 여기서 베타는 1hop이냐 2hop이냐의 strength 조절"

Supergat

[https://openreview.net/pdf?id=Wi5KUNlqWty](https://openreview.net/pdf?id=Wi5KUNlqWty)

"4 가지의 방법을 비교: GO, DP, SD, MX
GO: GAT Original Single Layer Neural Network
(concat 후 a라는 vector 써서 곱하는 방식)
DP: Dot-Product
(그냥 feature a, b를 dot product로 dot(a, b.T) 하는 방식)
SD: Scaled Dot-Product
(DP와 같지만 scale 해서 sqrt(F)를 나누는 방식)
MX: GO와 DP를 mix 하는 방식
비교 결과, Node Classification의 경우 큰 차이는 없지만 DP가 GO보다 대부분 더 좋았음.
MX가 SD보다 Cora, CiteSeer, PubMed 에서 모두 더 좋은 성능을 보임
질문: 원래 GAT에서 Label을 고려했었는가? "

ATPGNN

[https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9319003](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9319003)

"우리가 원하는 핵심 내용을 놀라울 정도로 잘 짚고 넘어가주는 논문
semantic attention 'e'
remote similar structure node attention 'f'
node embedding attention 'g'
이렇게 3개를 구해서 3개를 다 짬뽕하는 논문.
이 논문의 좋은 점은 GAT, GCN, ADSF, Deepwalk 등 우리가 생각했던 모든 method들이 다 등장하고 무엇이 어떤거에 더 좋은지 등을 상세히 설명하고 넘어감
결과는 e를 GAT로, f를 GraphWave로, g를 GraphWave로 계산한 방식이 node classification이 Cora, Citeseer, Pubmed에서 가장 좋았음"

RoGAT

[https://arxiv.org/pdf/2009.13038.pdf](https://arxiv.org/pdf/2009.13038.pdf) 

ROBUST GAT --> 말그대로 튼튼한 GAT. 튼튼하다는 말은 버그, 즉 못된 edge에 견고한 GAT를 만드는 것.

"Modeling Graph Structure via Relative Position for Text Generation from
Knowledge Graphs"

[https://arxiv.org/pdf/2006.09242.pdf](https://arxiv.org/pdf/2006.09242.pdf)

 graph attention based Knowledge Graph

"Unifying the Global and Local Approaches:
An Efficient Power Iteration with Forward Push"

[https://arxiv.org/pdf/2101.03652.pdf](https://arxiv.org/pdf/2101.03652.pdf)

personalized PageRank

Watch Your Step:Learning Node Embeddings via Graph Attention

[https://arxiv.org/pdf/1710.09599.pdf](https://arxiv.org/pdf/1710.09599.pdf)

E[D]를 구해보자.
D = co-occurance matrix from random walks.
E[D] = Expectation on the co-occurance matrix
—> Rather than obtaining D by simulation of random walks and sampling co-occurances, we formulate an 'expectation' of this sampling.
T = A normalized. = transition matrix
initial probability distribution —> p_0
p_1 = p_0_transpose * T
p_k = p_0_transpose * T^k
C —> walk length
k —> steps
Q —> context distribution
q —> variable trained via backpropagation, jointly while learning node embeddings
C를 매번 따로 구하지 않고싶다
C —> training window length
보통 C와 Q로 node neighborhood의 weight이 정해지는데, 이 C라는 hyperparameter를 trainable parameters로 바꾸고싶다
그러면 이 attention parameters가 random walk을 'guide' 해줄 수 있다

Graph Representation Learning via Hard and Channel-Wise Attention Networks

[https://dl.acm.org/doi/pdf/10.1145/3292500.3330897](https://dl.acm.org/doi/pdf/10.1145/3292500.3330897)

"저자 성이 Gao라서 그런지 Graph Attention OPERATOR라고 부르면서 이걸 GAO 라고 부른다...
hGAO, cGAO를 소개하는데, 둘 다 computational cost를 줄이는데 집중한다. (Cora, Citeseer, Pubmed 등의 성능이 0.2, 0.5 정도 오른걸로 보아 성능 면으론 큰 차이가 없는 듯...)
hGAO는 hard GAO로, 그냥 GAO를 soft GAO로 칭한다.
soft GAO는 center node가 자신의 '모든' 이웃 노드를 참고해야하는데, hGAO는 top K개만 참고한다. 참고의 기준은 trainable projection vector 'p'가 있을 때, feature vector X와 p를 project한 값을 가지고 가장 높은 top k개를 산출하는 방식.
cGAO는 Channel-Wise GAO로, node대node-feature vector로 attention을 뽑는 다른 GAO와는 달리, Channel을 이용해서 attention을 뽑는다.
근데 cGAO의 각 channel을 Xi라고 부르는데 (pg 744), 이게 각 채널의 feature matrix끼리 곱하려는건지..?"

"HighwayGraph: Modelling Long-distance Node Relations for
Improving General Graph Neural Networks"

[https://arxiv.org/pdf/1911.03904.pdf](https://arxiv.org/pdf/1911.03904.pdf)

우리 토픽에 매우 연관되어있을것같다는 대작스멜!! (다 읽고나니.. 맞나..? 하는 생각이...)
Implicit Modelling과 Explicit Modelling을 사용한다.
Implicit Modelling은 center node와 remote nodes와의 feature가 같은 카테고리인지 비교
remote nodes = unlabeled nodes that have a long topological distance
GNN 모델을 써서 나온 hidden rep 'y'를 구하고, 그 값에 softmax를 취한걸 l-hat이라 부른다. 이건 각 노드의 hidden rep.
'y' 와 'y-T'의 sigmoid, 즉 두 노드간의 hidden rep.
Explicit Modelling은 Implicit Modelling에서 나온 값의 confidence가 어떤 threshold보다 크면 그 노드 둘 사이에 edge를 만들어준다
threshold 는 여기서 0.9로 잡았다고 한다
(( 여기서 GAT Citeseer가 66.7이라 한다..? 그리고 자기 모델은 68.5로 올렸다고 한다 ))

"Topology and Content Co-Alignment Graph
Convolutional Learning"

[https://arxiv.org/pdf/2003.12806.pdf](https://arxiv.org/pdf/2003.12806.pdf)

또다른 'robust' GNN인 CoGL (견고한). Co-alignment이 핵심
real world 그래프는 사실 정확하지 않다. noisy message passing(unnecessary edge)이나 node relationship imparing(missing link) 등의 문제가 있다.
TO-GCN처럼 Topology 자체를 수정하는 방식이 있지만 overfitting의 위험이 있다.
그래서 network topology와 node content를 distinct yet highly correlated 한, 'co-alignment' learning principle를 제안한다.
이 논문이 하고자하는것: graph structure나 node content를 아예 fixed로 보거나 아예 손을 대는 것이 아닌, 'co-alignment' 한다! (사실 별거 없다)
3가지로 분류:

- Content-aligned Graph Topology Learning
A-bar를 사용하는데 A-bar = Feature Difference
(말그대로 node i의 feature 에서 node j의 feature를 빼버린다)
요런 애들 A로 두고 그냥 GCN처럼 구해서 나오는 Loss를 L-cont라고 부른다.
- Semi-supervised Graph Embedding
A-물결을 사용하는데 이건 우리가 아는 GCN과 동일하다.
여기서 나오는 Loss를 L-GCN이라 부른다.
- Adversarial Graph Embedding
A-bar를 통해 나온 representation을 class 1로, A-물결을 통해 나온 representation을 class 0으로 두고 classify를 진행한다. 여기서 나오는 Loss를 L-gan이라고 부른다.
나왔던 Loss들을 다 더해주면 최종 Loss!
(여기서 GAT Citeseer가 71.0이라고 말하고 자기들은 72.4가 나왔다고 함)

Topology Optimization based Graph Convolutional Network

[https://www.ijcai.org/Proceedings/2019/0563.pdf](https://www.ijcai.org/Proceedings/2019/0563.pdf)

GCN에서 Topology가 바뀌지 않는다는 단점을 보완하기 위한 TO-GCN
'가까운 노드면 같은 레이블일 가능성이 높다'는 assumption을 전제로 깔고 시작.
GCN의 Topology는 그냥 A로 끝나는데, 이 논문에선 A를 'refined network topology' O로 바꾼다.
원래 GCN의 Loss(classify)에다가 Loss(refine)을 결합한 Loss Function을 쓴다.
쉽게 설명하자면, 이미 레이블된 노드의 ground truth 값을 Loss에 반영하여,
맞은 labeled 노드들은 그대로 킵하고 틀린 labeled 노드들은 Loss를 이용하여 O를 바꿔서 맞출 가능성을 높게 O를 변경해주는 방식.
(predicted - groundTruth)
( 1 - 1 ) 이나 ( 0 - 0 ) 처럼 서로 같은 label이면 loss가 없고
( 1 - 0 ) 이나 ( 0 - 1 ) 처럼 서로 다른 label이면 loss가 생기는 방식
우리가 배울 점:
GAT가 아니다. (성능표에서도 GAT가 언급이 안될 정도)
애초에 성능이 뛰어나게 좋아진 것도 아니다 (GCN에 비교해서 2.4%가 올랐지만 그래봤자 72.7)
이미 레이블 된 노드를 이용하는, Loss 를 이용해 A값을 update해준다는 것은 참신함.

Overcoming Catastrophic Forgetting in Graph Neural Networks

[https://arxiv.org/pdf/2012.06002.pdf](https://arxiv.org/pdf/2012.06002.pdf)

'Catastrophic Forgetting'을 방지하는 Topology-Aware Weight Preserving (TWP)를 쓰자!
즉, node-classification task를 여러개 sequence적으로 하다 보면, 새로운 task를 배울 때마다 예전 것을 forgetting하게 된다.
Topology-aware Weight Preserving
Minimized Loss Preserving
각 task 마다 생성되는 Loss의 gradient
Topological Structure Preserving
attention의 task마다 달라지는 gradient
이 둘의 'importance'를 구해서 최종 loss나 H'을 만들때 적용하는 방법

Robust Hierarchical Graph Classification with
Subgraph Attention

[https://arxiv.org/pdf/2007.10908.pdf](https://arxiv.org/pdf/2007.10908.pdf)

이 논문의 Task = to predict the label of a graph (node의 label을 구하는게 아님)
'GNN은 노드끼리 비교를 하지만, real world에선 sub-structure에 따라 label이 사실상 결정된다'
한 노드마다 subgraph들을 만들어내는데, 그 subgraph 안에 들어가는 총 노드 수는 T로 정한다. 이 때, subgraph의 수가 엄청 많아질 수 있는데, 그래서 L이라는 숫자를 만들고 subgraph 수가 L보다 많아지면 랜덤샘플링으로 subgraph를 선택한다. 만약 L이 더 크면, round robin sampling으로 subgraph 수를 뻥튀기한다.
3.1.2 (여러가지 방법을 했는데 concatenation of features가 제일 좋다 '카더라')
여기서 concate 할려면 각 subgraph의 feature#가 같아야해서 작은 애들한테 000000을 붙여준다
T=4인데 3개짜리 subgraph면 (xi||xj||xk||0) 이렇게. 참고로 여기서 feature dimension = D (F 아님)
그다음 attention coefficient 알파il을 구함. i는 노드, l은 L 중 한 숫자. 즉, center node - sub graph 의 attention coefficient
이게 한 개의 subgraph attention layer에 대한 과정.
이 논문이 소개하는 과정은 사실 R 개의 level을 가진 architecture. 처음부터 시작해서 그래프 크기를 subgraph로 묶으면서 점점 더 줄이는 과정. 처음에는 SubGatt 임베딩 layer를 통하고 나머지는 다 GIN 임베딩 layer로 줄여나감.
이렇게 줄여나가는데 있어서 noisy structure때문에 information loss가 일어날 수 있기 때문에 여기서 또 attention을 쓴다. 여기서 첫 layer는 빼고 두번째부터 r번째까지 level만 이용한다. 거기서 나오는 attention을 intra-level attention이라 하고, 각 level마다 나온걸 하나의 Xinter, (R-1 * K)로 만든 다음 마지막 attention 구함.

DropEdge

[https://openreview.net/pdf?id=Hkx1qkrKPr](https://openreview.net/pdf?id=Hkx1qkrKPr)

A를 p의 probability로 edge들을 random하게 드랍해서 새로 만든 A'를 가지고 GNN 모델에 적용하는 방법. 각 Layer마다 계속 다른 A들이 들어가게 된다.
Overfitting, Over-smoothing을 방지할 수 있다고 주장.

Graph Attention Auto-Encoders

[https://arxiv.org/abs/1905.10715](https://arxiv.org/abs/1905.10715)

GCN이나 GAT는 label information 기반인데 사실 real-world에선 label이 많이 없다.
그리고 GNN 모델들은 사실 semi-supervised나 supervised한 모델들이 많은데 unsupervised 한 모델은 많이 없다.
그래서 encoder / decoder를 활용하여 unsupervised model을 만듦.
encoder에서 k번의 layer를 거쳐서 H'가 만들어졌다면, 그걸 다시 백트레이싱하는 decoder를 이용해서,
결국 원래 x가 있고 그게 encoding 되고 다시 decoding 되서 predicted 한 x-hat의 차이를 보는 모델

Hierarchical Graph Convolutional Networks for Semi-supervised Node Classification

[https://arxiv.org/pdf/1902.06667.pdf](https://arxiv.org/pdf/1902.06667.pdf)

Hierarchical Graph Convolution Network (H-GCN)
(성능표에서 GAT가 72.5, GCN이 70.3인데 이건 72.8 라서 대충 읽었음)
First work to design a deep hierarchical model for semi-supervised node classification task

Simple and Deep Graph Convolutional Networks

[http://proceedings.mlr.press/v119/chen20v/chen20v.pdf](http://proceedings.mlr.press/v119/chen20v/chen20v.pdf)

GCN2 —> 사실 GCN II —> GCN + Initial residual + Identity mapping
여기서 사실 GCN + Initial residual = APPNP이라서 APPNP + Identity mapping임
그럼 Identity Mapping이 무엇이냐 (derived from ResNet)
at the l-th layer, add identity matrix In to the weight matrix W(l)
—> ensures that a 'deep' model achieves at least the same performance as its shallow version does
—> by setting 베타 sufficiently small, deep GCNII ignores the weight matrix W(l) and essentially simulates APPNP

Sparse GAT

[https://arxiv.org/pdf/1912.00552.pdf](https://arxiv.org/pdf/1912.00552.pdf)

열심히 읽다가 성능표보고 갑자기 벙쪄서 더이상 못읽겠음.. 메모리 efficient한건 알겠는데 original gat보다 낮으면 어캄 ㅜㅜ

GraphSAGE

[https://arxiv.org/pdf/1706.02216.pdf](https://arxiv.org/pdf/1706.02216.pdf)

Unseen nodes, Newly added nodes에 너무나 취약한 GNN의 현주소... 더 좋은 방법이 없을까?
(transductive —> inductive)
center node에서 random sampling 과 random walk를 통한 이웃 노드 반영.
Appendex, Algorithm 2에서 minibatch가 만들어지는 과정을 봤을 때,
k 가 layer인데 이게 우리가 생각하는 1hop 원이 아님을 유의
k = K ... 1임을 유의
Bk 는 중심노드. 중심노드의 이웃을 샘플링한 것을 Bk-1에 넣어줌
(GCN에서는 1hop에 있는 애들 싹다 긁어모으는 방식)
이웃 노드를 반영할때 mean / max / sum / weight 등의 'degree 정보, 즉 unsupervised 방식을 채택!'
(GCN에서는 degree만큼 가중치 내려주는 'aggregate' 방식)
learnable parameter를 통해서 저 mean/max/sum/weight 등의 정보가 inductive하게 바뀌는 방식
(GCN에서는 그냥 DAD를 쓰는 '자신포함학습'이 끝)
Loss는 이웃을 더 좋게, negative는 더 멀게 학습

GNNExplainer

[https://arxiv.org/pdf/1903.03894.pdf](https://arxiv.org/pdf/1903.03894.pdf)

1. 해당 GNN 모델의 신뢰도 향상
2. model의 transparency(투명성), fairness(공평성), privacy(보안성) 확보
3. 사용자의 이해도 향상, 취약점 확인에 도움
등의 이유로 explaining은 필요합니다!

어떤 노드가 GNN모델의 학습을 통해서 어떠한 레이블로 분류가 되었을때, 어떤 노드들에게 중요하게 영향을 받게 되어서 이렇게 되었는가?
subgraph로 나눠서 생각해보자. 원래 A에서 pertrub를 시켜서 subgraph를 만든다 (몇개의 edge가 사라지겠죠?) 그러면 거기서 원래 값이랑 비교해서 무엇이 중요하게 작용했는지 확인할 수 있다.
subgraph는 엄청 많을텐데 그럼 어떻게 만드나요? random하게 뽑습니다. 그러면 기대값이 나오게 됩니다.
그리고 거기서 나온 0과 1 값들을 softmax를 취해서 여러개의 softmax개의 값을 meanfield approximation을 통해서 분포가 나오게 되는데, 사실 그게 어떻게 보면 masking이랑 똑같다
그래서 '모델이 저 prediction을 한 이유가 뭘까'를 알 수가 있는데, 이걸 더 키우자면,
'모델이 저 label로 predict한 이유가 뭘까'도 crossentropy로 알 수 있다고 한다

DeepWalk

[https://arxiv.org/pdf/1403.6652.pdf](https://arxiv.org/pdf/1403.6652.pdf)

Goal:
learn social representation of graph's node
== unsupervised methods to learn featuers that 'capture' the graph structure, independent of label's distribution
== learn X(E) ←- R |V| * d, where d = small number of latent dimension을 찾는것.
Section 4.1에서부터 차근차근 가다보면 Algorithm2, Skipgram에 대해 나오는데, 여기서 3번 라인에서 loss를 가장 낮게 하기 위해선 Pr 부분, 즉 probability를 maximize 하는게 중요하다. 이걸 하기 위해서 skipgram을 씀

WHAT GRAPH NEURAL NETWORKS CANNOT LEARN:
DEPTH VS WIDTH

[https://arxiv.org/pdf/1907.03199.pdf](https://arxiv.org/pdf/1907.03199.pdf)

GNN 모델은 dataset의 depth와 width에 따라 엄청나게 성능이 좌지우지된다.

Community aware random walk for network embedding

[https://reader.elsevier.com/reader/sd/pii/S0950705118300911?token=0DEAAF66C24D00C58CBBF0ED2F38AFD387D52AA38B6BB5450FE3E0D1A69BD31725EF160212F7CE3EE0037F8DFF88A7B6&originRegion=us-east-1&originCreation=20210412082217](https://reader.elsevier.com/reader/sd/pii/S0950705118300911?token=0DEAAF66C24D00C58CBBF0ED2F38AFD387D52AA38B6BB5450FE3E0D1A69BD31725EF160212F7CE3EE0037F8DFF88A7B6&originRegion=us-east-1&originCreation=20210412082217)

CARE (Community Aware Random walk for node Embedding)
Deepwalk, LINE, Node2Vec 등의 approach들은 represent nodes with some informative feature vectors이다
BUT extract only local structure from each node만 하는 단점이 있다.
Community! 같은 Community 안에 있는 노드들은 다른 Community 의 노드와 비교해서 더 similar 해야한다. 또한, local structure에서 weak 하다고 해도 community structure constraint으로 인해서 더 높이 줘야하는 경우가 있다. 그래서 community를 고려해 줘야한다.
CARE는 Louvain Method를 통해 Community 정보를 추출하고 그 추출한 정보를 가지고 Random Walk을 진행한다. 그 후, skipgram을 통해 representation vector of the node를 배운다.
(결국 deepwalk이랑 똑같은데 randomwalk에서 community 정보를 추가해주는 것 뿐!)
그러면 community정보는 뭐냐? Section 3.1, equation (1)
Algorithm 2를 보면, 해당 노드가 이웃이 있고!!! 알파 값에 따라 그냥 neighbor에 주거나 아니면 community값을 이용해서 community안에 random node로 간다. 만약 이웃이 없다면!!! 백트레이스로 노드들 돌아가면서 해당 노드가 이미 만들어진 path 외의 다른 이웃이 있는 노드일때까지 백트레이스 한다.

PREDICT THEN PROPAGATE: GRAPH NEURAL
NETWORKS MEET PERSONALIZED PAGERANK

[https://arxiv.org/pdf/1810.05997.pdf](https://arxiv.org/pdf/1810.05997.pdf)

GCNII 에서 나오는 내용.  APPNP임
node embedding methods use 'random walks' or 'matrix factorization'이라고 하는걸 봐서 저거 두개 말곤 아직까지 나온게 없는듯...

- inherent connection between the limit distribution and pagerank
- algorithm that utilizes a propagation scheme derived from personalized pagerank (adds a chance of teleporting back to the root node, which ensures that the PageRank score encodes the local neighborhood for every root node)
이 논문이 주장하는 바는, pagerank를 사용하면 INFINITELY MANY layer를 쌓아도 된다고 합니다

Diffusion Improves Graph Learning

[https://arxiv.org/pdf/1911.05485.pdf](https://arxiv.org/pdf/1911.05485.pdf)

Instead of aggregating information only from the first-hop neighbors, GDC aggregates information from a larger neighborhood. —> Neighborhood is constructed via a new graph generated by sparsifying a generalized form of a graph diffusion
Generalized Graph Diffusion (page 2 참고) —> S:
weighting coefficients * generalized transition matrix T.
T in an undirected graph include the random walk transition matrix T_rw = A * D^-1
T of symmetric transition matrx T_sym = D^-1/2 * A * D^-1/2

JK-Net

[https://arxiv.org/pdf/1806.03536.pdf](https://arxiv.org/pdf/1806.03536.pdf)

JK networks = Jumping Knowledge Networks (GCN, GAT, GraphSAGE 와 결합 가능)
Introduction에서 2 layer보다 더 많이 보는걸 residual로 가능하게 했지만 'citation networks'의 경우 그렇지 않다고 말했음...!!!!!!!!!!!!
we propose an architecture that, as opposed to existing models, enables adaptive, structure-aware representations. Such representations are particularly interesting for representation learning on large complex graphs with diverse subgraph structures.
dataset의 특성에 따라 어떻게 결합해야하는지가 다르다고 생각해서 그거에 초점을 맞췄음
3. Influence Distribution and Random Walks
measure the sensitivity of node x to node y, or the influence of y on x, by measuring how much a change in the input feature of y affects the representation of x in the last layer. For any node x, the influence distribution captures the relative influences of all other nodes.
Influence score I(x,y) —> Definition 3.1
Large radii may lead to too much averaging, while small radii may lead to insufficient information aggregation. Thus, we propose two simple yet powerful architectural changes —> jump connection and a subsequent selective but adaptive aggregation mechanism
Layer Aggregation!!!
근데 엄청난 사실. Table 2를 보면 GAT가 Citeseer에서 76.2가 나온다고함 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ