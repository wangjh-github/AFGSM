Implementation of paper **Fast Adversarial Attack on Graph Data by Injecting Vicious Nodes**

by Jihong Wang, Minnan Luo, Fnu Suya, Jundong Li, Zijiang Yang, Qinghua Zheng

## Requirements
* `numpy`
* `scipy`
* `scikit-learn`
* `matplotlib`
* `tensorflow`

## Run the code
Here is a demo.py for you, just run `python demo.python`

## Example output
![AFGSM example](https://raw.githubusercontent.com/wangjh-github/AFGSM/master/demo.png)


## References
### Datasets
We provide two datasets in the `data `folder:  
#### Cora
McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie.  
*Automating the construction of internet portals with machine learning.*   
Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of   
attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

#### Citeseer
Sen, Prithviraj, Namata, Galileo, Bilgic, Mustafa, Getoor, Lise, Galligher, Brian, and Eliassi-Rad, Tina.   
*Collective classification in network data.*   
AI magazine, 29(3):93, 2008.

### Graph Convolutional Networks
Our implementation of the GCN algorithm is based on the authors' implementation,
available on GitHub [here](https://github.com/tkipf/gcn).

The paper was published as  

Thomas N Kipf and Max Welling. 2017.  
*Semi-supervised classification with graph
convolutional networks.* ICLR (2017).