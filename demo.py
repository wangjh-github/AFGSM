import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import scipy.sparse as sp
import GCN
import utils
import AFGSM


direct_attack = True # dicrect attack or indirect attack
perturb_features = True # whether perturb features
dataset = 'cora' #dataset name
num_vicious_nodes = 10 # the number of vicious nodes
num_vicious_edges = 30 # the number of vicious edges
adaptive = False # adaptive attack or not
gpu_id = 0 # your GPU ID
seed = 1234 # the random seeds
dmin=1 #the min degree for vicious nodes
idx = 5 #the ID of target node
retrain_iters = 5

'''
read data
'''
_A_obs, _X_obs, _z_obs = utils.load_npz(
    './data/{}.npz'.format(dataset))

_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1

lcc = utils.largest_connected_components(_A_obs)

_A_obs = _A_obs[lcc][:, lcc]
_A_obs.setdiag(0)
_A_obs = _A_obs.astype("float32")
_A_obs.eliminate_zeros()
_X_obs = _X_obs.astype("float32")

assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"

_X_obs = _X_obs[lcc]
_z_obs = _z_obs[lcc]

_An = utils.preprocess_graph(_A_obs)

X_degree = np.squeeze(np.array(sp.spmatrix.sum(_X_obs, axis=1)))
mean_x = int(np.mean(X_degree))

_N = _A_obs.shape[0]
_K = _z_obs.max() + 1
_Z_obs = np.eye(_K)[_z_obs]

sizes = [16, _K]

unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share
np.random.seed(seed)
label_idx = _z_obs[idx]
split_train, split_val, split_test = utils.train_val_test_split_tabular(np.arange(_N), train_size=train_share,
                                                                         val_size=val_share,
                                                                         test_size=unlabeled_share,
                                                                         stratify=_z_obs,
                                                                         random_state=seed)



with tf.Graph().as_default():
    surrogate_model = GCN.GCN(sizes, _An, _X_obs, with_relu=False, name="surrogate", gpu_id=gpu_id)
    surrogate_model.train(split_train, split_val, _Z_obs)
    W1 = surrogate_model.W1.eval(session=surrogate_model.session)
    W2 = surrogate_model.W2.eval(session=surrogate_model.session)
    logits = surrogate_model.logits.eval(session = surrogate_model.session)
    surrogate_model.session.close()


'''
perform AFGSM 
'''
with tf.Graph().as_default():
    afgsm = AFGSM.AFGSM(_A_obs, _X_obs, _Z_obs, num_vicious_nodes, num_vicious_edges, dmin=dmin)
    if adaptive:
        afgsm.adaptive_train(
            sizes, idx, split_train, split_val,
            perturb_features=perturb_features, direct_attack=direct_attack,verbose=False)
    else:
        afgsm.train(np.dot(W1, W2), logits, idx, perturb_features=perturb_features, direct_attack=direct_attack)
'''
Evaluation
'''
gcn_probs_attacked = []
for _ in range(retrain_iters):
    with tf.Graph().as_default():
        _An_mod = utils.preprocess_graph(afgsm.A)
        gcn_attacked = GCN.GCN(sizes, _An_mod, afgsm.X, with_relu=True, name="surrogate",
                               seed=np.random.randint(0, 9999))
        gcn_attacked.train(split_train, split_val, afgsm.Z)
        probs = gcn_attacked.predictions.eval(session=gcn_attacked.session,
                                              feed_dict={gcn_attacked.node_ids: [idx]})
        gcn_probs_attacked.append(probs)
        gcn_attacked.session.close()
gcn_probs_attacked = np.array(gcn_probs_attacked)

gcn_probs_clean = []
for _ in range(retrain_iters):
    with tf.Graph().as_default():
        _An_mod = utils.preprocess_graph(_A_obs)
        gcn_clean = GCN.GCN(sizes, _An_mod, _X_obs, with_relu=True, name="surrogate",
                               seed=np.random.randint(0, 9999))
        gcn_clean.train(split_train, split_val, _Z_obs)
        probs = gcn_clean.predictions.eval(session=gcn_clean.session,
                                              feed_dict={gcn_clean.node_ids: [idx]})

        gcn_probs_clean.append(probs)
        gcn_clean.session.close()
gcn_probs_clean = np.array(gcn_probs_clean)

'''
visualization 
'''
def make_xlabel(ix, correct):
    if ix==correct:
        return "Class {}\n(correct)".format(ix)
    return "Class {}".format(ix)

figure = plt.figure(figsize=(12,4))
plt.subplot(1, 2, 1)
center_ixs_clean = []
for ix, block in enumerate(gcn_probs_clean.T):
    block = block[0]
    x_ixs= np.arange(len(block)) + ix*(len(block)+2)
    center_ixs_clean.append(np.mean(x_ixs))
    color = '#555555'
    if ix == label_idx:
        color = 'darkgreen'
    print(x_ixs, block)
    plt.bar(x_ixs, block, color=color)

ax=plt.gca()
plt.ylim((-.05, 1.05))
plt.ylabel("Predicted probability")
ax.set_xticks(center_ixs_clean)
ax.set_xticklabels([make_xlabel(k, label_idx) for k in range(_K)])
ax.set_title("Predicted class probabilities for node {} on clean data\n({} re-trainings)".format(idx, retrain_iters))

fig = plt.subplot(1, 2, 2)
center_ixs_retrain = []
for ix, block in enumerate(gcn_probs_attacked.T):
    block = block[0]
    x_ixs= np.arange(len(block)) + ix*(len(block)+2)
    center_ixs_retrain.append(np.mean(x_ixs))
    color = '#555555'
    if ix == label_idx:
        color = 'darkgreen'
    plt.bar(x_ixs, block, color=color)


ax=plt.gca()
plt.ylim((-.05, 1.05))
ax.set_xticks(center_ixs_retrain)
ax.set_xticklabels([make_xlabel(k, label_idx) for k in range(_K)])
ax.set_title("Predicted class probabilities for node {} after injecting {} nodes {} edges\n({} re-trainings)".format(idx, num_vicious_nodes, num_vicious_edges,  retrain_iters))
plt.tight_layout()
plt.savefig('demo.png', format='png',dpi=1000)
plt.show()
