
import tensorflow as tf
import scipy.sparse as sp
import numpy as np
import GCN
import utils

class AFGSM:
    def __init__(self, A, X, Z, num_vicious_nodes, num_vicious_edges, dmin=1):
        """
        :param A:  sparse matrix, the adjacency matrix ,[n X n]
        :param X:  sparse or dense  matrix, the feature matrix ,[n x d], d is the dimension of features
        :param Z:  sparse matrix, the labels, [n x c], c is the dimension of one-hot label
        :param num_vicious_nodes:  int, the number of vicious nodes
        :param num_vicious_edges:  int, the number of vicous edges
        :param dmin: int, min degree assigned for vicious nodes
        """
        self.A = A.tocsr()
        self.A_orig = self.A.copy()
        self.A.setdiag(0)
        self.X = X
        self.Z = Z
        self.labels = list(np.squeeze(np.argmax(self.Z, axis=1)))

        self.num_vicious_nodes = num_vicious_nodes

        self.num_vicious_edges = num_vicious_edges

        self.An = utils.preprocess_graph(self.A)
        self.degree = np.squeeze(self.A.sum(axis=1).getA()) + 1
        self.old_degree = np.squeeze(self.A.sum(axis=1).getA()) + 1
        if sp.issparse(self.X):
            self.cooc_X = sp.csr_matrix(self.X.T.dot(self.X))
            self.cooc_X[self.cooc_X.nonzero()] = 1
            self.X_d = int(np.sum(X) / self.X.shape[0])
        else:
            self.cooc_X = None
            self.X_d = None
        self.D_inv = sp.diags(1 / self.degree)
        self.D_inv_sqrt = sp.diags(1 / np.sqrt(self.degree))
        self.dv = self.get_random_degrees(dmin)

    def get_random_degrees(self, dmin=1):
        """
        assign degrees of vicious nodes randomly
        :param dmin: min degree assigned for vicious nodes
        :return: a numpy array contains the degrees of vicious nodes
        """
        dv = np.zeros(self.num_vicious_nodes,dtype=int) + dmin
        for _ in range(int(self.num_vicious_edges-dmin*self.num_vicious_nodes)):
            dv[np.random.randint(0, self.num_vicious_nodes)] += 1
        return dv

    def  cal_loss(self, logits_attacked, idx):
        best_wrong_label = np.argmax(logits_attacked[idx, :] - 1000000 * self.Z[idx, :])
        true_label = np.argmax(self.Z[idx, :])
        loss_ = logits_attacked[idx, true_label] - logits_attacked[idx,  best_wrong_label]
        return loss_

    def classification_margins(self, probs, idx):
        true_label = np.argmax(self.Z[idx, :])
        best_wrong_label = np.argmax(probs[0, :] - 1000 * self.Z[idx, :])
        return -probs[0, best_wrong_label] + probs[0, true_label]

    def update_M(self, M1, M2, w, idx,i, direct_attack):
        """
        update the  intermediate quantity for calculation of gradients
        """
        nodes_changed = self.A[-1, 0:-1].nonzero()[1]
        if direct_attack:
            M1 = M1 * np.sqrt(self.dv[i-1] / self.dv[i])
            M1[nodes_changed] = M1[nodes_changed] / np.sqrt(self.degree[nodes_changed]) * np.sqrt(self.old_degree[nodes_changed])
            M1_1 = (1 / np.sqrt(self.dv[i]*self.degree[-1]) * self.X[-1, :]).dot(w)
            M1_ = np.zeros(M1.shape[0] + 1)
            M1_[0:-1] = M1
            M1_[-1] = M1_1
        else:
            M1_ = None

        M2[nodes_changed] = M2[nodes_changed] * self.degree[nodes_changed] / self.old_degree[nodes_changed]
        M2_1 = (self.A[-1, idx])/self.degree[-1]
        M2_ = np.zeros(M2.shape[0]+1)
        M2_[0:-1] = M2
        M2_[-1] = M2_1
        return M1_, M2_

    def get_sampled_features(self, w, w_idx,random_choose =True):
        """
        sample features under the constraints
        """
        x = sp.csr_matrix(np.zeros((1, self.X.shape[1])))
        nnz = []
        length = 0
        if random_choose:
            random_idx = np.random.choice(w_idx, w_idx.shape[0])
        else:
            random_idx = w_idx
        for i in list(random_idx):
            flag = 1
            if length == self.X_d:
                break
            for j in nnz:
                flag *= self.cooc_X[i,j]
                if flag==0:
                    break
            if flag==1:
                x[0, i] = 1
                nnz.append(i)
                length += 1
        return x


    def adaptive_train(self, sizes, idx, split_train, split_val,
              perturb_features = True, direct_attack=True, verbose=True,):
        """
        adaptive attack, AFGSM-ada
        :param sizes: list, the hidden size of GCN
        :param idx: int, the target node ID
        :param split_train:  list, train set for GCN
        :param split_val:  list, valuation set for GCN
        :param perturb_features: bool, if True, perturb features
        :param direct_attack:  bool, if True, direct attack
        :param verbose: bool, whether to show losses
        """

        true_label = np.argmax(self.Z[idx, :])

        for i in range(self.num_vicious_nodes):
            with tf.Graph().as_default():
                _An = utils.preprocess_graph(self.A)
                surrogate_model = GCN.GCN(sizes, _An, self.X, with_relu=False, name="surrogate",gpu_id=0)
                surrogate_model.train(split_train, split_val, self.Z)
                W1 = surrogate_model.W1.eval(session=surrogate_model.session)
                W2 = surrogate_model.W2.eval(session=surrogate_model.session)
                logits = surrogate_model.logits.eval(session=surrogate_model.session)
                surrogate_model.session.close()
                W = np.dot(W1, W2)
                best_wrong_label = np.argmax(logits[idx, :] - 1000 * self.Z[idx, :])

            w = W[:, best_wrong_label] - W[:, true_label]
            w_idx = np.argsort(w)[::-1]

            for j in range(w_idx.shape[0]):
                if w[w_idx[j]] < 0:
                    w_idx = w_idx[0:j]
                    break

            self.D_inv = sp.diags(1 / self.degree)
            self.D_inv_sqrt = sp.diags(1 / np.sqrt(self.degree))
            self.d_inv_sqrt = 1 / np.sqrt(self.degree)
            self.d_inv_sqrt = np.squeeze(self.d_inv_sqrt)


            if direct_attack:
                M1 = np.squeeze(1 / np.sqrt(self.dv[0]) * self.d_inv_sqrt*(self.X.dot(w)))
            else:
                M1 = None
            A_idx = self.A[:, idx]
            A_idx[idx] = 1
            M2 = np.squeeze(self.d_inv_sqrt * A_idx)
            e = sp.csr_matrix(np.zeros((self.A.shape[0], 1)))
            x = self.X[np.random.randint(0, self.X.shape[0]), :]
            x_nnz = np.array(x.nonzero())
            if x_nnz.shape[1] > self.X_d:
                sample_idx = np.random.randint(0, x_nnz.shape[1], size=[x_nnz.shape[1] - self.X_d])
                x[:, x_nnz[:, sample_idx][1, :]] = 0
            X_mod = sp.vstack((self.X, x))

            z_vi = np.random.randint(0, self.Z.shape[1])
            Z_vi = np.zeros((1, self.Z.shape[1]), dtype=np.int32)
            Z_vi[0, z_vi] = 1
            Z = np.vstack((self.Z, Z_vi))
            self.labels.append(np.random.randint(0, self.Z.shape[1]))
            if perturb_features:
                x = self.get_sampled_features(w, w_idx, False)
                X_mod = sp.vstack((self.X, x))

            if direct_attack:
                grad_e = np.squeeze(M1 + M2 * (x.dot(w)))
                grad_e[idx] = 999999
            else:
                grad_e = np.squeeze(M2 * (x.dot(w)))
                grad_e[idx] = -999999

            gradients_idx = np.argsort(grad_e)[::-1][0:self.dv[i]]

            if np.sum(grad_e > 0) < self.dv[i]:
                e[grad_e > 0, 0] = 1
            else:
                e[gradients_idx, 0] = 1

            A_mod = sp.hstack((sp.vstack((self.A, e.T)), sp.vstack((e, 0))))

            if verbose:
                with tf.Graph().as_default():
                    _An_mod = utils.preprocess_graph(A_mod)
                    logits_attacked = _An_mod.dot(_An_mod).dot(X_mod).dot(W)
                    loss_ = self.cal_loss(logits_attacked, idx)
                    print("losses:", loss_)


            self.A = A_mod.tocsr()
            self.X = X_mod.tocsr()
            self.Z = Z
            self.old_degree = self.degree
            self.degree = list(self.degree)
            self.degree.append(np.sum(e) + 1)
            self.degree = np.array(self.degree)
            self.degree[e.nonzero()[0]] += 1


    def train(self, W, logits, idx,
              perturb_features = True, direct_attack=True, verbose=False):
        """
        AFGSM attack
        :param W: the weights of GCN
        :param logits: the logits of GCN
        :param idx:  the target node
        :param perturb_features: bool, if True, perturb features
        :param direct_attack:  bool, if True, direct attack
        :param verbose: bool, whether to show losses
        """

        sur_margins = []
        true_label = np.argmax(self.Z[idx, :])
        best_wrong_label = np.argmax(logits[idx, :] - 1000 * self.Z[idx, :])
        w = W[:, best_wrong_label] - W[:, true_label]
        w_idx = np.argsort(w)[::-1]

        for i in range(w_idx.shape[0]): #values too small are ignored
            if w[w_idx[i]]<0.001:
                w_idx = w_idx[0:i]
                break

        self.d_inv_sqrt = 1 / np.sqrt(self.degree)
        self.d_inv_sqrt = np.squeeze(self.d_inv_sqrt)

        if direct_attack:
            M1 = np.squeeze(1 / np.sqrt(self.dv[0]) * self.d_inv_sqrt * np.squeeze(self.X.dot(w)))
        else:
            M1 = None
        A_idx = self.A[:, idx]
        A_idx[idx] = 1
        A_idx = np.squeeze(A_idx.toarray())
        M2 = np.squeeze(self.d_inv_sqrt * A_idx)

        for i in range(self.num_vicious_nodes):
            if i>=1:
                M1, M2 = self.update_M(M1, M2, w, idx, i, direct_attack)
            e = sp.csr_matrix(np.zeros((self.A.shape[0], 1)))
            x = self.X[np.random.randint(0, self.X.shape[0]), :]
            if sp.issparse(self.X):
                x_nnz = np.array(x.nonzero())
                if x_nnz.shape[1] > self.X_d:
                    sample_idx = np.random.randint(0, x_nnz.shape[1], size=[x_nnz.shape[1] - self.X_d])
                    x[:, x_nnz[:, sample_idx][1, :]] = 0
                X_mod = sp.vstack((self.X, x))
            else:
                X_mod = np.vstack((self.X, x))

            z_vi = np.random.randint(0, self.Z.shape[1])
            Z_vi = np.zeros((1, self.Z.shape[1]),dtype=np.int32)
            Z_vi[0, z_vi] = 1
            Z = np.vstack((self.Z, Z_vi))
            self.labels.append(np.random.randint(0, self.Z.shape[1]))
            if perturb_features:
                x = self.get_sampled_features(w, w_idx, False)
                X_mod = sp.vstack((self.X, x))

            if direct_attack:
                grad_e = np.squeeze(M1 + M2 * (x.dot(w)))
                grad_e[idx] = 999999
            else:
                grad_e = np.squeeze(M2 * (x.dot(w)))
                grad_e[idx] = -999999

            gradients_idx = np.argsort(grad_e)[::-1][0:self.dv[i]]
            if np.sum(grad_e > 0) < self.dv[i]:
                e[grad_e > 0, 0] = 1
            else:
                e[gradients_idx, 0] = 1

            A_mod = sp.hstack((sp.vstack((self.A, e.T)), sp.vstack((e, 0))))

            if verbose:
                with tf.Graph().as_default():
                    _An_mod = utils.preprocess_graph(A_mod)
                    logits_attacked = _An_mod.dot(_An_mod).dot(X_mod).dot(W)
                    loss_ = self.cal_loss(logits_attacked, idx)
                    print("losses:", loss_)
                    sur_margins.append(loss_)

            self.A = A_mod.tocsr()
            if sp.issparse(self.X):
                self.X =X_mod.tocsr()
            else:
                self.X = X_mod
            self.Z = Z
            self.old_degree = self.degree
            self.degree = list(self.degree)
            self.degree.append(np.sum(e)+1)
            self.degree = np.array(self.degree)
            self.degree[e.nonzero()[0]] += 1