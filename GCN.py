import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
from sklearn.metrics import f1_score
import logging
import scipy.sparse as sp

spdot = tf.sparse_tensor_dense_matmul
dot = tf.matmul


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, seed=1234)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    pre_out = tf.cast(pre_out, tf.float32)
    return pre_out * (1. / keep_prob)


def dense_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, seed=1234)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=x.dtype)
    pre_out = x * dropout_mask
    pre_out = tf.cast(pre_out, tf.float32)
    return pre_out * (1. / keep_prob)


class GCN:
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, params_dict={'dropout': 0.5}, gpu_id=0,
                 seed=123, logger=None):
        """
        Create a Graph Convolutional Network self in Tensorflow with one hidden layer.

        Parameters
        ----------
        sizes: list
            List containing the hidden and output sizes (i.e. number of classes). E.g. [16, 7]

        An: sp.sparse_matrix, shape [N,N]
            The input adjacency matrix preprocessed using the procedure described in the GCN paper.

        X_obs: sp.sparse_matrix, shape [N,D]
            The node features.

        name: string, default: ""
            Name of the network.

        with_relu: bool, default: True
            Whether there a nonlinear activation function (ReLU) is used. If False, there will also be
            no bias terms, no regularization and no dropout.

        params_dict: dict
            Dictionary containing other self parameters.

        gpu_id: int or None, default: 0
            The GPU ID to be used by Tensorflow. If None, CPU will be used

        seed: int, defualt: -1
            Random initialization for reproducibility. Will be ignored if it is -1.
        """

        self.graph = tf.Graph()
        if seed > -1:
            tf.set_random_seed(seed)
        else:
            tf.set_random_seed(np.random.randint(0, 99999))

        if An.format != "csr":
            An = An.tocsr()

        if logger is None:
            logging.basicConfig(level=logging.DEBUG)
            logger = logging.getLogger(__name__)

        self.logger = logger

        with self.graph.as_default():

            with tf.variable_scope(name) as scope:
                w_init = slim.xavier_initializer
                self.name = name
                self.n_classes = sizes[1]

                self.dropout = params_dict['dropout'] if 'dropout' in params_dict else 0.
                if not with_relu:
                    self.dropout = 0
                # self.dropout = 0

                self.learning_rate = params_dict['learning_rate'] if 'learning_rate' in params_dict else 0.01

                self.weight_decay = params_dict['weight_decay'] if 'weight_decay' in params_dict else 5e-4
                self.N, self.D = X_obs.shape

                self.node_ids = tf.placeholder(tf.int32, [None], 'node_ids')
                self.node_labels = tf.placeholder(tf.int32, [None, sizes[1]], 'node_labels')

                # bool placeholder to turn on dropout during training
                self.training = tf.placeholder_with_default(False, shape=())

                self.An = tf.SparseTensor(np.array(An.nonzero()).T, An[An.nonzero()].A1, An.shape)
                self.An = tf.cast(self.An, tf.float32)
                if not sp.issparse(X_obs):
                    self.X = tf.constant(X_obs)
                    self.X_dropout = dense_dropout(self.X, 1 - self.dropout, self.X.shape)
                    self.X_comp = tf.cond(self.training,
                                          lambda: self.X_dropout,
                                          lambda: self.X) if self.dropout > 0. else self.X
                else:
                    self.X_sparse = tf.SparseTensor(np.array(X_obs.nonzero()).T,
                                                    X_obs[X_obs.nonzero()].A1.astype(np.float32), X_obs.shape)
                    self.X_dropout = sparse_dropout(self.X_sparse, 1 - self.dropout,
                                                    (int(self.X_sparse.values.get_shape()[0]),))
                    # only use drop-out during training
                    self.X_comp = tf.cond(self.training,
                                          lambda: self.X_dropout,
                                          lambda: self.X_sparse) if self.dropout > 0. else self.X_sparse

                self.W1 = slim.variable('W1', [self.D, sizes[0]], tf.float32, initializer=w_init(seed=seed))
                self.b1 = slim.variable('b1', dtype=tf.float32, initializer=tf.zeros(sizes[0]))

                if not sp.issparse(X_obs):
                    self.h1 = spdot(self.An, tf.matmul(self.X_comp, self.W1))
                else:
                    self.h1 = spdot(self.An, spdot(self.X_comp, self.W1))

                if with_relu:
                    self.h1 = tf.nn.relu(self.h1 + self.b1)

                self.h1_dropout = tf.nn.dropout(self.h1, 1 - self.dropout, seed=seed)

                self.h1_comp = tf.cond(self.training,
                                       lambda: self.h1_dropout,
                                       lambda: self.h1) if self.dropout > 0. else self.h1

                self.W2 = slim.variable('W2', [sizes[0], sizes[1]], tf.float32, initializer=w_init(seed=seed))
                self.b2 = slim.variable('b2', dtype=tf.float32, initializer=tf.zeros(sizes[1]))

                self.logits = spdot(self.An, dot(self.h1_comp, self.W2))
                if with_relu:
                    self.logits += self.b2
                self.logits_gather = tf.gather(self.logits, self.node_ids)

                self.predictions = tf.nn.softmax(self.logits_gather)

                self.loss_per_node = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_gather,
                                                                             labels=self.node_labels)
                self.loss = tf.reduce_mean(self.loss_per_node)

                # weight decay only on the first layer, to match the original implementation
                if with_relu:
                    self.loss += self.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in [self.W1, self.b1]])

                var_l = [self.W1, self.W2]
                if with_relu:
                    var_l.extend([self.b1, self.b2])
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                  var_list=var_l)

                self.varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                self.local_init_op = tf.variables_initializer(self.varlist)


                gpu_options = tf.GPUOptions(allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

                self.session = tf.InteractiveSession(config=config)
                self.init_op = tf.global_variables_initializer()
                self.session.run(self.init_op)

    def convert_varname(self, vname, to_namespace=None):
        """
        Utility function that converts variable names to the input namespace.

        Parameters
        ----------
        vname: string
            The variable name.

        to_namespace: string
            The target namespace.

        Returns
        -------

        """
        namespace = vname.split("/")[0]
        if to_namespace is None:
            to_namespace = self.name
        return vname.replace(namespace, to_namespace)

    def set_variables(self, var_dict):
        """
        Set the self's variables to those provided in var_dict. This is e.g. used to restore the best seen parameters
        after training with patience.

        Parameters
        ----------
        var_dict: dict
            Dictionary of the form {var_name: var_value} to assign the variables in the self.

        Returns
        -------
        None.
        """

        with self.graph.as_default():
            if not hasattr(self, 'assign_placeholders'):
                self.assign_placeholders = {v.name: tf.placeholder(v.dtype, shape=v.get_shape()) for v in self.varlist}
                self.assign_ops = {v.name: tf.assign(v, self.assign_placeholders[v.name])
                                   for v in self.varlist}
            to_namespace = list(var_dict.keys())[0].split("/")[0]
            self.session.run(list(self.assign_ops.values()),
                             feed_dict={val: var_dict[self.convert_varname(key, to_namespace)]
                                        for key, val in self.assign_placeholders.items()})

    def train(self, split_train, split_val, Z_obs, patience=30, n_iters=200, print_info=True):
        """
        Train the GCN self on the provided data.

        Parameters
        ----------
        split_train: np.array, shape [n_train,]
            The indices of the nodes used for training

        split_val: np.array, shape [n_val,]
            The indices of the nodes used for validation.

        Z_obs: np.array, shape [N,k]
            All node labels in one-hot form (the labels of nodes outside of split_train and split_val will not be used.

        patience: int, default: 30
            After how many steps without improvement of validation error to stop training.

        n_iters: int, default: 200
            Maximum number of iterations (usually we hit the patience limit earlier)

        print_info: bool, default: True

        Returns
        -------
        None.

        """

        varlist = self.varlist
        self.session.run(self.local_init_op)

        early_stopping = patience

        best_performance = 0
        patience = early_stopping

        feed = {self.node_ids: split_train,
                self.node_labels: Z_obs[split_train]}

        if hasattr(self, 'training'):
            feed[self.training] = True
        for it in range(n_iters):
            _loss, _ = self.session.run([self.loss, self.train_op], feed)
            # print('loss:', _loss)

            f1_micro, f1_macro = eval_class(split_val, self, np.argmax(Z_obs, 1))
            perf_sum = f1_micro + f1_macro
            if perf_sum > best_performance:
                best_performance = perf_sum
                patience = early_stopping
                # var dump to memory is much faster than to disk using checkpoints
                var_dump_best = {v.name: v.eval(self.session) for v in varlist}
            else:
                patience -= 1
            if it > early_stopping and patience <= 0:
                break
        if print_info:
            self.logger.debug('converged after {} iterations'.format(it - patience))
        # Put the best observed parameters back into the self
        self.set_variables(var_dump_best)


def eval_class(ids_to_eval, self, z_obs):
    """
    Evaluate the self's classification performance.

    Parameters
    ----------
    ids_to_eval: np.array
        The indices of the nodes whose predictions will be evaluated.

    self: GCN
        The self to evaluate.

    z_obs: np.array
        The labels of the nodes in ids_to_eval

    Returns
    -------
    [f1_micro, f1_macro] scores

    """
    test_pred = self.predictions.eval(session=self.session, feed_dict={self.node_ids: ids_to_eval}).argmax(1)
    test_real = z_obs[ids_to_eval]

    return f1_score(test_real, test_pred, average='micro'), f1_score(test_real, test_pred, average='macro')


def evision(An, X_obs, W1, b1, W2, b2, name="", with_relu=True, params_dict={'dropout': 0.5}, gpu_id=0,
            seed=123):
    if seed > -1:
        tf.set_random_seed(seed)
    else:
        tf.set_random_seed(np.random.randint(0, 99999))

    if An.format != "csr":
        An = An.tocsr()

    with tf.variable_scope(name) as scope:
        w_init = slim.xavier_initializer

        dropout = params_dict['dropout'] if 'dropout' in params_dict else 0.
        if not with_relu:
            dropout = 0
        # self.dropout = 0

        N, D = X_obs.shape

        # bool placeholder to turn on dropout during training
        training = tf.placeholder_with_default(False, shape=())

        An = tf.SparseTensor(np.array(An.nonzero()).T, An[An.nonzero()].A1, An.shape)
        An = tf.cast(An, tf.float32)
        X_sparse = tf.SparseTensor(np.array(X_obs.nonzero()).T,
                                   X_obs[X_obs.nonzero()].A1.astype(np.float32), X_obs.shape)
        X_dropout = sparse_dropout(X_sparse, 1 - dropout,
                                   (int(X_sparse.values.get_shape()[0]),))
        # only use drop-out during training
        X_comp = tf.cond(training,
                         lambda: X_dropout,
                         lambda: X_sparse) if dropout > 0. else X_sparse

        W1 = tf.constant(W1, dtype=tf.float32, shape=W1.shape)
        b1 = tf.constant(b1, dtype=tf.float32, shape=b1.shape)

        h1 = spdot(An, spdot(X_comp, W1))

        if with_relu:
            h1 = tf.nn.relu(h1 + b1)

        h1_dropout = tf.nn.dropout(h1, 1 - dropout, seed=seed)

        h1_comp = tf.cond(training,
                          lambda: h1_dropout,
                          lambda: h1) if dropout > 0. else h1

        W2 = tf.constant(W2, dtype=tf.float32, shape=W2.shape)
        b2 = tf.constant(b2, dtype=tf.float32, shape=b2.shape)

        logits = spdot(An, dot(h1_comp, W2))
        if with_relu:
            logits += b2
    return logits


class GCN_Batch:
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, X_sparse=True, A_input=False,
                 params_dict={'dropout': 0.5}, gpu_id=0,
                 seed=-1, retrain=False):
        """
        Create a Graph Convolutional Network self in Tensorflow with one hidden layer.

        Parameters
        ----------
        sizes: list
            List containing the hidden and output sizes (i.e. number of classes). E.g. [16, 7]

        An: sp.sparse_matrix, shape [N,N]
            The input adjacency matrix preprocessed using the procedure described in the GCN paper.

        X_obs: sp.sparse_matrix, shape [N,D]
            The node features.

        name: string, default: ""
            Name of the network.

        with_relu: bool, default: True
            Whether there a nonlinear activation function (ReLU) is used. If False, there will also be
            no bias terms, no regularization and no dropout.

        params_dict: dict
            Dictionary containing other self parameters.

        gpu_id: int or None, default: 0
            The GPU ID to be used by Tensorflow. If None, CPU will be used

        seed: int, defualt: -1
            Random initialization for reproducibility. Will be ignored if it is -1.
        """

        self.with_relu = with_relu
        self.graph = tf.Graph()
        self.retrain = retrain
        if seed > -1:
            tf.set_random_seed(seed)

        with self.graph.as_default():

            with tf.variable_scope(name) as scope:
                w_init = slim.xavier_initializer
                self.name = name
                self.n_classes = sizes[-1]

                self.dropout = params_dict['dropout'] if 'dropout' in params_dict else 0.
                if not with_relu:
                    self.dropout = 0

                self.learning_rate = params_dict['learning_rate'] if 'learning_rate' in params_dict else 0.01

                self.weight_decay = params_dict['weight_decay'] if 'weight_decay' in params_dict else 5e-4
                self.N, self.D = X_obs.shape
                self.AXfeatures = tf.placeholder(tf.float32, shape=(None, X_obs.shape[1]))
                self.node_labels = tf.placeholder(tf.float32, shape=(None, sizes[-1]))
                self.batchA = tf.sparse_placeholder(tf.float32)
                _W1 = slim.variable('W1', [self.D, sizes[0]], tf.float32, initializer=w_init())
                _b1 = slim.variable('b1', dtype=tf.float32, initializer=tf.zeros(sizes[0]))
                self.W1 = _W1
                self.b1 = _b1
                tf.summary.histogram('W1', _W1)
                self.h1 = dot(self.AXfeatures, _W1)
                if (with_relu):
                    self.h1 = self.h1 + _b1
                    self.h1 = tf.nn.relu(self.h1)
                _W2 = slim.variable('W2', [sizes[0], sizes[1]], tf.float32, initializer=w_init())
                _b2 = slim.variable('b2', dtype=tf.float32, initializer=tf.zeros(sizes[1]))
                self.W2 = _W2
                self.b2 = _b2
                tf.summary.histogram('W2', _W2)
                self.logits = spdot(self.batchA, dot(self.h1, _W2))
                if (with_relu):
                    self.logits = self.logits + _b2
                self.prediction = tf.nn.softmax(self.logits)

                self.loss_per_node = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                             labels=self.node_labels)
                self.loss = tf.reduce_mean(self.loss_per_node)

                tf.summary.scalar('loss', self.loss)

                # weight decay only on the first layer, to match the original implementation
                if with_relu:
                    self.loss += self.weight_decay * tf.add_n(
                        [tf.nn.l2_loss(v) for v in [self.W1, self.W2]])

                var_l = [self.W1, self.W2]
                # var_l = self.W_list
                # if with_relu:
                #     var_l.extend(self.b_list)
                self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                                  var_list=var_l)

                self.varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
                self.local_init_op = tf.variables_initializer(self.varlist)


                gpu_options = tf.GPUOptions(allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_options)

                log_dir = './log'
                self.merged = tf.summary.merge_all()
                # 写到指定的磁盘路径中
                self.train_writer = tf.summary.FileWriter(log_dir + '/train', self.graph)
                self.test_writer = tf.summary.FileWriter(log_dir + '/test')

                self.session = tf.Session(config=config)
                self.init_op = tf.global_variables_initializer()
                if not retrain:
                    self.session.run(self.init_op)

    def convert_varname(self, vname, to_namespace=None):
        """
        Utility function that converts variable names to the input namespace.

        Parameters
        ----------
        vname: string
            The variable name.

        to_namespace: string
            The target namespace.

        Returns
        -------

        """
        namespace = vname.split("/")[0]
        if to_namespace is None:
            to_namespace = self.name
        return vname.replace(namespace, to_namespace)

    def get_logits(self, u, As, Xs, W_list, b_list, regularizer=False):
        # W_list, b_list = self.session.run([self.W_list, self.b_list])
        # W_list = self.W_list
        # b_list = self.b_list
        if (regularizer):
            self.As_diag = tf.diag_part(As)
            As = As - tf.diag(self.As_diag - 1)
            Ds = tf.reduce_sum(As, 1)
            # Ds_inv_sqrt = tf.diag(tf.pow(Ds, -0.5))
            Ds_inv = tf.diag(tf.pow(Ds, -1))
            As = tf.matmul(Ds_inv, As)
            # As = tf.matmul(Ds_inv_sqrt,tf.matmul(As,Ds_inv_sqrt))
        _h = Xs
        for _w, _b in zip(W_list, b_list):
            _h = tf.matmul(As, tf.matmul(_h, _w))
            if (self.with_relu):
                _h = tf.nn.relu(_h + _b)
        return _h[u]

    def set_variables(self, var_dict):
        """
        Set the self's variables to those provided in var_dict. This is e.g. used to restore the best seen parameters
        after training with patience.

        Parameters
        ----------
        var_dict: dict
            Dictionary of the form {var_name: var_value} to assign the variables in the self.

        Returns
        -------
        None.
        """

        with self.graph.as_default():
            if not hasattr(self, 'assign_placeholders'):
                self.assign_placeholders = {v.name: tf.placeholder(v.dtype, shape=v.get_shape()) for v in self.varlist}
                self.assign_ops = {v.name: tf.assign(v, self.assign_placeholders[v.name])
                                   for v in self.varlist}
            to_namespace = list(var_dict.keys())[0].split("/")[0]
            self.session.run(list(self.assign_ops.values()),
                             feed_dict={val: var_dict[self.convert_varname(key, to_namespace)]
                                        for key, val in self.assign_placeholders.items()})

    def train(self, _An, AX, split_train, split_val, Z, patience=30, n_iters=200, batchsize=1000, print_info=True):
        best_acc = 0
        p_count = 0
        if self.retrain:
            self.load_model()
        for it in range(n_iters):
            for An_batch, AX_batch, Y_batch in iterate_minibatches_listinputs(_An, AX, Z, split_train, min(batchsize, len(split_train))):
                feed = {self.AXfeatures: AX_batch,
                        self.node_labels: Y_batch,
                        self.batchA: tf.SparseTensorValue(np.array(An_batch.nonzero()).T,
                                                          An_batch[An_batch.nonzero()].A1, An_batch.shape)}
                loss, train_op = self.session.run([self.loss, self.train_op], feed_dict=feed)
            #print(it, "train_loss:", loss)
            for An_val, AX_val, Y_val in iterate_minibatches_listinputs(_An, AX, Z, split_val,
                                                                        min(batchsize,len(split_val))):
                feed = {self.AXfeatures: AX_val,
                        self.node_labels: Y_val,
                        self.batchA: tf.SparseTensorValue(np.array(An_val.nonzero()).T,
                                                          An_val[An_val.nonzero()].A1, An_val.shape)}
                val_loss, val_pred = self.session.run([self.loss, self.prediction], feed_dict=feed)
                val_acc = np.sum(val_pred.argmax(1) == Y_val.argmax(1)) / len(split_val)
                #print("val_loss:", val_loss, "val_acc:", val_acc)
                if it == n_iters-1:
                    print("val_loss:", val_loss, "val_acc:", val_acc)
                break
            if (val_acc > best_acc):
                p_count = 0
                W1, W2 = self.session.run([self.W1, self.W2])
                best_W1 = W1
                best_W2 = W2
                best_acc = val_acc
            else:
                p_count += 1
                if (p_count > patience):
                    break
            if (p_count > patience):
                break
    def save_model(self, Path = r'./model/GCN'):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.session, Path+'.ckpt')

    def load_model(self, Path = r'./model/GCN'):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.session, Path+'.ckpt')


    def eval(self, _An, AX, Z, split_unlabeled, batchsize=5000):
        logits = None
        for An_test, AX_test, Y_test in iterate_minibatches_listinputs(_An, AX, Z, split_unlabeled,
                                                                       batchsize, random_shuffle=False):
            feed = {self.AXfeatures: AX_test,
                    self.node_labels: Y_test,
                    self.batchA: tf.SparseTensorValue(np.array(An_test.nonzero()).T,
                                                      An_test[An_test.nonzero()].A1, An_test.shape)}
            logits_batch = self.session.run([self.logits], feed_dict=feed)[0]
            #print(logits_batch)
            if logits is None:
                logits = logits_batch
            else:
                logits = np.vstack((logits, logits_batch))

        return logits

    def pred(self, _An, AX, Z, idx):
        An_expt = _An[idx]
        onehup = np.unique(np.nonzero(An_expt)[1])
        An_test = An_expt[:, onehup]
        AX_test = AX[onehup]
        Y_test = np.reshape(Z[idx],[1,-1])
        #print("An_test.nonzero():",An_test.nonzero(), type(An_test[An_test.nonzero()]))
        feed = {self.AXfeatures: AX_test,
                self.node_labels: Y_test,
                self.batchA: tf.SparseTensorValue(np.array(An_test.nonzero()).T,
                                                  An_test[An_test.nonzero()].A1, An_test.shape)}
        pred = self.session.run(self.prediction, feed_dict=feed)[0]
        label = np.argmax(Z[idx])
        wrong_label = np.argmax(pred - 10 * Z[idx])

        return pred[label] - pred[wrong_label]


def iterate_minibatches_listinputs(An, AX, labels, train_index, batchsize, random_shuffle=True):
    numSamples = len(train_index)
    indices = np.arange(numSamples)
    if random_shuffle:
        np.random.shuffle(indices)
    for start_id in range(0, numSamples, batchsize):
        excerpt = train_index[indices[start_id:min(start_id + batchsize, numSamples)]]
        An_expt = An[excerpt]
        onehup = np.unique(np.nonzero(An_expt)[1])
        yield An_expt[:, onehup], AX[onehup], labels[excerpt]
