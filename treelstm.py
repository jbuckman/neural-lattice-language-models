import dynet
import numpy
import util

class WeightedTreeLSTMBuilder(object):
    def __init__(self, model, dim, layers, layer_devices, input_dim=None):
        assert layers > 0
        self.model = model
        self.dim = dim
        self.layers = layers
        self.layer_devices = layer_devices
        if input_dim is None: self.input_dim = dim
        else:                 self.input_dim = input_dim
        self.Ws = [self.model.add_parameters((self.dim * 2, self.dim + self.input_dim + 1), device=layer_devices[0])] + \
                  [self.model.add_parameters((self.dim * 2, self.dim * 2 + 1), device=layer_devices[li]) for li in range(1, self.layers)]
        self.Wfs = [self.model.add_parameters((self.dim, self.input_dim + 1), device=layer_devices[0])] + \
                   [self.model.add_parameters((self.dim, self.dim + 1), device=layer_devices[li]) for li in range(1, self.layers)]
        self.Ufs = [self.model.add_parameters((self.dim, self.dim), device=layer_devices[li]) for li in range(self.layers)]

        self.dropout = False
        self.dropout_mask_x = None
        self.dropout_mask_h = None
        self.path_dropout = False

    def initialize_dropout(self, dropout, mb_size=1):
        mask = (1. / (1. - dropout)) * (
        numpy.random.uniform(size=[self.input_dim, mb_size]) > dropout)
        self.dropout_mask_x = [dynet.inputTensor(mask, batched=True, device=self.layer_devices[0])]

        mask = (1. / (1. - dropout)) * (
        numpy.random.uniform(size=[self.dim, mb_size]) > dropout)
        self.dropout_mask_h = [dynet.inputTensor(mask, batched=True, device=self.layer_devices[0])]

        for li in range(1, self.layers):
            mask = (1. / (1. - dropout)) * (
                numpy.random.uniform(size=[self.dim, mb_size]) > dropout)
            layer_dropout_mask_x = dynet.inputTensor(mask, batched=True, device=self.layer_devices[li])
            self.dropout_mask_x.append(layer_dropout_mask_x)

            mask = (1. / (1. - dropout)) * (
                numpy.random.uniform(size=[self.dim, mb_size]) > dropout)
            layer_dropout_mask_h = dynet.inputTensor(mask, batched=True, device=self.layer_devices[li])
            self.dropout_mask_h.append(layer_dropout_mask_h)

        self.dropout = True

    def disable_dropout(self):
        self.dropout = False
        self.dropout_mask_x = [None]*self.layers
        self.dropout_mask_h = [None]*self.layers

    def initialize_path_dropout(self): self.path_dropout = True
    def disable_path_dropout(self): self.path_dropout = False

    def fresh_state(self, init_to_zero=False):
        layers = [WeightedTreeLSTMLayer(self.model, self.dim, self.Ws[layer_i], self.Wfs[layer_i], self.Ufs[layer_i],
                                        self.dropout, self.dropout_mask_x[layer_i], self.dropout_mask_h[layer_i],
                                        self.path_dropout, self.layer_devices[layer_i], init_to_zero)
                  for layer_i in range(self.layers)]
        for l1, l2 in zip(layers, layers[1:]):
            l1.next_layer = l2
        return layers[0]

class WeightedTreeLSTMLayer(object):
    def __init__(self, model, dim, W, Wf, Uf, dropout, dropout_mask_x, dropout_mask_h, path_dropout, device, init_to_zero=False):
        self.model = model
        self.device = device
        self.dim = dim
        self.W = dynet.parameter(W)
        self.Wf = dynet.parameter(Wf)
        self.Uf = dynet.parameter(Uf)
        self.bias = dynet.inputVector([1], device=self.device)
        self.h_t = None
        self.c_t = None

        self.next_layer = None
        self.h_t_sources = []
        self.c_t_sources = []
        self.weights = []
        if init_to_zero:
            self.h_t_sources = [dynet.vecInput(dim, device=self.device)]
            self.c_t_sources = [dynet.vecInput(dim, device=self.device)]
            self.weights = [dynet.scalarInput(0.0, device=self.device)]

        self.dropout = dropout
        self.dropout_mask_x = None
        self.dropout_mask_h = None
        if self.dropout:
            self.dropout_mask_x = dropout_mask_x
            self.dropout_mask_h = dropout_mask_h
        self.path_dropout = path_dropout
        self.path_selected = None

    def add_history(self, c_t_stack, h_t_stack, weight):
        self.c_t_sources.append(c_t_stack.pop(0))
        self.h_t_sources.append(h_t_stack.pop(0))
        self.weights.append(weight)
        if self.next_layer:
            self.next_layer.add_history(c_t_stack, h_t_stack, weight)

    def get_path(self, weights=None):
        if self.path_selected is None:
            assert weights is not None
            if len(weights) == 1: self.path_selected = 0
            else: self.path_selected = util.weightedChoice(weights, range(len(weights)), apply_softmax=True)
        return self.path_selected

    def concat_weights(self):
        self.weights = dynet.nobackprop(dynet.concatenate(self.weights))
        if self.next_layer is not None: self.next_layer.concat_weights()

    def apply_gumbel_noise_to_weights(self, temperature=1.0, noise=None):
        shape, batch = self.weights.dim()
        if shape == (1,): return
        if noise is None: noise = dynet.random_gumbel(shape, batch_size=batch)
        self.weights += noise
        if temperature != 1.0: self.weights *= 1./temperature
        if self.next_layer is not None: self.next_layer.apply_gumbel_noise_to_weights(temperature, noise)

    def weights_to_argmax(self):
        shape, batch = self.weights.dim()
        if shape == (1,): return
        m_is = numpy.argmax(self.weights.npvalue(), 0)
        if batch == 1: self.weights = dynet.inputTensor([-99999 if i != m_is else 99999 for i in range(shape[0])], device=self.device)
        else: self.weights = dynet.inputTensor([[-99999 if i != m_i else 99999 for m_i in m_is] for i in range(shape[0])], batched=True, device=self.device)
        if self.next_layer is not None: self.next_layer.weights_to_argmax()

    def calculate_h_t(self):
        if self.h_t is None:
            if len(self.h_t_sources) == 1:
                self.h_t = self.h_t_sources[0]
            elif self.path_dropout:
                self.h_t = self.h_t_sources[self.get_path([w.scalar_value() for w in self.weights])]
            else:
                self.h_t = dynet.concatenate_cols(self.h_t_sources) * dynet.to_device(dynet.softmax(self.weights), self.device)
        return self.h_t

    def calculate_c_t(self):
        if self.c_t is None:
            if len(self.c_t_sources) == 1:
                self.c_t = self.c_t_sources[0]
            elif self.path_dropout:
                self.c_t = self.c_t_sources[self.get_path([w.scalar_value() for w in self.weights])]
            else:
                self.c_t = dynet.concatenate_cols(self.c_t_sources) * dynet.to_device(dynet.softmax(self.weights), self.device)
        return self.c_t

    def add_input(self, x_t):
        x_t = dynet.to_device(x_t, self.device)
        h_t = self.calculate_h_t()

        if self.dropout:
            x_t = dynet.cmult(x_t, self.dropout_mask_x)
            h_t = dynet.cmult(h_t, self.dropout_mask_h)

        # bias
        bias = self.bias

        # calculate all information for all gates in one big matrix multiplication
        gates = self.W * dynet.concatenate([x_t, h_t, bias])

        # input gate
        # i = dynet.logistic(dynet.pickrange(gates, 0, self.dim))
        # output gate
        # o = dynet.logistic(dynet.pickrange(gates, self.dim, self.dim*2))
        # input modulation gate
        # g = dynet.tanh(dynet.pickrange(gates, self.dim*2, self.dim*3))

        # output gate
        o = dynet.logistic(dynet.pickrange(gates, 0, self.dim))
        # input modulation gate
        g = dynet.tanh(dynet.pickrange(gates, self.dim, self.dim*2))

        # forget gate
        Wfx = self.Wf*dynet.concatenate([x_t, bias])
        if len(self.h_t_sources) == 1 or self.path_dropout:
            if len(self.h_t_sources) == 1: idx = 0
            else: idx = self.get_path()
            c_t = self.c_t_sources[idx]

            f_k = dynet.logistic(Wfx + self.Uf*h_t)

            # input gate
            i = 1. - f_k

            # cell state
            c_t = dynet.cmult(f_k, c_t) + dynet.cmult(i, g)
        else:
            weights = dynet.to_device(dynet.softmax(self.weights), self.device)
            if self.dropout: f_k = [dynet.logistic(Wfx + self.Uf*dynet.cmult(h, self.dropout_mask_h))*w for h, w in zip(self.h_t_sources, weights)]
            else: f_k = [dynet.logistic(Wfx + self.Uf*h)*w for h, w in zip(self.h_t_sources, weights)]

            # input gate
            i = 1. - dynet.esum(f_k)

            # cell state
            c_t = dynet.esum([dynet.cmult(f, c) for f, c in zip(f_k, self.c_t_sources)]) + dynet.cmult(i, g)

        # hidden state
        h_t = dynet.cmult(o, dynet.tanh(c_t))

        if self.next_layer is not None:
            c_stack, h_stack = self.next_layer.add_input(h_t)
            return [c_t] + c_stack, [h_t] + h_stack
        else:
            return [c_t], [h_t]

    def output(self):
        if self.next_layer is None:
            return self.calculate_h_t()
        else:
            return self.next_layer.output()

    def all_layer_outputs(self):
        if self.next_layer is None:
            return [self.calculate_h_t()]
        return [self.calculate_h_t()] + self.next_layer.all_layer_outputs()

    def all_layer_states(self):
        if self.next_layer is None:
            return [self.calculate_c_t()]
        return [self.calculate_c_t()] + self.next_layer.all_layer_outputs()