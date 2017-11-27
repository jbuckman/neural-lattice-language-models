import dynet
import numpy
import code

class TiedWeightsLSTMBuilder(object):
    def __init__(self, model, dim, layers, layer_devices, input_dim=None):
        assert layers > 0
        self.model = model
        self.dim = dim
        self.layers = layers
        self.layer_devices = layer_devices
        if input_dim is None: self.input_dim = dim
        else:                 self.input_dim = input_dim
        self.Ws = [self.model.add_parameters((self.dim * 3, self.dim + self.input_dim + 1), device=layer_devices[0])] + \
                  [self.model.add_parameters((self.dim * 3, self.dim * 2 + 1), device=layer_devices[li]) for li in range(1,self.layers)]

    def initial_state(self, mb_size=1, dropout=None, nobackprop=False):
        layers = [TiedWeightsLSTMLayer(self.model, self.dim, self.input_dim, self.Ws[0], mb_size, dropout, self.layer_devices[0], nobackprop)] + \
                 [TiedWeightsLSTMLayer(self.model, self.dim, self.dim, self.Ws[layer_i], mb_size, dropout, self.layer_devices[layer_i], nobackprop)
                  for layer_i in range(1,self.layers)]
        for l1, l2 in zip(layers, layers[1:]):
            l1.next_layer = l2
        return layers[0]

class TiedWeightsLSTMLayer(object):
    def __init__(self, model, dim, input_dim, W, minibatch_size, dropout, device, nobackprop=False, init_h_t=None, init_c_t=None):
        self.model = model
        self.dim = dim
        self.input_dim = input_dim
        self.W = dynet.parameter(W)
        self.nobackprop = nobackprop
        if nobackprop: self.W = dynet.nobackprop(self.W)
        self.W_param = W
        self.minibatch_size = minibatch_size
        self.dropout = dropout
        self.device = device

        self.next_layer = None
        if init_h_t is None: self.h_t = dynet.inputVector([0]*self.dim, device=self.device)
        else:                self.h_t = init_h_t
        if init_c_t is None: self.c_t = dynet.inputVector([0]*dim, device=self.device)
        else:                self.c_t = init_c_t
        self.bias = dynet.inputVector([1], device=self.device)

        if self.dropout is not None:
            mask = (1./(1.-self.dropout)) * (numpy.random.uniform(size=[self.input_dim, self.minibatch_size]) > self.dropout)
            self.dropout_mask_x = dynet.inputTensor(mask, batched=True, device=self.device)
            mask = (1./(1.-self.dropout)) * (numpy.random.uniform(size=[self.dim, self.minibatch_size]) > self.dropout)
            self.dropout_mask_h = dynet.inputTensor(mask, batched=True, device=self.device)

    def add_input(self, x_t, mask=None):
        x_t = dynet.to_device(x_t, self.device)
        if self.dropout is None:
            x_t = x_t
            h_t = self.h_t
            bias = self.bias
        else:
            x_t = dynet.cmult(x_t, self.dropout_mask_x)
            h_t = dynet.cmult(self.h_t, self.dropout_mask_h)
            bias = self.bias

        # calculate all information for all gates in one big matrix multiplication
        gates = self.W * dynet.concatenate([x_t, h_t, bias])

        # input gate
        i = dynet.logistic(dynet.pickrange(gates, 0, self.dim))
        # forget gate
        f = 1.0 - i
        # output gate
        o = dynet.logistic(dynet.pickrange(gates, self.dim, self.dim*2))
        # input modulation gate
        g = dynet.tanh(dynet.pickrange(gates, self.dim*2, self.dim*3))
        # cell state
        c_t = dynet.cmult(f, self.c_t) + dynet.cmult(i, g)
        # hidden state
        h_t = dynet.cmult(o, dynet.tanh(c_t))

        if mask is None:
            self.c_t = c_t
            self.h_t = h_t
        else:
            self.c_t = (c_t * mask) + (self.c_t * (1.0-mask))
            self.h_t = (h_t * mask) + (self.h_t * (1.0-mask))

        if self.next_layer is not None:
            self.next_layer.add_input(self.h_t, mask)

    def output(self):
        if self.next_layer is None:
            return self.h_t
        return self.next_layer.output()

    def all_layer_outputs(self):
        if self.next_layer is None:
            return [self.h_t]
        return [self.h_t] + self.next_layer.all_layer_outputs()

    def transduce(self, inps, masks=None):
        if masks is None: masks = [None] * len(inps)
        outs = []
        for inp, mask in zip(inps, masks):
            self.add_input(inp, mask)
            outs.append(self.output())
        return outs

    def clone(self):
        clone = TiedWeightsLSTMLayer(self.model, self.dim, self.input_dim, self.W_param, self.minibatch_size, self.dropout, self.nobackprop, self.h_t, self.c_t)
        if self.next_layer is None:
            return clone
        clone.next_layer = self.next_layer.clone()
        return clone

    def merge(self, other_state):
    # def merge(self, other_state, h_merge_bias, h_merge_R, c_merge_bias, c_merge_R):
        self.h_t = self.h_t * .5 + other_state.h_t * .5
        self.c_t = self.c_t * .5 + other_state.c_t * .5
        # self.h_t = dynet.affine_transform([h_merge_bias, h_merge_R, dynet.concatenate([self.h_t, other_state.h_t])])
        # self.c_t = dynet.affine_transform([c_merge_bias, c_merge_R, dynet.concatenate([self.c_t, other_state.c_t])])
        if self.next_layer is not None:
            self.next_layer.merge(other_state.next_layer)