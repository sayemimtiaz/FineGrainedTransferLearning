from util.common import softmax, sigmoid
import numpy as np
from data_type.enums import LayerType, ActivationType


class LayerPropagator:

    def feedback_loop(self, layer, x, apply_activation=True):
        for ts in range(layer.timestep):
            x_t = x[ts].reshape(1, layer.number_features)
            if ts == 0:
                x_t = self.propagateThroughRNN(layer, x_t,
                                               layer.getHiddenState(ts), apply_activation=apply_activation)
            else:
                x_t = self.propagateThroughRNN(layer, x_t,
                                               layer.getHiddenState(ts - 1), apply_activation=apply_activation)
            layer.setHiddenState(x_t, ts)

        return layer.hidden_state

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True):
        if layer.type == LayerType.RNN:
            layer.hidden_state = self.feedback_loop(layer, x_t, apply_activation=apply_activation)

        elif layer.type == LayerType.Dense:

            layer.hidden_state = self.propagateThroughDense(layer, x_t=x_t,
                                                            apply_activation=apply_activation)
        elif layer.type == LayerType.Embedding:

            return self.embeddingLookup(layer, x_t=x_t)

        elif layer.type == LayerType.TimeDistributed:

            layer.hidden_state = self.propagateThroughTimeDistributed(layer, x_t=x_t,
                                                                      apply_activation=apply_activation)
        elif layer.type == LayerType.Activation:

            layer.hidden_state = self.propagateThroughActivation(layer, x_t=x_t,
                                                                 apply_activation=apply_activation)
        elif layer.type == LayerType.RepeatVector:
            layer.hidden_state = self.repeatVector(layer, x_t)

        elif layer.type == LayerType.Flatten:
            layer.hidden_state = self._flatten(x_t)

        if layer.type == LayerType.RNN and not layer.return_sequence:
            return layer.getHiddenState(layer.timestep-1)
        
        return layer.hidden_state

    def propagateThroughTimeDistributed(self, layer, x_t=None, apply_activation=True):
        output = []
        for ts in range(len(x_t)):
            h_t = self.propagateThroughDense(layer, x_t[ts], apply_activation=apply_activation)
            output.append(h_t)

        return np.asarray(output)

    def propagateThroughRNN(self, layer, x_t=None, h_t_previous=None, apply_activation=True):
        x_t = (x_t.dot(layer.W) +
               h_t_previous.dot(layer.U) +
               layer.B)

        return self.propagateThroughActivation(layer, x_t, apply_activation)

    def _flatten(self, x):
        return x.flatten()

    def repeatVector(self, layer, a):
        c = []
        for x in range(layer.timestep):
            c.append(a)
        c = np.asarray(c)
        return c

    # def propagateThroughLSTM(self, weight, x_t, h_t_previous, c_t_previous):
    #
    #     warr, uarr, barr = self.WRecurrnentForX, self.WReccurentForHiddenState, self.BRecurrent
    #     s_t = (x_t.dot(warr) + h_t_previous.dot(uarr) + barr)
    #     hunit = uarr.shape[0]
    #     i = sigmoid(s_t[:, :hunit])
    #     f = sigmoid(s_t[:, 1 * hunit:2 * hunit])
    #     _c = np.tanh(s_t[:, 2 * hunit:3 * hunit])
    #     o = sigmoid(s_t[:, 3 * hunit:])
    #     c_t = i * _c + f * c_t_previous
    #     h_t = o * np.tanh(c_t)
    #     return h_t, c_t

    def propagateThroughDense(self, layer, x_t=None, apply_activation=True):

        x_t = (x_t.dot(layer.W) + layer.B)

        return self.propagateThroughActivation(layer, x_t, apply_activation)

    def embeddingLookup(self, layer, x_t=None):
        embed = []
        for _x in x_t:
            embed.append(layer.W[_x])

        return np.asarray(embed)

    def propagateThroughActivation(self, layer, x_t, apply_activation=True, ):
        if not apply_activation or layer.activation == ActivationType.Linear:
            return x_t

        if ActivationType.Softmax == layer.activation:
            x_t = x_t.reshape(layer.num_node)
            x_t = softmax(x_t)
        elif ActivationType.Relu == layer.activation:
            x_t[x_t < 0] = 0
        elif ActivationType.Tanh == layer.activation:
            x_t = np.tanh(x_t)
        elif ActivationType.Sigmoid == layer.activation:
            x_t = sigmoid(x_t)

        return x_t
