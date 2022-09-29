from util.common import *
from data_type.enums import LayerType, ActivationType


class ConcernIdentification:

    def feedback_loop(self, layer, x, apply_activation=True):

        for ts in range(layer.timestep):
            x_t = x[ts].reshape(1, layer.number_features)

            if ts == 0:
                x_t = self.propagateThroughRNN(layer, x_t, layer.getHiddenState(ts), apply_activation=apply_activation)
            else:
                x_t = self.propagateThroughRNN(layer, x_t, layer.getHiddenState(ts - 1),
                                               apply_activation=apply_activation)

            layer.setHiddenState(x_t, ts)

        return layer.hidden_state

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True, stop_remvoing=False):
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
        elif layer.type == LayerType.RepeatVector:
            layer.hidden_state = self.repeatVector(layer, x_t)

        elif layer.type == LayerType.Flatten:
            layer.hidden_state = self._flatten(x_t)

        if layer.type == LayerType.RNN and not layer.return_sequence:
            return layer.getHiddenState(layer.timestep - 1)

        if layer.type == LayerType.Dropout:
            return x_t

        return layer.hidden_state

    def _flatten(self, x):
        return x.flatten()

    def repeatVector(self, layer, a):
        c = []
        for x in range(layer.timestep):
            c.append(a)
        c = np.asarray(c)
        return c

    def embeddingLookup(self, layer, x_t=None):
        embed = []
        for _x in x_t:
            embed.append(layer.W[_x])

        return np.asarray(embed)

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

    def propagateThroughDense(self, layer, x_t=None, apply_activation=True):

        x_t = (x_t.dot(layer.W) + layer.B)

        return self.propagateThroughActivation(layer, x_t, apply_activation)

    def propagateThroughActivation(self, layer, x_t, apply_activation=True, ):
        if not apply_activation or layer.activation == ActivationType.Linear:
            return x_t

        if ActivationType.Softmax == layer.activation:
            x_t = x_t.reshape(layer.num_node)
            x_t = softmax(x_t)
        elif ActivationType.Relu == layer.activation:
            x_t[x_t < 0] = 0
        elif ActivationType.Tanh == layer.activation:
            x_t = tanh(x_t)
        elif ActivationType.Sigmoid == layer.activation:
            x_t = sigmoid(x_t)

        return x_t
