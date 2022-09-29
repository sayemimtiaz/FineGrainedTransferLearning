from modularization.concern.util import *
from util.common import *
from data_type.enums import LayerType, ActivationType


class ConcernIdentificationEnDe:
    encoderLoop = True

    def feedback_loop(self, layer, x, apply_activation=True, mask=None, initial_state=None):
        previous_state = initial_state
        if initial_state is None:
            previous_state = layer.getHiddenState(0)
        for ts in range(layer.timestep):
            if mask is not None and mask[ts] == False:
                layer.setHiddenState(previous_state, ts)
            else:

                x_t = x[ts].reshape(1, layer.number_features)

                x_t = self.propagateThroughRNN(layer, x_t, previous_state, apply_activation=apply_activation)

                previous_state = x_t

                layer.setHiddenState(x_t, ts)

            if layer.unrolled or layer.return_sequence:
                updateNodeConcernForRnnAtTimestep(layer, ts)

        return layer.hidden_state

    def compute_mask(self, layer, x):
        if not layer.mask_zero:
            return None
        return x != 0

    @staticmethod
    def get_output_layer(layers):
        for layer in layers:
            if layer.type == LayerType.Input or layer.type == LayerType.Dropout:
                continue
            if 'output' in layer.name.lower():
                return layer
        return layers[-1]

    @staticmethod
    def get_encoder_layers(layers):
        out = []
        for layer in layers:
            if layer.type == LayerType.Input or layer.type == LayerType.Dropout:
                continue
            if 'encoder' in layer.name.lower():
                out.append(layer)
        return out

    @staticmethod
    def is_decoder_layer(layer):
        if 'decoder' in layer.name.lower():
            return True
        return False

    @staticmethod
    def get_decoder_layers(layers):
        out = []
        for layer in layers:
            if layer.type == LayerType.Input or layer.type == LayerType.Dropout:
                continue
            if 'decoder' in layer.name.lower():
                out.append(layer)
        return out

    def get_encoder_output(self, layers, x, apply_activation=True):
        self.encoderLoop = True
        layers = ConcernIdentificationEnDe.get_encoder_layers(layers)
        mask = None
        for layerNo, _layer in enumerate(layers):
            if _layer.type == LayerType.Embedding:
                x, mask = self.propagateThroughLayer(_layer, x,
                                                     apply_activation=apply_activation)
            elif _layer.type == LayerType.RNN:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation, mask=mask)
            else:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation)
        return x

    def get_decoder_output(self, layers, x, encoder_state, apply_activation=True):
        self.encoderLoop = False
        layers = ConcernIdentificationEnDe.get_decoder_layers(layers)
        mask = None
        for layerNo, _layer in enumerate(layers):
            if _layer.type == LayerType.Embedding:
                x, mask = self.propagateThroughLayer(_layer, x,
                                                     apply_activation=apply_activation)
            elif _layer.type == LayerType.RNN:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation,
                                               mask=mask, initial_state=encoder_state)
                encoder_state = None
            else:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation)
        return x

    def propagateThroughEncoderDecoder(self, layers, x, y):
        encoder_state = self.get_encoder_output(layers, x)
        decoder_state = self.get_decoder_output(layers, y, encoder_state)
        return self.propagateThroughLayer(ConcernIdentificationEnDe.get_output_layer(layers), decoder_state,
                                          apply_activation=True)

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True, mask=None, initial_state=None):
        if layer.type == LayerType.RNN:
            layer.hidden_state = self.feedback_loop(layer, x_t,
                                                    apply_activation=apply_activation,
                                                    mask=mask, initial_state=initial_state)
            if not layer.unrolled and not layer.return_sequence:
                updateNodeConcernForRnn(layer)

        elif layer.type == LayerType.Dense:

            layer.hidden_state = self.propagateThroughDense(layer, x_t=x_t,
                                                            apply_activation=apply_activation)

            if layer.last_layer:
                updateNodeConcernForOutputLayer(layer)
            else:
                updateNodeConcernForRegular(layer)

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

        embed = np.asarray(embed)

        mask = self.compute_mask(layer, x_t)

        return embed, mask

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
            x_t = np.tanh(x_t)
        elif ActivationType.Sigmoid == layer.activation:
            x_t = sigmoid(x_t)

        return x_t
