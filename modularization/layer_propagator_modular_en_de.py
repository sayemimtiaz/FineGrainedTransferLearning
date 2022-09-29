from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe
from modularization.concern.util import *
from modularization.layer_propagator_modular import LayerPropagatorModular
from util.common import *
from data_type.enums import LayerType, ActivationType

from data_type.constants import Constants

class LayerPropagatorModularEnDe(LayerPropagatorModular):
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

                x_t = self.propagateThroughRNN(layer, x_t, previous_state,ts, apply_activation=apply_activation)

                previous_state = x_t

                layer.setHiddenState(x_t, ts)

        return layer.hidden_state

    def compute_mask(self, layer, x):
        if not layer.mask_zero:
            return None
        return x != 0

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

    def get_decoder_output(self, layers, x, encoder_state,apply_activation=True):
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
                encoder_state=None
            else:
                x = self.propagateThroughLayer(_layer, x, apply_activation=apply_activation)
        return x

    def propagateThroughEncoderDecoder(self, layers, x, y, apply_activation=True):
        encoder_state = self.get_encoder_output(layers, x, apply_activation=apply_activation)
        decoder_state = self.get_decoder_output(layers, y, encoder_state, apply_activation=apply_activation)
        return self.propagateThroughLayer(ConcernIdentificationEnDe.get_output_layer(layers), decoder_state,
                                          apply_activation=True)

    def propagateThroughLayer(self, layer, x_t=None, apply_activation=True, mask=None,
                              initial_state=None):
        if layer.type == LayerType.RNN:
            layer.hidden_state = self.feedback_loop(layer, x_t,
                                                    apply_activation=apply_activation,
                                                    mask=mask, initial_state=initial_state)

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

    def embeddingLookup(self, layer, x_t=None):
        embed = []
        for _x in x_t:
            embed.append(layer.DW[_x])

        embed = np.asarray(embed)

        mask = self.compute_mask(layer, x_t)

        return embed, mask

    def propagateThroughRNN(self, layer, x_t=None, h_t_previous=None, timestep=None, apply_activation=True):
        if Constants.UNROLL_RNN and not self.encoderLoop:
            x_t = (x_t.dot(layer.DW[timestep]) +
                   h_t_previous.dot(layer.DU[timestep]) +
                   layer.DB[timestep])
        else:
            x_t = (x_t.dot(layer.DW) +
                   h_t_previous.dot(layer.DU) +
                   layer.DB)

        return self.propagateThroughActivation(layer, x_t, apply_activation)
