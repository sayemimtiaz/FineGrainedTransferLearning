import math

from util.common import isNodeActive, isIntrinsicallyTrainableLayer


def updateNodeConcernForRnn(layer):
    for nodeNum in range(layer.num_node):
        if not isNodeActive(layer, nodeNum):
            layer.inactive_count[nodeNum] += 1
        else:
            layer.active_count[nodeNum] += 1


def updateNodeConcernForRefLayer(updateLayer, referenceLayer, current_timestep=0, unrolled=False):
    for nodeNum in range(referenceLayer.num_node):
        if not unrolled:
            if not isNodeActive(referenceLayer, nodeNum, timestep=current_timestep):
                updateLayer.inactive_count[nodeNum] += 1
            else:
                updateLayer.active_count[nodeNum] += 1
        else:
            if not isNodeActive(referenceLayer, nodeNum, timestep=current_timestep):
                updateLayer.inactive_count[current_timestep][nodeNum] += 1
            else:
                updateLayer.active_count[current_timestep][nodeNum] += 1


def updateNodeConcernForRnnAtTimestep(layer, current_timestep):
    for nodeNum in range(layer.num_node):
        if not isNodeActive(layer, nodeNum, timestep=current_timestep):
            if layer.unrolled:
                layer.inactive_count[current_timestep][nodeNum] += 1
            else:
                layer.inactive_count[nodeNum] += 1
        else:
            if layer.unrolled:
                layer.active_count[current_timestep][nodeNum] += 1
            else:
                layer.active_count[nodeNum] += 1


def updateNodeConcernForRegular(layer):
    for nodeNum in range(layer.num_node):
        if not isNodeActive(layer, nodeNum):
            # layer.DW[:, nodeNum] = [0 for x in layer.DW[:, nodeNum]]
            # layer.DB[nodeNum] = 0
            layer.inactive_count[nodeNum] += 1
    else:
        # layer.DW[:, nodeNum] = layer.W[:, nodeNum]
        # layer.DB[nodeNum] = layer.B[nodeNum]
        layer.active_count[nodeNum] += 1
        # layer.node_sum[nodeNum] += layer.hidden_state[0, nodeNum]


def updateNodeConcernForOutputLayer(layer):
    # Assuming softmax
    # for nodeNum in range(layer.output_shape):
    #     for edgeNum in range(0, len(layer.W[:, nodeNum])):
    #         if layer.W[edgeNum, nodeNum] < 0:
    #             layer.DW[edgeNum, nodeNum] = max(layer.DW[edgeNum, nodeNum], layer.W[edgeNum, nodeNum])
    #         else:
    #             layer.DW[edgeNum, nodeNum] = min(layer.DW[edgeNum, nodeNum], layer.W[edgeNum, nodeNum])
    #     layer.DB[nodeNum] = layer.B[nodeNum]
    for nodeNum in range(layer.num_node):
        layer.DW[:, nodeNum] = layer.W[:, nodeNum]
        layer.DB[nodeNum] = layer.B[nodeNum]


def updateNodeConcernForOutputLayerForRefLayer(updateLayer, referenceLayer, timestep, unrolled=False):
    for nodeNum in range(updateLayer.num_node):
        if not unrolled:
            updateLayer.DW[:, nodeNum] = referenceLayer.W[:, nodeNum]
            updateLayer.DB[nodeNum] = referenceLayer.B[nodeNum]
        else:
            updateLayer.DW[timestep][:, nodeNum] = referenceLayer.W[:, nodeNum]
            updateLayer.DB[timestep][nodeNum] = referenceLayer.B[nodeNum]


def isNodeRemoved(layer, current_timestep=None, nodeNum=None):
    if layer.unrolled and current_timestep is not None:
        if layer.DB[current_timestep][nodeNum] != 0:
            return False
        for x in layer.DW[current_timestep][:, nodeNum]:
            if x != 0:
                return False

    else:

        if layer.DB[nodeNum] != 0:
            return False
        for x in layer.DW[:, nodeNum]:
            if x != 0:
                return False
    return True


def removeNode(layer, nodeNum=None):
    layer.DW[:, nodeNum] = [0 for x in layer.DW[:, nodeNum]]
    if layer.DU is not None:
        layer.DU[:, nodeNum] = [0 for x in layer.DU[:, nodeNum]]
        layer.DU[nodeNum, :] = [0 for x in layer.DU[nodeNum, :]]
    layer.DB[nodeNum] = 0

    if layer.next_layer is not None and isIntrinsicallyTrainableLayer(
            layer.next_layer) and not layer.next_layer.last_layer:
        layer.next_layer.DW[nodeNum, :] = [0 for x in layer.next_layer.DW[nodeNum, :]]


def keepNode(layer, nodeNum=None):
    layer.DW[:, nodeNum] = layer.W[:, nodeNum]
    if layer.DU is not None:
        layer.DU[:, nodeNum] = layer.U[:, nodeNum]
        layer.DU[nodeNum, :] = layer.U[nodeNum, :]
    layer.DB[nodeNum] = layer.B[nodeNum]

    if layer.next_layer is not None and isIntrinsicallyTrainableLayer(layer.next_layer):
        layer.next_layer.DW[nodeNum, :] = layer.next_layer.W[nodeNum, :]


def removeConfusingNeurons(layerOld, layerNew, layerOldNeg, medianThreshold=0.0, maxRemove=0.3):
    d = {}
    for nodeNum in range(layerOld.num_node):
        d[nodeNum] = layerNew.median_node_val[:, nodeNum]
        # d[nodeNum] = layerOld.median_node_val[:, nodeNum]
        # d[nodeNum] = layerOldNeg.median_node_val[:, nodeNum]

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1], reverse=True)}

    removeCount = 0
    keepCount = 0
    for nodeNum in d.keys():
        removeFlag = False

        if layerNew.median_node_val[:, nodeNum] > layerOld.median_node_val[:, nodeNum] \
                and layerNew.median_node_val[:, nodeNum] > \
                layerOldNeg.median_node_val[:, nodeNum]:
            removeFlag = True

        # if layerNew.median_node_val[:, nodeNum] < layerOld.median_node_val[:, nodeNum] \
        #         and layerOld.median_node_val[:, nodeNum] > \
        #         layerOldNeg.median_node_val[:, nodeNum]:
        #     removeFlag = True

        # if layerNew.median_node_val[:, nodeNum] < layerOldNeg.median_node_val[:, nodeNum] \
        #         and layerOld.median_node_val[:, nodeNum] < \
        #         layerOldNeg.median_node_val[:, nodeNum]:
        #     removeFlag = True

        if (keepCount + removeCount) > 0 and (removeCount / (keepCount + removeCount)) >= maxRemove:
            removeFlag = False

        if removeFlag:
            removeCount += 1
            removeNode(layerOld, nodeNum)
        else:
            keepCount += 1
            keepNode(layerOld, nodeNum)


def createTopNeuronMask(targetLayer, compareLayer1, compareLayer2, maxActiveThrehold=0.1):
    d = {}
    for nodeNum in range(targetLayer.num_node):
        d[nodeNum] = targetLayer.median_node_val[:, nodeNum]

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1], reverse=True)}

    removeCount = 0
    keepCount = 0
    for nodeNum in d.keys():
        removeFlag = False

        if targetLayer.median_node_val[:, nodeNum] > compareLayer1.median_node_val[:, nodeNum] \
                and targetLayer.median_node_val[:, nodeNum] > \
                compareLayer2.median_node_val[:, nodeNum]:
            removeFlag = True

        if (keepCount + removeCount) > 0 and (removeCount / (keepCount + removeCount)) >= maxActiveThrehold:
            removeFlag = False

        if removeFlag:
            removeCount += 1
            targetLayer.most_active_neuron_mask[:, nodeNum] = True
        else:
            keepCount += 1
            targetLayer.most_active_neuron_mask[:, nodeNum] = False


def pruneTopWeights(layer, pruneRate=0.1):
    for nodeNum in range(layer.num_node):

        d = {}
        for prevNode in range(layer.W.shape[0]):
            d[prevNode] = layer.weight_importance_count[:, prevNode]

        d = {k: v for k, v in
             sorted(d.items(), key=lambda item: item[1], reverse=True)}

        removeUntil = layer.W.shape[0] * pruneRate
        removeUntil = min(removeUntil, 1)
        count = 0
        for prevNode in d.keys():
            if d[prevNode] > 0 and count < removeUntil:
                layer.DW[prevNode, nodeNum] = 0
            count += 1


def removeNeurons2(layerPos, layerNeg, activeThreshold=0.9, maxRemove=1.0):
    d = {}
    for nodeNum in range(layerPos.num_node):
        # activeRatio = (layerPos.median_node_val[:, nodeNum] / layerNeg.median_node_val[:, nodeNum])
        # activeRatio+=layerPos.median_node_val[:, nodeNum]
        d[nodeNum] = layerPos.active_count[:, nodeNum]
        # d[nodeNum] = activeRatio

    d = {k: v for k, v in
         sorted(d.items(), key=lambda item: item[1], reverse=True)}

    removeCount = 0
    keepCount = 0
    removed_neurons = set()
    for nodeNum in d.keys():
        removeFlag = False

        activeRatio = (layerPos.active_count[:, nodeNum] - layerNeg.active_count[:, nodeNum])

        if activeRatio > 0.0:
            removeFlag = True

        if removeCount / layerPos.num_node >= maxRemove:
            removeFlag = False

        if removeFlag:
            removeCount += 1
            removeNode(layerPos, nodeNum)
            # removed_neurons.add((layerPos.layer_serial, nodeNum, activeRatio[0]))
            removed_neurons.add((layerPos.layer_serial, nodeNum, layerPos.median_node_val[:, nodeNum][0]))
        else:
            keepCount += 1
            keepNode(layerPos, nodeNum)
    return removed_neurons


def removeNeurons3(layerNum, layerPos, layerNeg, removeThreshold=0.0, discrimnating_set=None):
    removed_neurons = set()
    for nodeNum in range(layerPos.num_node):
        if layerNum not in discrimnating_set or nodeNum not in discrimnating_set[layerNum]:
            continue

        if layerPos.active_count[:, nodeNum] <= layerNeg.active_count[:, nodeNum]:
            continue

        removeNode(layerPos, nodeNum)
        removed_neurons.add((layerPos.layer_serial, nodeNum, tuple(discrimnating_set[layerNum][nodeNum])))
        # removed_neurons.add((layerPos.layer_serial, nodeNum, layerPos.median_node_val[:, nodeNum][0]))

    return removed_neurons
