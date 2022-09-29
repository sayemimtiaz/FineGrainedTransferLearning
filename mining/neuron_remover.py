from modularization.concern.util import removeNode


def removeNeurons(concern, frozenNeurons, invNeuronMap):
    frozenNeurons = list(frozenNeurons)

    for nodeId in frozenNeurons:
        (layerNo, nodeNum) = invNeuronMap[nodeId]

        removeNode(concern[layerNo], nodeNum)


def union_all_pattern(frequent):
    gSet = set()
    for fsi in range(len(frequent)):
        fs = frequent.iloc[[fsi]]

        frozenNeurons = set(fs.itemsets.values[0])
        gSet = gSet.union(frozenNeurons)
    return gSet


def unique_all_pattern(A, B):
    gSet = set()
    for fsi in range(len(B)):
        fs = B.iloc[[fsi]]

        frozenNeurons = set(fs.itemsets.values[0])
        gSet = gSet.union(frozenNeurons)
    return A - gSet


def differentialAnalysis(frequentPos, frequentNeg):
    gSet = []
    for fsia in range(len(frequentPos)):
        a = set(frequentPos.iloc[[fsia]].itemsets.values[0])

        removeFlag = False
        for fsib in range(len(frequentNeg)):
            b = set(frequentNeg.iloc[[fsib]].itemsets.values[0])
            if a.issubset(b):
                removeFlag = True

        if not removeFlag:
            for fsib in range(len(frequentNeg)):
                b = set(frequentNeg.iloc[[fsib]].itemsets.values[0])
                if b.issubset(a):
                    a = a - b
            if len(a) > 0:
                gSet.append(a)
    return gSet


def differentialAnalysis2(frequentPos, frequentNeg):
    gSet = []
    for fsia in range(len(frequentPos)):
        a = set(frequentPos.iloc[[fsia]].itemsets.values[0])

        removeFlag = False
        for fsib in range(len(frequentNeg)):
            b = set(frequentNeg.iloc[[fsib]].itemsets.values[0])
            sim = len(a.intersection(b)) / len(a)
            if sim >= 0.9:
                removeFlag = True
                break

        if not removeFlag:
            gSet.append(a)
    return gSet
