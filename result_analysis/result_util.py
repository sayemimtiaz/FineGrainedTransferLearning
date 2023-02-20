import csv

from result_analysis.result_type import Observation


def parseLine(line):
    architecture = line[0]
    task = line[1]
    classifierType = line[2]
    transferType = line[3]
    alpha = line[4]
    accuracy = float(line[7])
    std = float(line[8])
    minAcc = float(line[9])
    maxAcc = float(line[10])
    elapsed = float(line[11])
    delRate = float(line[12])

    r = Observation()
    r.architecture = architecture
    r.transferType = transferType
    r.classifierType = classifierType
    r.accuracy = accuracy
    r.alpha = alpha
    r.minAcc = minAcc
    r.maxAcc = maxAcc
    r.elapsed = elapsed
    r.delRate = delRate
    r.std = std
    r.task = task
    return r


def getAllArchitectures(file):
    archs = set()
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line)>0:
                pr = parseLine(line)
                archs.add(pr.architecture)
    return list(archs)


def getBaseline(file, architecture):
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line)>0:
                pr = parseLine(line)
                if pr.architecture == architecture and pr.transferType.lower() == 'baseline':
                    return pr
    return None


def getTafeObservation(file, architecture):
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line)>0:
                pr = parseLine(line)
                if pr.architecture == architecture and pr.transferType.lower() != 'baseline':
                    yield pr
    return None
