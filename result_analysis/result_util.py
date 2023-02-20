import csv

from result_analysis.result_type import Observation


def parseLine(line):
    studyType = None
    if len(line) == 12:
        archStart = 0
    else:
        studyType = line[0]
        archStart = 1

    architecture = line[archStart]
    task = line[archStart + 1]
    classifierType = line[archStart + 2]
    transferType = line[archStart + 3]
    alpha = line[archStart + 4]
    accuracy = float(line[archStart + 7])
    std = float(line[archStart + 8])
    minAcc = float(line[archStart + 9])
    maxAcc = float(line[archStart + 10])
    elapsed = float(line[archStart + 11])
    delRate = float(line[archStart + 12])

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
    r.studyType = studyType
    return r


def getAllArchitectures(file):
    archs = set()
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line) > 0:
                pr = parseLine(line)
                archs.add(pr.architecture)
    return list(archs)


def getBaseline(file, architecture):
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line) > 0:
                pr = parseLine(line)
                if pr.architecture == architecture and pr.transferType.lower() == 'baseline':
                    return pr
    return None


def getTafeObservation(file, architecture):
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line) > 0:
                pr = parseLine(line)
                if pr.architecture == architecture and pr.transferType.lower() != 'baseline':
                    yield pr
    return None


def getAllAplha(file):
    archs = set()
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line) > 0:
                pr = parseLine(line)
                if pr.alpha != 'None':
                    archs.add(pr.alpha)
    return list(archs)

def getObsForAblationStudy(file, architecture, alpha, type='ablation'):
    with open(file, 'r') as input:
        reader = csv.reader(input)
        next(reader)
        for line in reader:
            if len(line) > 0:
                pr = parseLine(line)
                if pr.architecture == architecture and pr.transferType.lower() != 'baseline' \
                        and pr.alpha == alpha and pr.studyType==type:
                    return pr
