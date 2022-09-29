import time
import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.frequent_patterns import fpmax

from util.common import filter_dataframe, filter_dataframe_by_multiple_condition


def find_growth_set(dataset, min_support=0.5):
    start_time = time.time()

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = fpgrowth(df, min_support=min_support, use_colnames=True)

    # print('Time to find Max frequent itemset')
    # print("--- %s seconds ---" % (time.time() - start_time))
    return frequent


def get_association_rules(frequent, min_lift=1.0, min_confidence=0.95):
    frequent = association_rules(frequent, metric="confidence", min_threshold=min_confidence)
    frequent = frequent[frequent['lift'] > min_lift]

    rules=[]
    vis=[]
    for index, row in frequent.iterrows():
        ant=set(row.antecedents)
        con=set(row.consequents)
        support=row.support
        conf=row.confidence
        lift=row.lift
        con=ant.union(con)
        con=set(sorted(con))
        if con in vis:
            continue
        vis.append(con)
        rules.append((con, (support, conf, lift)))
    return rules




def find_maximal_set(dataset, min_support=0.5):
    start_time = time.time()

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    frequent = fpmax(df, min_support=min_support, use_colnames=True)

    # print('Time to find Max frequent itemset')
    # print("--- %s seconds ---" % (time.time() - start_time))

    return frequent


def max_support_max_set(dataset, start_sup=0, end_sup=100):
    start_time = time.time()
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    while start_sup <= end_sup:
        mid_sup = int((start_sup + end_sup) / 2)
        frequent = fpmax(df, min_support=(mid_sup / 100.0), use_colnames=True)

        if len(frequent) > 0:
            start_sup = mid_sup + 1
        else:
            end_sup = mid_sup - 1

    frequent = fpmax(df, min_support=(end_sup / 100.0), use_colnames=True)
    # print('Time to find Max Support Max frequent itemset')
    # print("--- %s seconds ---" % (time.time() - start_time))

    return frequent


def toTupleList(rules, sorted_order=True):
    d = []
    for fsi in range(len(rules)):
        fs = rules.iloc[[fsi]]

        r = set(fs.itemsets.values[0])
        d.append((r, fs.support.values[0]))

    d.sort(key=lambda a: a[1], reverse=sorted_order)
    return d


def rulesToConditions(rule):
    cond = []
    for r in rule:
        r = r.replace('+', '')
        if r.startswith('-1'):
            r = r[1:]
        sp = ' '
        if '>=' in r:
            sp = '>='
        elif '<=' in r:
            sp = '<='
        elif '<' in r:
            sp = '<'
        elif '>' in r:
            sp = '>'
        elif '=' in r:
            sp = '='
        elif '!' in r:
            sp = '!'
        r = r.split(sp)
        cond.append((r[0], sp + r[1]))
    return cond


def measure_rule_strength(df, rule, target_variable_name, target_variable_value,model=None):
    cond1 = rulesToConditions(rule)
    cond1.append((target_variable_name,'=' + str(target_variable_value)))
    cond2 = rulesToConditions(rule)
    cond2.append((target_variable_name,'!' + str(target_variable_value)))
    numPos = len(filter_dataframe_by_multiple_condition(df, cond1))
    numNeg = len(filter_dataframe_by_multiple_condition(df, cond2))

    if (numPos+numNeg)==0.0:
        return 0.0,0.0
    dataStrength = round((numPos / (numPos + numNeg)) * 100.0, 2)

    df=filter_dataframe_by_multiple_condition(df, cond1[0:-1])
    xor = df.drop(['Survived', 'PassengerId'], axis=1).values
    p = model.predict(xor, verbose=0)
    p = [[1 * (x[0] >= 0.5)] for x in p]

    p=np.asarray(p)
    predictionStrength=round((np.sum(p==target_variable_value)/(numPos+numNeg))*100.0,2)

    return dataStrength, predictionStrength

# dataset = [[1, 2], [3, 4, 5, 6, 1], [1, 7], [9, 5, 2], [6, 2, 1, 4, 8], [9, 6], [3, 2, 8, 1], [9, 7, 5, 1, 2, 3],
#            [4, 7, 8, 3, 2], [5, 6, 1], [9, 7, 8, 3], [1, 5, 6], [2, 4, 6], [1, 2, 5]]

# print(find_maximal_set([['gh','ghs'],['<=gh','tf']]))
# print(find_maximal_set(dataset))
