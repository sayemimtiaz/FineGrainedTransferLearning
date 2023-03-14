import csv
import os
import seaborn as sns
from matplotlib.colors import ListedColormap

from util.common import get_project_root
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ALPHA = '0.05'
ARCH = 'inceptionv3'
inpuFile = os.path.join(get_project_root(), "final_results", "task_similarity.csv")
outputDir = os.path.join(get_project_root(), "final_results", "processed")


def get_lower_tri_heatmap(df, outputName):
    mask = np.zeros_like(df, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Want diagonal elements as well
    mask[np.diag_indices_from(mask)] = False

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns_plot = sns.heatmap(df, mask=mask, cmap="BuPu", annot=True, square=True, linewidths=1.5,
                           fmt='g')
    # save to file
    # fig = sns_plot.get_figure()
    # fig.savefig(os.path.join(outputDir, outputName))

    plt.show()


data = {}

all_tasks = set()
with open(inpuFile, 'r') as input:
    reader = csv.reader(input)
    next(reader)
    for line in reader:
        arch = line[0].strip()
        alpha = line[1].strip()
        task1 = line[2].strip()
        task2 = line[3].strip()
        score = float(line[4].strip())

        all_tasks.add(task1)
        all_tasks.add(task2)

        if arch not in data:
            data[arch] = {}
        if alpha not in data[arch]:
            data[arch][alpha] = {}
        if task1 not in data[arch][alpha]:
            data[arch][alpha][task1] = {}

        data[arch][alpha][task1][task2] = score

all_tasks = sorted(list(all_tasks))

sndata = []
for task1 in all_tasks:
    t = []
    avg = []
    for task2 in all_tasks:
        t.append(data[ARCH][ALPHA][task1][task2])

        if task1 != task2:
            avg.append(data[ARCH][ALPHA][task1][task2])

    sndata.append(t)
    print(task1+' average score: '+str(np.asarray(avg).mean()))


df = pd.DataFrame(sndata, columns=all_tasks, index=all_tasks)

get_lower_tri_heatmap(df, ARCH+"_"+str(ALPHA)+".pdf")
