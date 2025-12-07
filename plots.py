import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os.path as op

def plot_scores(data_dict):
    for key, data in data_dict.items():
        plt.plot(data, label=key)

    plt.hlines(y=19.5, xmin=0, xmax=2500, linestyles='dashed', colors=['black'], label="Target Performance")
    plt.legend()
    plt.xlabel('Episodes')
    plt.ylabel('Average Score')
    plt.title("Average Score over Training Episodes")
    plt.savefig(op.join('plots', 'figures', 'AverageScores.png'))

def process_log(logfile):
    lines = logfile.readlines()
    data_points = []
    for l in lines:
        if 'Score' not in l:
            continue
        point = l.split('Average Score: ')[-1].split('Loss:')[0].strip()
        data_points.append(float(point))
    return np.array(data_points)

if __name__ == "__main__":
    data_dict = {}
    for file in sorted(glob(op.join('logs', '*.txt'))):
        if 'NoFrame' in file:
            continue
        with open(file, 'r') as f:
            data = process_log(f)
            if len(data) > 0:
                data_dict[op.basename(file).split('.')[0]] = data
                np.savetxt(op.join('plots', 'data', f'Scores-{op.basename(file).split('.')[0]}'), data)
    
    plot_scores(data_dict)