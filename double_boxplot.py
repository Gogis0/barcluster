import matplotlib.pyplot as plt
import numpy as np

# original source: https://stackoverflow.com/a/20132614


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def make_double_boxplot(ticks, data, path, color1='#D7191C', color2='#2C7BB6'):
    plt.figure(figsize=(10, 5))

    N = len(data[0])
    bplr = plt.boxplot(data[0], positions=np.array(range(N))*2.0-0.1, sym='', widths=0.1)
    bplb = plt.boxplot(data[1], positions=np.array(range(N))*2.0+0.1, sym='', widths=0.1)

    set_box_color(bplr, color1)
    set_box_color(bplb, color2)

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c=color1, label='Same barcode')
    plt.plot([], c=color2, label='Different barcode')
    plt.legend()

    plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    plt.title('Score distribution in terms of "delta"')
    plt.tight_layout()
    plt.savefig(path)
