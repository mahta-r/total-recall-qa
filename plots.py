import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def plot_instance_count_histogram(seed_class, subclasses, bin_width=100, log=True):
    counts = [sub['instance_count'] for sub in subclasses if 'instance_count' in sub]
    min_count = min(counts)
    max_count = max(counts)
    
    # Create bin edges from min to max with the given bin width
    bins = np.arange(min_count, max_count + bin_width, bin_width)

    plt.figure(figsize=(8, 5))
    plt.hist(counts, bins=bins, log=log, edgecolor='black')
    plt.xlabel('Number of instances')
    plt.ylabel('Number of subclasses')
    plt.title(f'Distribution of instance counts for subclasses of {seed_class}')
    plt.tight_layout()
    plt.show()



def num_classes_bar_chart(stats, cutoffs):
    
    def lighten_color(color, amount):
        try:
            c = mcolors.cnames[color]
        except:
            c = color
        c = mcolors.to_rgb(c)
        return tuple(1 - (1 - channel) * (1 - amount) for channel in c)

    
    bars = stats.keys()
    bar_positions = range(len(bars))

    segments = {}
    for seed, counts in stats.items():
        segments[seed] = []
        prev = 0
        for cutoff in cutoffs:
            segments[seed].append((prev, counts[cutoff]))
            prev = counts[cutoff]

    bar_width = 0.6

    fig, ax = plt.subplots()

    # Draw segmented bars
    for i, bar in enumerate(bars):
        n_segments = len(segments[bar])
        for j, (start, end) in enumerate(segments[bar]):
            shade = lighten_color('blue', j / (n_segments * 1.2))
            ax.bar(
                x = i,
                height = end - start,
                bottom = start,
                width = bar_width,
                color = shade,
                edgecolor = shade
            )

            border_value = end
            if j == len(segments[bar]) - 1 or j == 0:
                ax.text(
                    i, border_value, str(border_value),  # (x, y, label)
                    ha='center', va='bottom', fontsize=9, color='black'
                )

    # Aesthetic options
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(bars, rotation=90)
    ax.set_ylabel("Count")
    ax.set_title("Number of classes after each filtering step")

    plt.show()
