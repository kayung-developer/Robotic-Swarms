import numpy as np
import matplotlib.pyplot as plt

# Data for the chart
domains = ['Manufacturing', 'Oncology (Medicine)', 'Smart Materials']
metrics = ['Adaptability', 'Precision', 'Autonomy']
traditional_scores = [1, 2, 0]  # Scored on a 1-10 scale: Low Adaptability, Low Precision, No Autonomy
eptm_scores = [10, 10, 9]      # Scored on a 1-10 scale: High Adaptability, High Precision, High Autonomy

x = np.arange(len(domains))  # the label locations
width = 0.35  # the width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
rects1 = ax.bar(x - width/2, traditional_scores, width, label='Traditional Method', color='skyblue')
rects2 = ax.bar(x + width/2, eptm_scores, width, label='EP-TM Approach', color='royalblue')

# Add some text for labels, title and axes ticks
ax.set_ylabel('Performance Score (0-10)', fontsize=12)
ax.set_title('Figure 8: Comparative Analysis of Traditional vs. EP-TM Approaches', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f'{d}\n(Metric: {m})' for d, m in zip(domains, metrics)], fontsize=11)
ax.legend(fontsize=11)
ax.set_ylim(0, 11)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Function to attach a text label above each bar
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

fig.tight_layout()
plt.show()