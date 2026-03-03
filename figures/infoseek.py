import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Set scientific style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['mathtext.fontset'] = 'stix'

# Data
docs = [1, 3, 5, 10, 20, 50, 100]
model_7b = [30.49, 35.71, 36.95, 38.4, 39.73, 42.13, 39.62]
model_3b = [26.24, 30.22, 31.3, 32.48, 32.9, 34.74, 31.04]
reranker_7b = [30.49, 37.1, 38.51, 40.2, 41.18, 43.95, 44.44]
reranker_3b = [26.24, 30.41, 31.82, 33.95, 34.25, 35.75, 35.79]

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Plot lines
line1 = ax.plot(docs, model_7b, 'o-', color='#1f77b4', linewidth=2, markersize=8, 
                markeredgewidth=1.5, markeredgecolor='white', label='Model 7B')
line2 = ax.plot(docs, model_3b, 's-', color='#ff7f0e', linewidth=2, markersize=8, 
                markeredgewidth=1.5, markeredgecolor='white', label='Model 3B')
line3 = ax.plot(docs, reranker_7b, '^-', color='#2ca02c', linewidth=2, markersize=8, 
                markeredgewidth=1.5, markeredgecolor='white', label='Reranker 7B')
line4 = ax.plot(docs, reranker_3b, 'd-', color='#d62728', linewidth=2, markersize=8, 
                markeredgewidth=1.5, markeredgecolor='white', label='Reranker 3B')

# Set x-scale to log
ax.set_xscale('log')
ax.set_xticks(docs)
ax.set_xticklabels(docs)

# Labels and title
ax.set_xlabel('Number of Retrieved Documents', fontsize=14, fontweight='bold')
ax.set_ylabel('Performance', fontsize=14, fontweight='bold')
ax.set_title('Scaling Law Behavior in Document Retrieval Performance', fontsize=16, fontweight='bold', pad=20)

# Grid
ax.grid(True, which="both", axis='both', alpha=0.3, linestyle='--')

# Legend
legend = ax.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='black', 
                  loc='best', handlelength=2.5)
legend.get_frame().set_linewidth(1.5)

# Set axis limits
ax.set_xlim(0.8, 120)
ax.set_ylim(25, 46)

# Tick parameters
ax.tick_params(axis='both', labelsize=12, direction='in', length=6, width=1.5)
ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)

# Spine linewidth
for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Tight layout
plt.tight_layout()

# Save figure
plt.savefig('performance_vs_documents.pdf', dpi=300, bbox_inches='tight')
plt.savefig('performance_vs_documents.png', dpi=300, bbox_inches='tight')

# Show plot
# plt.show()