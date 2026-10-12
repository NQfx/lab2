import seaborn as sns
import correlation_tasks as cor
import pandas as pd
import matplotlib.pyplot as plt
from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

cor_data = cor.analyze_brain_correlations(BrainDataInput(csv_path="brainsize.txt"))

df = pd.DataFrame([cor_data.men_mri_correlation, cor_data.women_mri_correlation], index=['Men', 'Women'])

fig, (ax1, ax2) = plt.subplots(1, 2)

sns.heatmap(df, annot=True, cmap='RdBu_r', center=0.5, ax=ax1)
ax1.set_title('Heatmap')

sns.scatterplot(data=df.T, ax=ax2)
ax2.set_title('Scatter')

plt.tight_layout()
plt.show()
sns.scatterplot(df.T)
