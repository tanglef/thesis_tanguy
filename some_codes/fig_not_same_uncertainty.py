import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
titles = ["consensus", "2 class ambiguity", "uniform ambiguity"]
sns.set_style("whitegrid")
fig, axs = plt.subplots(1, 3, sharey="row", figsize=(15, 8))
for i in range(3):
    if i == 0:
        distrib = [0, 1, 0, 0]
    elif i== 1:
        distrib = [0, .5, .5, 0]
    else:
        distrib = [.25, .25, .25, .25]
    sns.barplot(
        data=pd.DataFrame(
            {"Label": ["class1", "class2", "class3", "class4"] , "voting distribution": distrib},
        ),
        x="Label",
        y="voting distribution",
        ax=axs[i],
    )
    axs[i].set_title(titles[i])
    if i > 0:
        axs[i].set_ylabel("")
    for ax in axs.flatten():
      ax.xaxis.label.set_size(15)
      ax.yaxis.label.set_size(15)
      ax.xaxis.set_tick_params(labelsize=13)
      ax.yaxis.set_tick_params(labelsize=13)
plt.show()
