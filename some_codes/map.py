# %%
import geopandas as gpd

levels2 = gpd.read_file("/home/tlefort/Téléchargements/level2.geojson")
print(levels2.head())

# %%
import matplotlib.pyplot as plt

plt.figure()
levels2.plot(color="azure", linewidth=1, edgecolor="grey")
plt.axis("off")
plt.tight_layout()
plt.savefig("level2.pdf")
# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
# Example neural network
model = [torch.tensor([2.0])]
df = {"T0": [], "Tmult": [], "epoch": [], "lr": [], r"$(T_0, T_{mult})$": []}
# Example optimizer
optimizer = optim.SGD(model, lr=0.1)
for T_0, T_mult in zip([50, 100, 50], [1, 1, 2]):
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=0.001
    )
    for epoch in range(300):
        df["T0"].append(T_0)
        df["Tmult"].append(T_mult)
        df[r"$(T_0, T_{mult})$"].append(f"({T_0}, {T_mult})")
        df["lr"].append(scheduler.get_last_lr()[0])
        df["epoch"].append(epoch)
        scheduler.step()
df = pd.DataFrame(df)
# %%
ax = sns.lineplot(data=df, x="epoch", y="lr", hue=r"$(T_0, T_{mult})$")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
ax.set_ylabel(r"Learning rate $\eta_t$")
ax.set_xlabel(r"Epoch $t$")
ax.set_yscale("log")
plt.tight_layout()
plt.savefig("../chapters/images_plantnet/cosine_annealing_restart.pdf")
# %%
