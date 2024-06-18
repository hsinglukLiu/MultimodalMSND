# libraries & dataset
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
plt.figure(figsize=(25, 15))
# sns.set_style({'font.size': 20.0})


# style="darkgrid"
# 加载示例数据集
df = pd.read_excel("box_data.xlsx", sheet_name='Sheet1')
df.head()
sns.set_theme(context='poster', style="ticks", font='Arial', font_scale=2.0)
plt.legend(loc='center left')
# sns.boxplot(x="Scale", y="Fleet size", hue="Mode", palette="Pastel1", data=df, width=0.8)
sns.lineplot(x="Scale", y="Fleet size", hue="Mode", data=df)

plt.savefig('./figure.6.3.1.pdf', format='pdf')

plt.show()


# MaaS_df = df.iloc[np.where(df['mode'] == 'MaaS')]
#
# sns.boxplot(x="Scale", y="Fleet size", data=MaaS_df, palette="Pastel1", width=0.5)
#
# x = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000]
# MaaS_y = [119.50, 134.29, 150.75, 168.00, 176.00, 185.67, 201.00, 196.00]

# plt.show()
#
# Ridesharing_df = df.iloc[np.where(df['mode'] == 'Ride-sharing')]
#
# sns.boxplot(x="Scale", y="Fleet size", data=Ridesharing_df, palette="Set1", width=0.5)
# plt.show()