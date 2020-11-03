#! /usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv("blade_dump.csv")

print (data.info())


N = 500
iters = [10, 20, 100]#np.arange(0,9)
value = "velZ"
abscissa = "pointId"
print(iters)
for i in iters:
    print(i)
    plt.plot(data[abscissa][i*N:(i+1)*N],data[value][i*N:(i+1)*N],label="iter={f}".format(f=i))
plt.legend()
plt.ylabel(value)
plt.xlabel(abscissa)
plt.show()
