import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [x1**2 for x1 in x]
plt.plot(x,y)
plt.show()


import numpy as np
new_input = np.arange(1,101,1)
print(new_input)
new_input=new_input.reshape(-1,1)
print(new_input)

