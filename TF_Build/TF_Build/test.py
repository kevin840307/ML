import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
width = 17
height = 17
kernel = 5
dilated_rate = 1
dilated_kernel = (kernel + (kernel - 1) * dilated_rate)
pad = (dilated_kernel - 1) >> 1


matrix = np.zeros((height + (dilated_kernel - 1), width + (dilated_kernel - 1)))
matrix[pad:-pad, pad:-pad] = np.arange(1, height * width + 1).reshape((height, width))
#print(matrix)
wedith = np.arange(1, kernel * kernel + 1).reshape((kernel, kernel))
dilated_wedith = np.zeros((dilated_kernel, dilated_kernel))
dilated_wedith[0::dilated_rate + 1, 0::dilated_rate + 1] = 1#wedith
result = np.zeros((height, width))
count = np.zeros_like(matrix)
for i in range(height):
    for k in range(width):
        result[i, k] = np.sum(matrix[i:i + dilated_kernel, k:k + dilated_kernel] * dilated_wedith)
        count[i:i + dilated_kernel:dilated_rate + 1, k:k + dilated_kernel: dilated_rate + 1] += 1
#print(result)
print(count[pad:-pad, pad:-pad])


plt.subplot(121)
sns.heatmap(count[pad:-pad, pad:-pad].astype(np.int), annot=True, cmap="Blues", xticklabels =False, yticklabels =False, cbar=False, annot_kws={'size': 16})

width = 17
height = 17
kernel = 5
dilated_rate = 0
dilated_kernel = (kernel + (kernel - 1) * dilated_rate)
pad = (dilated_kernel - 1) >> 1


matrix = np.zeros((height + (dilated_kernel - 1), width + (dilated_kernel - 1)))
matrix[pad:-pad, pad:-pad] = np.arange(1, height * width + 1).reshape((height, width))
#print(matrix)
wedith = np.arange(1, kernel * kernel + 1).reshape((kernel, kernel))
dilated_wedith = np.zeros((dilated_kernel, dilated_kernel))
dilated_wedith[0::dilated_rate + 1, 0::dilated_rate + 1] = 1#wedith
result = np.zeros((height, width))
count = np.zeros_like(matrix)
for i in range(height):
    for k in range(width):
        result[i, k] = np.sum(matrix[i:i + dilated_kernel, k:k + dilated_kernel] * dilated_wedith)
        count[i:i + dilated_kernel:dilated_rate + 1, k:k + dilated_kernel: dilated_rate + 1] += 1
#print(result)
print(count[pad:-pad, pad:-pad])

plt.subplot(122)
sns.heatmap(count[pad:-pad, pad:-pad].astype(np.int), annot=True, cmap="Blues", xticklabels =False, yticklabels =False, cbar=False, annot_kws={'size': 16})
plt.show()