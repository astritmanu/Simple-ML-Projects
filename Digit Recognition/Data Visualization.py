import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits=load_digits()
print("\nData type: ", type(digits))
print("\nDataset attributes: ", dir(digits))

print("\n\nColumns in raw data: ", digits.feature_names)
print("\nAll target classes (digits): ", digits.target_names)
print("\nFirst 15 correct digit labels in sample: \n", digits.target[:15])
print("\nFirst 5 raw data samples: \n", digits.images[:5])
print("\nFlattened data matrix (First 5 data samples): \n", digits.data[:5])


for i in range(15):
    plt.matshow(digits.images[i])
    
plt.show()

