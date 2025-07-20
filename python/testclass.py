import pandas as pd

df = pd.read_csv(r'D:\projects ai\tamil-handwritten-recognition\data\test.csv')
class_counts = df['Class Label'].value_counts()
print("Class distribution in test.csv:")
print(class_counts)