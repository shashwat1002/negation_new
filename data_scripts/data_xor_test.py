import datasets

data = datasets.load_dataset("data_xor.py", add_sep=False)["train"]


print(data[0])
print(data[1])

print(data[2])
print(data[3])

print(data[4])
print(data[5])


data2 = datasets.load_dataset("data_detect_neg.py")["train"]

print(data2[0])
print(data2[1])

print(data2[2])
print(data2[3])

print(data2[4])
print(data2[5])
