from datasets import load_dataset

ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1")

train_data = ds["train"]["text"]
test_data = ds["test"]["text"]
valid_data = ds["validation"]["text"]

# Save the data to txt files
with open("test.txt", "w") as f:
    for l in test_data:
        f.write(l)
with open("valid.txt", "w") as f:
    for l in valid_data:
        f.write(l)

# Keep saving data to train.txt until it hits 55MB (5x bigger than wikitext-2)
with open("train.txt", "w") as f:
    size = 0
    for l in train_data:
        f.write(l)
        size += len(l)
        if size > 55 * 1024 * 1024:
            break
