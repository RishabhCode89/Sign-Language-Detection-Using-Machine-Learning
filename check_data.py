import pickle

try:
    with open('data.pickle', 'rb') as f:
        dataset = pickle.load(f)
except FileNotFoundError:
    print("Error: data.pickle file not found.")
    exit(1)
except Exception as e:
    print(f"Error loading data.pickle: {e}")
    exit(1)

print(f"Number of samples: {len(dataset['data'])}")
print(f"Number of labels: {len(dataset['labels'])}")
print(f"First sample data: {dataset['data'][0]}")
print(f"First sample label: {dataset['labels'][0]}")
