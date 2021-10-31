

def get_label(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    labels = []
    for line in lines:
        _, _, label = line.split()
        labels.append(label)

    return labels


