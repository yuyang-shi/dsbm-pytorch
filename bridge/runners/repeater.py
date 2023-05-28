from itertools import repeat

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data