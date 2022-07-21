import pickle

if __name__ == '__main__':
    empty = []
    with open('val.pkl', 'rb') as f:
        empty = pickle.load(f)
    
    good = []
    counter = [0 for i in range(17+1)]
    byungshin = 0
    for label, data in empty:
        if len(data) == 16:
            good.append((label, data))
            counter[label] += 1
        else:
            byungshin += 1
    
    print(f'total : {len(empty)}')
    print(f'bad : {byungshin}')
    zero = 0
    for c in counter:
        zero += c
    print(f'good : {zero}')
    
    print(counter)

    with open('val_clean.pkl', 'wb') as f:
        pickle.dump(good, f, pickle.HIGHEST_PROTOCOL)