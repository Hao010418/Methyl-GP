def load_fasta_format_data(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    seqs, labels = [], []
    for idx, seq in enumerate(data):
        if idx == 0:
            if seq[1] == 'P':  # the sample is pos
                labels.append(1)
            else:  # the sample is neg
                labels.append(0)
        elif idx % 2 == 0:
            if seq[1] == 'P':
                labels.append(1)
            else:
                labels.append(0)
        else:
            seqs.append(seq.replace('\n', '').replace('\r', ''))
    return seqs, labels


def seq2kmer(seqs, k):
    kmers = []
    for seq in seqs:
        kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
        kmers.append(" ".join(kmer))
    return kmers


def get_kmer_sequence(original_string, kmer=1):
    if kmer == -1:
        return original_string

    sequence = []
    original_string = original_string.replace("\n", "")
    for i in range(len(original_string) - kmer):
        sequence.append(original_string[i:i + kmer])

    sequence.append(original_string[-kmer:])
    return sequence


if __name__ == '__main__':
    import pandas as pd
    data_path = 'dataset/Lin_2017/Independent test'
    species = 'G_subterraneus'
    path_train_data = f'{data_path}/{species}/train.txt'
    path_test_data = f'{data_path}/{species}/test.txt'
    train_data, train_label = load_fasta_format_data(path_train_data)
    test_data, test_label = load_fasta_format_data(path_test_data)
    for k in range(3, 7):
        save_train_path = f'train/{k}mer/4mC/{species}.tsv'
        save_test_path = f'dev/{k}mer/4mC/{species}.tsv'
        train_kmer = seq2kmer(train_data, k)
        test_kmer = seq2kmer(test_data, k)
        train = {'sequence': train_kmer, 'label': train_label}
        test = {'sequence': test_kmer, 'label': test_label}
        train = pd.DataFrame(data=train)
        test = pd.DataFrame(data=test)
        train.to_csv(save_train_path, sep='\t', index=False)
        test.to_csv(save_test_path, sep='\t', index=False)

