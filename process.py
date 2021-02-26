import random
import jsonlines as jl


def process_dataset(input_file, train_file, test_file):
    file = open(input_file, 'r', encoding='utf-8')
    data = []
    for line in file:
        sentence = line.split('</d>')[0].strip().split()
        topic = line.split('</d>')[1].strip().split()
        data.append({
            'sentence': sentence,
            'topic': topic
        })

    random.shuffle(data)
    train_data = data[:-5000]
    test_data = data[-5000:]

    with jl.open(train_file, 'w') as f:
        for line in train_data:
            f.write(line)

    with jl.open(test_file, 'w') as f:
        for line in test_data:
            f.write(line)


def process_word2vec(input_file, output_file, embedding_dim=300):
    file = open(input_file, 'r', encoding='utf-8')
    writer = jl.open(output_file, 'w')
    for index, line in enumerate(file):
        if index != 0:
            line = line.strip().split()
            token = " ".join(line[:-embedding_dim])
            vec = [float(one) for one in line[-embedding_dim:]]
            writer.write({
                'token': token,
                'vec': vec
            })


if __name__ == '__main__':
    process_dataset('./raw_data/zhihu.txt', './data/zhihu/train.jl', './data/zhihu/test.jl')
    process_dataset('./raw_data/composition.txt', './data/essay/train.jl', './data/essay/test.jl')
    process_word2vec('./raw_data/sgns.zhihu.word', './data/embedding/word2vec.300dim.jl')
