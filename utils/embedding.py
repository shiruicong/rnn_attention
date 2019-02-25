import numpy as np
from gensim.models import Word2Vec
import collections
import operator


class Embedding:
    def load_vocab(self, vocab_file):
        vocab = []
        dic = {}
        with open(vocab_file, encoding="utf8") as file:
            i = 0
            for line in file:
                vocab.append(line[:-1])  # remove \n symbol and add to list
                dic[vocab[-1]] = i  # assign newly added word to dictionary
                i += 1
        return vocab, dic

    def load_sentences(self, train_file):
        sentences = []
        with open(train_file, encoding='utf8') as file:
            for line in file:
                line = line.split()
                sentences.append(line)
        return sentences

    def create_embedding(self, sentences, vocab, vector_size=200, window=10):
        vocab = list(map(lambda x: [x], vocab))
        # train model
        model = Word2Vec(size=vector_size, window=window, min_count=5, sg=1, hs=0)
        model.build_vocab(vocab)
        model.train(sentences, total_examples=len(sentences), epochs=20)
        return model

    def save_embedding(self, model, output_file):
        model.save(output_file)

    def load_embedding(self, path):
        return Word2Vec.load(path)

    def parse_embedding_to_list_from_vocab(self, word2vec, vocab):
        embeddings = []
        for w in vocab:
            embeddings.append(word2vec[w])
        embeddings = np.asarray(embeddings)
        return embeddings

    def find_vector_word(self, word, embedding):
        words = embedding.wv.vocab
        if word in words:
            return embedding[word]
        else:
            return embedding["<unk>"]

    def convert_sentences_to_ids(self, dic, list_sentences):
        sentences_as_ids = []
        for sentence in list_sentences:
            sentence_ids = []
            for word in sentence:
                try:
                    sentence_ids.append(dic[word])
                except KeyError:
                    sentence_ids.append(1)  # 1 is index of <unk> in vocabulary
            sentences_as_ids.append(sentence_ids)
        return sentences_as_ids

    def ids_to_words(self, list_ids, vocab):
        return [vocab[id] for id in list_ids]

    def words_to_ids(self, list_words, dic):
        result = []
        for word in list_words:
            try:
                result.append(dic[word])
            except KeyError:
                result.append(1)
        return result

    # --------------------
    def create_vocab(self, train_file, output_file):
        counter = collections.Counter()
        with open(train_file, encoding='utf8') as f:
            for line in f:
                for word in line.strip().split():
                    counter[word] += 1
        # 按词频顺序对单词降序排列
        sorted_word_to_cnt = sorted(counter.items(), key=operator.itemgetter(1), reverse=True)
        sorted_words = [x[0] for x in sorted_word_to_cnt]
        # 添加句子结束符号
        sorted_words = ["<eos>"] + sorted_words
        # 添加未知单词符号
        sorted_words = ["<unk>"] + sorted_words
        # 添加句子起始符号
        sorted_words = ["<sos>"] + sorted_words
        # 删除低频词汇  （暂时省略）
        with open(output_file, "w", encoding='utf8') as fp:
            for word in sorted_words:
                fp.write(word + "\n")

    def create_ids(self, train_file, vocab_file, output):
        with open(vocab_file, encoding='utf8') as f_vocab:
            vocab = [w.strip() for w in f_vocab.readlines()]
        word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

        # 如果出现了被删除的的低频单词，则替换为"<unk>"
        def get_id(word):
            return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]
        fin = open(train_file, "r", encoding="utf8")
        fout = open(output, "w", encoding="utf8")
        for line in fin:
            words = line.strip().split() + ["<eos>"]   # 读取单词并添加结束符
            # 将每个单词替换为词汇表中的编号
            out_line = " ".join([str(get_id(w)) for w in words]) + '\n'
            fout.write(out_line)
        fin.close()
        fout.close()










