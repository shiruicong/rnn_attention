from utils import embedding
import os

data_path = 'data/'  # path of data folder
embeddingHandler = embedding.Embedding()
# create vocabulary
en_train_path = os.path.join(data_path, "en_train.txt")
en_test_path = os.path.join(data_path, "en_test.txt")
en_vocab_path = os.path.join(data_path, "en_vocab.txt")

cn_train_path = os.path.join(data_path, "cn_train.txt")
cn_test_path = os.path.join(data_path, "cn_test.txt")
cn_vocab_path = os.path.join(data_path, "cn_vocab.txt")

en_train_ids_path = os.path.join(data_path, "en_train_ids.txt")
en_test_ids_path = os.path.join(data_path, "en_test_ids.txt")
cn_train_ids_path = os.path.join(data_path, "cn_train_ids.txt")
cn_test_ids_path = os.path.join(data_path, "cn_test_ids.txt")

embeddingHandler.create_ids(en_train_path, en_vocab_path, en_train_ids_path)
embeddingHandler.create_ids(en_test_path, en_vocab_path, en_test_ids_path)
embeddingHandler.create_ids(cn_train_path, cn_vocab_path, cn_train_ids_path)
embeddingHandler.create_ids(cn_test_path, cn_vocab_path, cn_test_ids_path)
''' 
embeddingHandler.create_vocab(en_train_path, en_vocab_path)
print(en_vocab_path)
embeddingHandler.create_vocab(cn_train_path, cn_vocab_path)
print(cn_vocab_path)
'''

# create en_embedding
'''
src_embedding_output_path = os.path.join(data_path, 'en_embedding.txt')  # path to file word embedding
src_vocab_path = os.path.join(data_path, 'en_vocab.txt')  # path to file vocabulary
vocab_src, dic_src = embeddingHandler.load_vocab(src_vocab_path)
en_sentences = embeddingHandler.load_sentences(en_train_path)
model = embeddingHandler.create_embedding(en_sentences, vocab_src)
embeddingHandler.save_embedding(model, src_embedding_output_path)
'''
# create cn_embedding
'''
src_embedding_output_path = os.path.join(data_path, 'cn_embedding.txt')  # path to file word embedding
src_vocab_path = os.path.join(data_path, 'cn_vocab.txt')  # path to file vocabulary
vocab_src, dic_src = embeddingHandler.load_vocab(src_vocab_path)
cn_sentences = embeddingHandler.load_sentences(cn_train_path)
model = embeddingHandler.create_embedding(cn_sentences, vocab_src)
embeddingHandler.save_embedding(model, src_embedding_output_path)
'''


