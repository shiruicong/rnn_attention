import tensorflow as tf
import sys
from nltk.translate.bleu_score import corpus_bleu
import os

CHECKPOINT_PATH = "checkpoint/attention/attention-800"
HIDDEN_SIZE = 1024  # LSTM的隐藏层规模
NUM_LAYERS = 2   # 深层循环神经网络中LSTM结构的层数
SRC_VOCAB_SIZE = 11872  # 源语言词汇表大小
TRG_VOCAB_SIZE = 13289  # 目标语言词汇表大小
BATCH_SIZE = 64  # 测试数据的batch的大小
SHARE_EMB_AND_SOFTMAX= True  # 在softmax层和词向量之间共享参数

# 词汇表文件
SRC_VOCAB = "data/en_vocab.txt"
TRG_VOCAB = "data/cn_vocab.txt"

# 词汇表中<sos>和<eos>的ID。在解码过程中需要用<sos>作为第一步的输入，并将检查
# 是否是<eos>，因此需要知道这两个符号的ID。
SOS_ID = 0
EOS_ID = 2

# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每一句话，单词已经转化为单词编号
def MakeDataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset

# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和batching操作
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset，现在每个Dataset中每一项数据ds由4个张量组成
    # ds[0][0]是源句子
    # ds[0][1]是源句子长度
    # ds[1][0]是目标句子
    # ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))


    # 解码器需要两种格式的目标句子：
    # 1.解码器的输入(trg_input), 形式如同'<sos> X Y Z'
    # 2.解码器的目标输出(trg_label), 形式如同'X Y Z <eos>'
    # 上面从文件中读到的目标句子是'X Y Z <eos>'的形式，我们需要从中生成'<sos> X Y Z'形式并加入到Dataset
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))
    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据
    dataset = dataset.shuffle(10000)

    # 规定填充后的输出的数据维度
    padded_shapes = (
        (tf.TensorShape([None]),    # 源句子是长度未知的向量
         tf.TensorShape([])),       # 源句子长度是单个数字
        (tf.TensorShape([None]),    # 目标句子(解码器输入)是长度未知的向量
         tf.TensorShape([None]),    # 目标句子(解码器目标输出)是长度未知的向量
         tf.TensorShape([]))        # 目标句子长度是单个数字
    )
    # 调用padded_batch方法进行batching操作
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


# 定义NMTModel类来描述模型
# attention 编码器双向循环，解码器单向循环
class NMTModelA(object):
    # 在模型的初始化函数中定义模型要用到的变量
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构
        self.enc_cell_fw = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)

        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])
        # 为源语言和目标语言分别定义词向量
        self.src_embedding = tf.get_variable('src_emb', [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable('trg_emb', [TRG_VOCAB_SIZE, HIDDEN_SIZE])
        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable('weight', [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable('softmax_loss', [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里
        # 将输入句子整理为大小为1的batch
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder"):
            # 使用bidirectional_dynamic_rnn构造编码器。这一步与训练时相同。
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(
                self.enc_cell_fw, self.enc_cell_bw, src_emb, src_size,
                dtype=tf.float32)
            # 将两个LSTM的输出拼接为一个张量。
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        with tf.variable_scope("decoder"):
            # 定义解码器使用的注意力机制
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                HIDDEN_SIZE, enc_outputs,
                memory_sequence_length=src_size)

            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络。
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                self.dec_cell, attention_mechanism,
                attention_layer_size=HIDDEN_SIZE)

        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN = 100
        with tf.variable_scope("decoder/rnn/attention_wrapper"):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0,
                                        dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, SOS_ID)
            # 调用attention_cell.zero_state构建初始的循环状态。循环状态包含
            # 循环神经网络的隐藏状态，保存生成句子的TensorArray，以及记录解码
            # 步数的一个整数step
            init_loop_var = (
                attention_cell.zero_state(batch_size=1, dtype=tf.float32),
                init_array, 0)

            # tf.while_loop的循环条件：
            # 循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(tf.logical_and(
                    tf.not_equal(trg_ids.read(step), EOS_ID),
                    tf.less(step, MAX_DEC_LEN - 1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding,
                                                 trg_input)
                # 调用attention_cell向前计算一步。
                dec_outputs, next_state = attention_cell.call(
                    state=state, inputs=trg_emb)
                # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作为
                # 这一步的而输出。
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(
                continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


def main():
    # 定义训练用的循环神经网络模型
    with tf.variable_scope("nmt_model", reuse=None):
        model = NMTModelA()

    # 定义个测试句子
    test_en_text = "the present food surplus can specifically serve the purpose of helping western china restore its woodlands , grasslands , and the beauty of its landscapes . <eos>"
    print(test_en_text)

    # 根据英文词汇表，将测试句子转为单词ID
    with open(SRC_VOCAB, "r", encoding="utf8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    test_en_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>'])
                   for token in test_en_text.split()]

    print(test_en_ids)

    # 建立解码所需的计算图
    output_op = model.inference(test_en_ids)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, CHECKPOINT_PATH)
    print("loaded model")

    # 读取翻译结果
    output_ids = sess.run(output_op)
    print(output_ids)

    # 根据中文词汇表，将翻译结果转换为中文文字
    with open(TRG_VOCAB, "r", encoding="utf8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    output_text = ''.join([trg_vocab[x] for x in output_ids])

    # 输出翻译结果
    print(output_text.encode('utf8').decode(sys.stdout.encoding))
    sess.close()


if __name__ == "__main__":
    main()
