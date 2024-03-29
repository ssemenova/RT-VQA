import sys
sys.path.append(sys.path[0] + '/TMLGA')
from TMLGA.utils.vocab import Vocab
from TMLGA.vqa_input import setup_TMLGA_and_get_model
import pickle
import os
import torch
import numpy as np

class Chunk_Localization():
  def __init__(
    self,
    config_file_path,
    vocab_file_path,
    embeddings_file_path,
    max_question_length,
    min_question_length,
    chunk_size
  ):
    self.model = setup_TMLGA_and_get_model()

    self.max_question_length = max_question_length
    self.min_question_length = min_question_length
    self.chunk_size = chunk_size

    self._create_vocab(vocab_file_path)
    self.embedding_matrix = self._get_embedding_matrix(
      embeddings_file_path, max_question_length, min_question_length
    )

  def _create_vocab(self, vocab_file_path):
      with open(vocab_file_path, 'rb') as f:
        self.vocab = pickle.load(f)

  def _get_embedding_matrix(
        self, embeddings_file_path,
        max_question_length, min_question_length
    ):
    '''
    Gets you a torch tensor with the embeddings
    in the indices given by self.vocab.
    Unknown (unseen) words are each mapped to a random,
    different vector.
    '''
    print('TMLGA: loading embeddings into memory...')

    if not os.path.exists(embeddings_file_path):
      # if 'glove' in embeddings_path.lower():
      #     tmp_file = get_tmpfile("test_word2vec.txt")
      #     _ = glove2word2vec(embeddings_path, tmp_file)
      #     embeddings = KeyedVectors.load_word2vec_format(tmp_file)
      # else:
      embeddings = KeyedVectors.load_word2vec_format(file_path, binary=True)

      vocab_size = len(self.vocab)

      matrix = torch.randn(
          (vocab_size , embeddings.vector_size),
          dtype=torch.float32)

      for token, idx in self.vocab.token2index.items():
          if token in embeddings:
              matrix[idx] = torch.from_numpy(
                  embeddings[token])

      with open(embeddings_file_path, 'wb') as f:
        torch.save(matrix, f)

    else:
      with open(embeddings_file_path, 'rb') as f:
        matrix = torch.load(f)

    return matrix

  def _convert_question(self, question):
    raw_tokens = question.translate(
      str.maketrans('', '', string.punctuation)
    ).split(" ")[:max_length - 1]
    raw_tokens.append(".")

    indices = self.vocab.tokens2indices(raw_tokens)
    tokens = [self.embedding_matrix[index] for index in indices]
    tokens = torch.stack(tokens)

    return tokens

  def _pad_sequence(self, sequence):
    lengths = []
    for s in sequence:
        lengths.append(s.shape[0])
    lengths = np.array(lengths, dtype=np.float32)
    lengths = torch.from_numpy(lengths)

    return nn.utils.rnn.pad_sequence(sequence, batch_first=True), lengths

  def predict(self, cache, question):
    model = torch.load(load_path,map_location=torch.device('cpu'))
    model.eval()

    self.localize_array = torch.from_numpy(
      np.ones(feat_length, dtype=np.float32)
    )

    for chunk_id in range(cache.oldest_id, cache.newest_id):
      chunk = cache[chunk_id]
      
      features, feat_length = self._pad_sequence(chunk.i3d_features)

      tokens = self._convert_question(question)
      tokens, tokens_lengths = self._pad_sequence(tokens)

      start = 0
      end = chunk_size

      loss, individual_loss, pred_start, pred_end, attention, atten_loss = model(
        features, feat_length, tokens, tokens_lengths, start, end, self.localize_array
      )
      # TODO: early stop?

    #TODO: return most likely chunk
