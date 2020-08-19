from code import *


# negative skipgram
# skip_gram = SkipGram(embeddings_size=100, method='negative')
# skip_gram.train_negative_sampling(lr=0.01, num_epochs=10)

# negative cbow
# cbow = CBOW(embeddings_size=100, method='negative')
# cbow.train_negative_sampling(lr=0.01, num_epochs=10)

# h-softmax skipgram
# skip_gram = SkipGram(embeddings_size=100, method='h_softmax')
# skip_gram.train_h_softmax(lr=0.01, num_epochs=10)

# h-softmax CBOW
cbow = CBOW(embeddings_size=100, method='h_softmax')
cbow.train_h_softmax(lr=0.01, num_epochs=10)