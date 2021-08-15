# Day 3. Word Embedding

We've seen auto-encoders previously. We're aware that this auto-encoder has shallow layers, but it still works with a single image. However, in language models, our goal is to learn smaller dimensional representation from the original one hot representation, of **words**!

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%206.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%206.png)

And this z has to also overcome the orthogonality to capture some informative relationships between words as well. Achieving so is impossible if we use the low layered auto encoder, because the sparse vectors of one-hot-representation (zillions of 0s and a single 1) won't be so efficient in capturing any meaning between words.

So, let's learn about word embedding.

we add distributional hypothesis to auto-encoders.

For example, if we want to know what word 'Trump' means, at first, we can find millions of sentences that contain the word, 'Trump'. Then, maybe we'll be able to observe some similar 'surrounding' words or contexts, and maybe we'll be able to see that these contexts describe the meaning of the word 'Trump'.

Let's look at one implementation of this, quite simple one, called '**Continuous Bag of Words**' (**CBOW**). CBOW is actually one of the model that has been introduced by Word2Vec. Word2Vec has two models, one is CBOW and another is called Skipgram, which will be covered after CBOW.

![Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled.png](Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled.png)

Given a sentence shown above, let's say we want to predict the word 'Trump' using the surrounding words. We'll call the surrounding words as 'neighboring words/contexts'.

![Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled%201.png](Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled%201.png)

Given a set of neighboring words, we can predict a single word that potentially occur along with the context. In this example, we're setting the input window size as 2. This means that we will be looking at two neighboring words from the target words to the left, and also to the right. Using these as inputs, we'll aggregate them to find some hidden representation to represent 'Trump'.

Skipgram is a more popular model than CBOW.

Compared to CBOW, the input and output has been switched. We set the input as 'Trump', and we somehow want to make the output of possible surrounding words that describe the input word.

![Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled%202.png](Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled%202.png)

![Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled%203.png](Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/Untitled%203.png)

CBOW may be faster than CBOW, but we generally upvote Skipgram on its performance. Note that CBOW actually performs better in frequent words. However, in real world, we generally want to find the meaning of a word that does **not** come up frequently, so skipgram tends to be the better model.

For more information about CBOW and Skipgram, please refer to this presentation.

[발표1_EfficientEstimationOfWordPresentationsInVectorSpace.pdf](Day%203%20Word%20Embedding%202ca397d31b624b96a1f82d9dd3b34771/1_EfficientEstimationOfWordPresentationsInVectorSpace.pdf)