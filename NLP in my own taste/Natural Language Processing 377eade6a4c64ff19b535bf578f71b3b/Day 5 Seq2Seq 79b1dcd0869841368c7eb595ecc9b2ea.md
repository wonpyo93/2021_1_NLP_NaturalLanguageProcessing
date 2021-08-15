# Day 5. Seq2Seq

So far, we've learned that once we put some sort of input, we get some sort of hidden, and we get some output. But what are these exactly, in terms of language learning?

Let's consider this exmaple.

![Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled.png](Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled.png)

We learned hwo RNN works in Day 4. Let's expand and actually see what kind of inputs are placed to RNN and what we actually do with the output of RNN.

We first have $x_t$, which each word has a form of one-hot-vector. We can directly feed this $x_t$ as an input to RNN, but we usually have an embedding layer, $e_t$, that converts this one-hot-vector to a representational form of it.

To be more precise, $x_t$, the one-hot-vector, only consists of 0s and a single 1. However, $e_t$ looks more like a distributed vector that has fractional numbers between 0s and 1s, indicating the probability of each word.

$e_t = Ex_t$, where $E$ is the pre-trained word embeddings. Pre-trained word embeddings literally contains the words that are pre-trained. For example, if we have fish and dog already pre-trained, this E would already know that the word fish is correlated to swim more than run, whereas dog is correlated to run more than swim. Just for reference, $Ex_t$ does not literally mean E multiplied by $x_t$, because we already know that $x_t$ is like a look-up vector (one-hot-vector, remember?). Also, this E is trained from 'general' ones, not some specific-typed ones. If we take the gaming contexts and make pre-trained E, it wouldn't really be useful in the real-world contexts, would it?

![Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%201.png](Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%201.png)

This part is the typical RNN.

![Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%202.png](Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%202.png)

After we get the final hidden layer, softmax is done to get the most probable output word, between 0 to 1. The highest probable word is chosen, which will then be, in this example, 'chased'.

We've learned what encoder and decoder does previously. Encoder-decoder model can be used to predict the next word, or even translate one language to another. Let's look at this example below.

In machine translations, 

![Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%203.png](Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%203.png)

Let's say we fed 'the black cat drank milk', the english sentence into RNN and got each corresponding $h$s. This process is 'encoding'. Now, we can 'decode' these $h$s to produce french words. Theoretically, we should be able to get the result only using $h_7$, because it should have all the information from $h_1$ to $h_6$ combined, but as we've mentioned before, due to the vanshing gradient problems and forgetting issues, we take all $h$s inside the function $F$. This smart bridging function F overcomes the difference of English and French. Note that English and French have different vocabs, meaning different dimensions of hidden states.

The way of predicting French words is done by some ways, and the most primal way is using greedy inference.

![Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%204.png](Day%205%20Seq2Seq%2079b1dcd0869841368c7eb595ecc9b2ea/Untitled%204.png)

If we predict the correct word, horray. If the prediction is wrong, sad. Simple as that. Obviously, this is risky because if we fail to judge the right word, all the remaining words will fail as well, and there would be no way to observe/correct this. For exmaple, if the desired word was cat, and the machine translated it to dog, yet it checks it as 'correct', we can kind of see the upcoming disasters as we go through the rest of the words.

Therefore, the better approach is Beam Search. The algorithm behind of this approach is shown in the figure above.

Seq2Seq is one of the encoder-decoder models.

58분부터 다시 보자