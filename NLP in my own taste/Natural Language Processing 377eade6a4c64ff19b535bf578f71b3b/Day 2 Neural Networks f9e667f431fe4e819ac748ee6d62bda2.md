# Day 2. Neural Networks

Now that you have some sense of what you're trying to learn, let's dive into Neural Networks, another basic of the 'models' in Machine Learning.

The reason why its name is Neural Network is simply because they work like neurons. Neurons receive a piece of information and passes the information to another neuron. In this process the information is modified in a sense that the next neuron can interpret. For example, if someone pinches your arm, the out-most neuron receives the pain and passes to the next neuron. This occurs until your brain cell receives the information of 'pain'. The brain cell is not 'pinched', but receives the information of 'pain'.

We want to create a model just like this. Pinching will be the input, the area of pinched spot will be the dimension, number of passes will be layers, and the ouput will be the 'pain'.

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled.png)

Okay, we get the picture now. Let's get into some more details.

What does the input look like?

**One-hot vector.** (usually!)

Let's say that the input is a sentence. Something like, 'I am human'.

Then, we can divide this into the state of words. 'I, am, a, human'.

Then, we can represent each word as one-hot vector.

Assuming that this universe is only made of 10 words, we can list up all these words and locate where each of these words are.

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%201.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%201.png)

The way to locate the word is by placing 1 on the spot, and 0s on all the other locations.

Then, we can represent the sentence 'I am human' with these three 1 by 10 vectors.

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%202.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%202.png)

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%203.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%203.png)

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%204.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%204.png)

This is One-Hot-Vector.

It's obvious that this representation is effective to indicate what word it is, but there are some obvious drawbacks as well.

Let's say we have created another word, 'awesome', in our universe. Then, we'll have to modify the dimension of One-Hot-Vector for all words to 11.

Now, coming back to our real universe, we have tons of words. Representing a word in One-Hot-Vector would then result in an enormous useless 0s, which would cause ample amount of memory use and computation time.

So? We want to reduce dimensions. There are methods like PCA, LDA, LSAs which have bases on the truncated SVD.

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%205.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%205.png)

Basically, we want to represent the matrix A by splitting it into three matrices, of lower dimensions. For those who are interested in learning more about SVD, visit:

[https://angeloyeo.github.io/2019/08/01/SVD.html](https://angeloyeo.github.io/2019/08/01/SVD.html)

So, we want to compress and truncate the given data for efficiency. We call this **Encoding**.

After we're done with encoding, we would naturally want to decode it so that we can show the output in same dimensions of the input. We call this **Decoding**.

We're ready to learn about Autoencoder now.

**Autoencoder**.

Autoencoder is a neural network to learn the identity functions in an **unsupervised** way, by compressing the original input data for more efficient representation to compute with, and then by decompressing it to reconstruct the original input (as similar as possible!).

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%206.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%206.png)

In this figure, the green part indicates the Encoder network, which translates the original input (usually high in dimension) into a representation low in dimension. Through this Encoder, we end up with z, which can also be called as features, latent representations, or code.

Then, the blue part, which indicates the Decoder network, attempts to recover the original input x using z. If the encoder and decoder are perfect, the input (x) and output (x') should be identical.

z is the input that will be computed with the function of your taste. Usually we use neural network models on z.

Let's take an example. Let's say you've seen a picture in an art museum. You want to remember this picture in your head so that you can copy this picture when you go back home. Then, x will be the original picture. Encoder will be your eyes and brains doing the job to somehow remember this picture. (You're not shoving the picture in your skull, are you?) Then, z will be a piece of memory of the original picture. You go home, then the decoder will be  your brain and your hand, as they will do the job to decode this piece of memory into an art of yours. Finally, x' will be the copied work that you've finished drawing. If your memory and artistic skills are perfect, in other words, if your encoder and decoder are perfect, x and x' should be identical.

We know that the two should be theoretically identical, but we also know that it's not always the case. Then, the difference between the two should be minimized. We call that difference '**loss**', and there are some methods of how to compute this loss.

**Loss**

Quantifying Loss

Empirical Loss

Cross Entropy Loss

Binary Cross Entropy Loss

Mean Squared Error Loss

**Minimizing** this loss is called '**loss optimization**'. Since we're computing this loss over and over again and changing some values so that this loss gets reduced, we'll then be able to express these 'many' losses through a graph.

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%207.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%207.png)

We call each of the 'difference' of losses as gradients, and since we're trying to reduce this loss to the minimum, it looks like we're descending from top to bottom. Thus, we call this '**Gradient Descent**'.

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%208.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%208.png)

![Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%209.png](Day%202%20Neural%20Networks%20f9e667f431fe4e819ac748ee6d62bda2/Untitled%209.png)

Computing this gradients can be done through '**Backpropagation**'. We've computed these losses, in other words, propagated through layers. Now, to go back, we'll have to 'back'-propagate.

These terms are the basics of Neural Networks that you must be familiar with in order to walk through the forest of neural network related papers.