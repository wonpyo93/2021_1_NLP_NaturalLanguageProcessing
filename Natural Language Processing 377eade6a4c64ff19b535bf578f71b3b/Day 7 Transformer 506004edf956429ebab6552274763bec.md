# Day 7. Transformer

So far, we've learned some key methods to learn sequence modeling and some other tasks for NLP. The flow looks somewhat like this.

RNN >> LSTM >> LSTM + Attention.

To solve some longer tasks without hindering from vanishing gradient problems, attention has been implemented to LSTM. The community of the NLP field was like a one peaceful town where researchers published almost one paper a day that covered similar topics to outperform others. (This was meant to be sarcastic, fyi). Then the meteor-like paper has emerged: Attention Is All You Need.

This paper basically states that sequential modeling is no longer necessary for NLP. We all know that language is not a sequence although we believe we use sentences after sentences which seems like a sequence. RNN and LSTM were based on the idea that language is sequential, and this belief has created its own pit, called vanishing gradient problem. **Transformer**, the model that was introduced in the paper, Attention Is All You Need, broke this stereotype. Note that this paper is not the *first* paper that has suggested non-sequential model to solve NLP tasks (for example, Convolutional Neural Networks!), but transformer is significant as it has indeed surpassed all the other models. So, let's get into what this Transformer is.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled.png)

The first thing we can tell from this picture is that Transformer follows the similar structure of Encoder-Decoder. However, it outperforms other models, mainly because of its self-attention mechanism in encoders and decoders.

Also, note that it's called encoderS and decoderS. If we go deeper into them, we can see that encoder and decoder structures actually consists of 6 of them each, like the figure shown below. The number 6 can be varied depending on different tasks.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%201.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%201.png)

Let's take a deeper dive into one encoder.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%202.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%202.png)

Each encoder consists onf self-attention layer and Feedforward Neural Network. To be more specific, one hot representation of the word goes through self-attention layer which becomes the weighted sum of multiple words of attention scores, then this sum representation goes through the feed forward neural network.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%203.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%203.png)

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%204.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%204.png)

Decoder looks similar to encoder structure, with Encoder-Decoder Attention layer attached in the middle. While Self-Attention layer attempts to find relationships between target words, Encoder-Decoder Attention layer focuses on the relationships between the source words and target words. Note that the dimension of one word (the green part $x_1$) is 512.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%205.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%205.png)

Let's look at the figure above, where the input sentence is 'The animal didn't cross the street because it was too tired'. The transformer shows attention based representation, so the word 'it_', which represents 'animal_', will indicate the word 'animal_' as the most attention-weighted word. This co-reference problem can be solved using self-attention.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%206.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%206.png)

Keep in mind that Queries, Keys and Values have different roles for a single input. For the word 'Thinking', for example, if we type the word 'thinking' into search engine, we use the Queries value. And if we searched up a random word and found 'Thinking' as one of the search results, we use the Keys value. These added up values become the final Values value. We separate these values for different roles because of self-attention, as the different representations for different purposes can lead to better accuracy.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%207.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%207.png)

This is the overall process green to red, from one encoder figure shown high above.

Now, let's go back to the sequential parts. Previously, I've mentioned that Transformers have broken the stereotype of language breing sequential. However, it's true that we do write sentences one after another in order. How does Transformer tackle this issue? **Positional Encoding Vectors**. The one hot representation $x_1$ isn't a simple one hot representation but it includes the positional information as well, for Transformers.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%208.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%208.png)

So far, we've covered some convenient and simple technology of Transformers. Let's take a deeper dive into how Transformer works.

The Residuals Connections.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%209.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%209.png)

We've covered Self=Attention and Feedforward Neural Network layers, but it's not really smooth in connecting one to another in value-wise. Actually, these layers have 'Add & Normalize' sublayer which does residual connection computation (including dotted arrow lines).

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%2010.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%2010.png)

To be more specific, $z_1$ which comes from $x_1$ actually goes through LayerNorm, which uses $x_1$again for normalizing its values for the purpose of reducing the size of values.

Now that we've learned specific parts of architectures, so by looking at this overall structure below, we can understand each part.

![Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%2011.png](Day%207%20Transformer%20506004edf956429ebab6552274763bec/Untitled%2011.png)