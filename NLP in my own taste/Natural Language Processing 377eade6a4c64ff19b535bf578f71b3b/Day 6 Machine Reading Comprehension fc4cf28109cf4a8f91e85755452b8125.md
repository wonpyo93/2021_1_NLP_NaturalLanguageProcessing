# Day 6. Machine Reading Comprehension

Let's go into some deeper topic of NLP, Machine Reading Comprehension. It's also known as Question and Answering. Basically we're trying to teach the machine how to read.

Many years ago, people tried to use NLP modules to teach the linguistics of language to make the machine learn. Linguistics include learning the grammars of each sentences, like the figure shown below.

![Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled.png](Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled.png)

Wendy Lehnert mentioned in his work 'The Process of Question Answering' in 1977, 'Since questions can be devised to query any aspect of text-comprehension, the ability to answer questions is the strongest possible demonstration of understanding.' We can get the gist of how important Machine Reading Comprehension is as a task of NLP, even decades ago. 

Reading Comprehension's overall process is to put 'passage' and 'question' as an input, and some kind of 'reading comprehension system' processes these inputs, then pop the 'answer' as an output. Note that the passage and question is different. This means that even though a question may have some hints to the answer, this is not enough to process the accurate answer without some background knowledge to the question.

However, recently, this kind of approach started to get replaced by new models to learn languages. Back then, the machine was not able to think of any other approach because computing systems didn't have the capacity to learn 'big data'. Before 2015, the datasets only included couple thousand questions, such as MCTest (2600 Qs) and ProcessBank (500 Qs). Reading Comprehension systems were hand-built, and as mentioned before, most of them handled linguistic features only. Nowadays, the language is learned as a whole, learning a big chunk of data. By a big chunk of data, we're talking about 100,000 examples. Even if you're inventing something on your own, the lowest bound should now be 10,000 examples these days.

With the help of neural networks, the importance of linguistic features has reduced. Yet, the accuracy has increased rapidly! To brag more, we have surpassed the point where machines can outperform humans.

![Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%201.png](Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%201.png)

Hoping that this has given some interests to you in learning MRC, let's talk about some important datasets and models nowadays.

Dataset: CNN/Daily Mail Dataset

We know what CNN is. It's the document from CNN news (documents). It has documents (passage), and a question that has a missing word. The task for this is to figure out what the missing word is.

For example, let's think about one document, maybe about Star Wars. This document should have a passage, and a highlight of this passage. However, this document has a missing word in this highlight. If the highlight was something like 'Official 'Star Wars', universe gets its first gay character, a lesbian governor.' The missing word would be 'Star Wars'. Then, the task is to understand the passage and somehow figure out what the missing word is.

It's kind of interesting how people came up with this kind of dataset and this kind of task. Back then, people tried really hard to even create the datasets to test upon. But nowadays? Well, you just 'mask' a random (yet important!) word. Since we already have billions of news online, you can guess how easy it became to create huge chunk of datasets compared to old days. What's more, this dataset is free!

![Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%202.png](Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%202.png)

This figure shows an example of the passage, question and answer.

So, how do machines perform? Simply put, encode and model interaction! We learned encoding and modeling, and one thing we can tell is that languages can be computed numerically, all in vector spaces!

Let's look at one of the MRC models, Attentive Reader.

Each word is transformed to vectors. Bidirectional LSTMs, or other models like RNNs, are used to learn these vectors. Then the hidden representations are concatenated to interpret the question, then pop the output.

![Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%203.png](Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%203.png)

The figure above shows 'Stanford' Attentive Reader. 

Attention is basically the importance of each word, in terms of interaction. As each vector is interacted with each other, the machine gets to understand which vectors are slightly more referenced more to the other vectors. These can be interpreted to scores, and we can do 'softmax' to compute the distributed probability, summing to 1.

Using the attnetion score, o랑 a에 대해서 다시 영상 봐야할듯

Actually, Stanford Attentive Reader is a variation of 'Deepmind' Attentive Reader. By variation, we're talking about exactly the same architecture yet different equations. Deepmind Attentive Reader is actually the model that has more complicated equations.

![Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%204.png](Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%204.png)

More complicated, right? Does the accuracy correlate with how complicated it is? No. Stanford Attentive Reader has shown huge leap in accuracy compared to Deepmind's.

![Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%205.png](Day%206%20Machine%20Reading%20Comprehension%20fc4cf28109cf4a8f91e85755452b8125/Untitled%205.png)

Dataset: Stanford Question Answering Dataset (SQuAD)

Model: Stanford Attentive Reader

Task: Open-domain QA

Model: DrQA