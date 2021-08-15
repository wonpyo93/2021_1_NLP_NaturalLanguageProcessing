# Day 1. Starting Off...

Whatever your interested field is, learning from scratch is always hard.

Especially, Natural Language Processing, as a topic that arose on board in recent years, it's inevitable to spend enormous time to learn since there isn't any guidebook for you to learn the basics. Fields like Mathematics have chapters and sample problems for you to learn step by step, but there is no sound textbook that can do the same thing for you in the field of NLP.

So, what is NLP?

According to Jason Brownlee from Machine Learning Mastery, Natural Language processing is defined as the automatic manipulation of natural language. Natural languages inlclude things like speech, text, signs, emails, web pages, and so on. NLP in terms of Machine Learning is basically trying to understand these languages through the help of machines, as little human intervention as possible. Whether there is a human work involved or not can be categorized as 'supervised' in NLP tasks.

Reference: [https://machinelearningmastery.com/natural-language-processing/](https://machinelearningmastery.com/natural-language-processing/)

Then, what is machine learning?

The broader term of machine learning would be Artificial Intelligence. AI involves all kinds of machine algorithms, so even hard coding can be categorized as AI. Then, machine learning narrows the term of AI, as the machine starts learning some kind of patterns of input data by itself to compute some output. Making it narrower, if the machine starts using the patterns to learn the representation of the patterns, we call it representation learning. If the machine takes many computation layers to do this task, it finally becomes deep learning.

![Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled.png](Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled.png)

Borrowing the words from Wikipedia, Machine Learning is a computer program that learns from experience with respect to some class of tasks and performance measures, and using these therms, it improves(updates) its experience and performance.

Now that we know what Machine Learning, generally, is, what should we know before we dive into some fancy stuffs?

1. Datasets

    It's all about datasets.

    ![Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled%201.png](Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled%201.png)

    Traditional programmings were all about the results, but that's not the end of the story here in Machine Learning. We need to create a program that can continuously perform well on various datasets. 

    Datasets can usually be categorized as 'train', 'valid', and 'test'. Train datasets are the datasets that already have answers. Through this dataset, your model will be able to learn what this dataset wants to tell us. Your model will learn the best function to interepret this dataset, and of course there will be some 'errors' (which we call it, Loss).

    After some amount of knowledge learned through train datasets, valid datasets come in action to see if your model does the right job. It doesn't affect the 'learning' procedures of your model, but it helps your model to 'know' up to how much of your model should learn.

    ![Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled%202.png](Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled%202.png)

    Looking at this figure, we can tell that the amount of training time doesn't always help the test results. We wan to 'optimize' so that the test error is minimized. This is when the valid datasets are used. If we train too much, we call it 'overfitting'.

    ![Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled%203.png](Day%201%20Starting%20Off%2024ad2ec372d54a8988928cb95f6c1a66/Untitled%203.png)

    Then, as you can already tell, test datasets come in to show the actual accuracy of your model.

    Side-Facts:

    There is no definite term for inputs and outputs, or even parameters and functions, but we usually call the inputs as x and outputs as y.

    This is the basics of basics that you need to be aware of before diving into numerous methods and papers that you'll have to encounter in the process of learning Machine Learning.