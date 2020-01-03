---
layout: post
title: Reducing bias in toxic comment classification
excerpt: "An overview of my results and findings from my project to build toxic comment classification model which is less biased against minority communities."
categories: [projects, NLP, Bias, LSTM ]
mathjax: true
comments: true
---

As machine learning is increasingly used in our day-to-day lives, issues surrounding the reinforcement of existing biases against minority identities are increasingly important. In this project, I have explored this from the scope of online toxic comment classification. I have looked into the shortcomings of traditional machine learning models and natural language processes and have sought to train a model which is better able to identify toxic comments while minimising bias against minority identities.  

This post provides an overview of my process and findings. You can find the code for this project on my [GitHub](https://github.com/GovindSuresh/reducing-bias-in-toxicity-classification).

## How do traditional ML and NLP models reinforce bias?

Ultimately, ML models learn from the data they are trained on, therefore the models will pick up on biases that already exist in the data due to societal norms. When looking at online comments, we unfortunately see that a large number of toxic comments are directed at a variety of minority groups (e.g. black, Muslim, LGBTQ) and therefore may repeatedly contain mentions of these groups. 

Therefore traditional ML classification models, when combined with an encoding process such as TF-IDF, have a tendency to learn that the occurance of words such as 'gay' or 'black' suggests that a comment is more likely to be toxic, irrespective of wider context. As a result, we see large numbers of false positives when looking specifically at comments mentioning these identities.

This is further exacerbated by standard NLP steps. It is common to reduce the complexity of the text data being passed in by going through certain steps such as lemmatization, and the removal of stop-words. Usually, this isn't a major issue given that we keep the majority of the useful information in the text. However, when we consider complex topic matters such as toxic comments, contextual clues given by any part of a comment can be useful. In addition, these processes reduce sentences down to key words, which in toxic comment cases may just be repeated mentions of a minority identity.

To help highlight this we have taken an example from the test dataset used for this project. This comment has been labelled as 'Non-Toxic', but my logistic regression model has misclassified it as toxic. You can see from the image that the model has highlighted the words 'gay', 'lesbian', 'transgender' as key words indicating toxicity (shown in green below). 

![missclass](/assets/images/missclass-lgbt2.png "Misclassified LGBT comment")

 **Original Text:** *'I think your hearts in the right place but gay men and lesbians have no issues using the correctly gendered bathrooms. Sexual orientation and gender identity are two totally different things. A transgender person can be gay, lesbian, bi, straight or any other Sexual orientation.'*

 So why is this a problem? Firstly, we would be effectively excluding members of minority groups from being able to discuss themselves online and further marginalizing them. Secondly, the models would be suppressing any talk regarding these groups, therefore potentially suppressing useful discussion about societal issues. There are numerous other impacts created by biased models at a societal and individual level and this remains a key area of study in ML.
 
## Training a better model

In this project, I have looked into training a more complex neural network model to help alleviate the bias issue. The architecture I have chosen is the LSTM (Long Short Term Memory) model, which is a variant on RNNs. RNN's are optimally suited to tackling this problem due to their ability to parse through sequences such as a comment. In other words, the RNN can assess each word of a sequence based on what it knows from the previous words. The model can therefore build contextual understanding sentences based on what it knows about the words used in the sequence. This should give it an advantage over a traditional ML model such as Logistic Regression which have a tendency to look at how often individual words appear in toxic comments.  

### LSTM

As sequences get longer, RNNs in particular suffer from the [vanishing gradient problem](https://www.superdatascience.com/blogs/recurrent-neural-networks-rnn-the-vanishing-gradient-problem). If the sequence is too long, gradients further back in the sequence will become very small and hard to update. To solve this we use the LSTM model. 

![LSTM](/assets/images/LSTM_gif.gif "LSTM")

{:.image-caption}
*Inner workings of an LSTM cell - Credit: [Raimi Karim](https://towardsdatascience.com/animated-rnn-lstm-and-gru-ef124d06cf45)*


The LSTM alleviates this by reducing the information the model needs to remember and also scaling up the importance of words which it learns to be important to solving the problem.

This is achieved via a series of 'gates' that are themselves feed-forward neural networks. The gates apply various matrix algebra calculations followed by specific non-linear transformations to scale unneeded information to zero and important information to higher values. By doing this, the model can focus on what is actually important in answering the question at hand and therefore handle longer sequences than a standard RNN. I highly recommend reading Chris Olah's excellent [blog on LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn more about how the model works in detail.

## Dataset

The dataset used comes from the Kaggle compeition [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/description).

As a quick overview, the dataset contains approximately 1.8m online comments from a variety of sources. Each comment has been labelled indicating whether the comment is toxic or not. We are also provided with identity labels, which show whether a particular comment had a specific mention of a certain identity.

The identity labels are particularly important. We use these labels to subset the data into comments that specifically mention certain identity groups and then assess the performance of our models based on the metrics which we have described in more detail further below. Labelling was done via human annotators and scores were averaged to get a final label. As we discuss in our final conclusions at the end of this post, even the use of human annotations introduces another layer of bias that needs to be considered, but the use of multiple annotators helps reduce this somewhat.

As with the Kaggle competition rules, we will be specifically focussing on the below subgroups. These subgroups have been chosen due to there being more than 500 examples of each case mentioned in our test set:

   * Male
   * Female
   * Homosexual, Gay, or Lesbian
   * Christian
   * Jewish
   * Muslim
   * Psychiatric or Mental Illness

## How can we measure bias?

A lot of research and thought has gone into this and often this depends on the type of bias we are trying to reduce. In this case, Jigsaw AI have provided a 'final bias metric' as part of the Kaggle competition that borrows heavily from a standard classification metric, the Reciever Operating Characteristic - Area Under Curve (ROC-AUC). We split the data into the subgroups mentioned earlier and calculate the AUC for each subgroup. These are then combined with the overall AUC to give a final score. Effectively we weight the standard ROC-AUC with performance against specific identity subgroups. A high overall AUC but much lower weighted AUC suggests the model is heavily biased despite strong overall performance at classification.  

If you are interested you can find Jigsaw AI's explanation of the metric [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation). In addition to this, Jigsaw AI have written [this paper](https://arxiv.org/abs/1903.04561) which delves more deeply into the development of the metric we are using. 

Ultimately, the final metric is calculated as follows: 

$score = w_0 AUC_{overall} + \sum_{a=1}^{A} w_a M_p(m_{s,a})$

$A$ = number of submetrics (3)

$m_{s,a}$ = bias metric for identity subgroup $s$ using submetric $a$

$w_a$ = a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

We will be comparing our models on the following:
   * Accuracy Score
   * F1 Score
   * Final Bias Metric

## Process
For this project my main aim is to train an LSTM model that is able to classify toxic comments. I will also compare this to a handful of traditional ML models where we have also applied standard NLP pre-processing methods to prepare the text data. If you are interested in looking at the code, I've mentioned the relevant notebook files for each part of the process from the GitHub repo. 

### Traditional ML models
*see ```ML_Models.ipynb```*

For the baseline models to compare I tested out the 3 classifiers below:

   * Logistic regression
   * XGBoost
   * Random forest

These were selected for a combination of speed and general performance at classification. In my view, logistic regression is always a great model to try out on a task given its ease of interpretability and generally strong performance. The other two models, XGBoost and random forest are both ensemble methods which generally also show very strong performance and speed. It was also our intention to try other models including SVM and naive Bayes, however we were limited by computing power and a relatively short timeframe. 

Hyperparameter optimization has been carried out for each model and calculate the metrics for each. Given the final bias metric cannot be easily loaded into Scikit-Learn's cross validation functions, we used the standard ROC-AUC metric as a proxy to find the best set of hyperparameters and regularization strength for our models. 

Text preprocessing can be found in the ```preprocessing.ipynb``` file. For our three models above we used the same NLP pipeline that covered the below:
   
   * Cased to lower
   * Expanded contractions
   * Tokenization
   * Removed punctuation
   * Removed stop words
   * Lemmatized words 

As the dataset was made up of online comments I had to factor in the numerous cases of non-standard language, such as deliberate misspellings, slang, emojis, and so on. To do so I took advantage of pre-made tools such as [NLTK's tweet tokenizer](https://www.nltk.org/api/nltk.tokenize.html). In a perfect world, there would have been more time to parse through the processed text to correct any other edge-cases but the size of the dataset made this unrealistic.

### LSTM 
*see ```NN_model.ipynb```*

#### Word embeddings for neural networks 
In terms of word embeddings, we will be using the pre-trained [GloVE Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors](https://nlp.stanford.edu/projects/glove/) word embeddings. These have been trained on a common crawl of the web, covering 2.2m different words and containing 840B tokens. These word embeddings have 300 dimensions, which would suggest that each word should be unique enough to capture contextual differences. 

The selection of which word embeddings to use is particularly important as they will directly impact how the model understands each word. I selected the common crawl GloVE over the Wikipedia set as the latter uses language in a different way to how people talk on online comment platforms. Wikipedia is written in a formal style of English which is unlikely to translate well to a domain such as online comments. 

#### Text preprocessing for neural networks
The process of getting text ready in this instance is quite different than what we did for the ML models earlier. Here our aim is more directed towards getting as many words in our vocabulary to match up with words in the embeddings file. So in this case we will avoid taking steps such as stop word removal and lemmatization. In addition, the word embedding file we have used includes vector representations for certain emojis, symbols and punctuation, which means that these characters need not be removed. I focussed my attention more on fixing misspellings / slang, incorrect use of punctuation such as apostrophes, and expanding contractions. 

As a result, I was able to increase the vocabulary coverage percentage from 15.8% to 50.2% which was a very positive result! With more time I could look to increase this even further, but at this stage having over 50% coverage felt like it would be enough for now. 


![model](/assets/images/tf_summary.png){:height="330px" width="600px"}

{:.image-caption}
*Tensorflow model summary*



#### Model set-up
The next stage is to train the keras tokenizer on our pre-processed corpus. This creates a large dictionary of words to unique indexes based on the number of occurances of that word. Once the trained tokenizer is ready we call the ```text_to_sequences``` method which will then convert our comments into word indexes to pass into the model. We also need to pad each sequence to a pre-set length so that they are all the same length when fed into the model.

The final preperation stage is to create the embedding matrix. This will be a matrix of word embedding vectors, taken from our embeddings file, for all words where we can find a match in our vocabulary. This matrix will then be used as the weights in our embedding layer as discussed in the next stage. 

#### Input layer and applying the word embeddings
We will pass the text sequences into a input layer with a size determined by the maximum size we set for the sequences. Controlling sequence length is a way for us to reduce the complexity of the model and how long it takes for us to train. The next layer is the embedding layer, where the text sequences are converted into the vector representations. To do this, we pass in the tokenized sequences which are then compared to a set of fixed weights given by the matrix of word embeddings created earlier. This layer is frozen so the weights do not change during the training process.

#### LSTM Layers and final output
For this model, I decided on using a single bidirectional LSTM layer, using tanh activation. The difference between this and a standard LSTM is that the model reads accross a given sequence both forwards and backwards and then combines the output of each pass through. The idea behind doing this is that the words which come after a given word also give useful context to a word, therefore we should build better understanding by reading through the sequence in each direction. LSTM layers can be stacked on each other to try and parse out further information, however this adds to an already very long training process. 

The data then goes through a global max pooling layer before being passed through a 462 node dense layer using ReLU activation. The final layer is a 2 node output layer which uses the sigmoid activation function to generate binary classes.  

## Results:

| Model | Accuracy | F1 | Final Bias Metric |
|:--------|:-------:|:--------:|:--------:|
| Logistic   | 94.7%   | 59.9%   | 71.3%   |
| XGBoost   | 94.4%   | 53.7%   | 66.6%   |
| Random Forest   | 94.2%   | 47.9%   | 61.1%   |
|=====
| **LSTM**   | **95.3%**   | **68.1%**   | **92.0%**   |   

THe LSTM model outperformed the three other models trained accross all the metrics tested. In particular, the result for the final bias metric was significantly better at 92%. This compares favorably to the next highest score of 71.3% and justifies our initial expectation that the LSTM model would perform better than the other models at minimising bias. 

Delving into the metrics further we can see that all 4 models somewhat struggled with their recall. Once again, the  LSTM model was the best here, but the score was only 62.2%. I believe that one key reason for this is that the dataset contains a significant class imbalance, with only 8% of the ~1.8m comments being toxic. As a result, the models do not have enough toxic cases to train on that allow them to capture the different nuances in toxicity. We will mention this further in the section below.

Overall, I consider the results to be a positive indication of our ability to reduce bias in machine learning. The LSTM model has shown clear superiority over the other models in detecting toxic comments while minimising bias. 



## Future avenues of exploration

 We have learned how we can combine RNNs with carefully selected word embeddings to achieve models with better contextual understanding of the comments they are classifying. However, there remains many areas to investigate further:
 
   1. Addressing class imbalance. I believe a driver of the lower than expected recall levels for the models we trained is the large class imbalance present. In the future we would like to re-run the models after using a combination of different techniques to address this imbalance such as up-sampling, down-sampling, and SMOTE. 
   
   2. The impact of different word embeddings. How text data is converted into numerical data is a key factor in any model. For our LSTM, we used word embeddings trained from a common crawl of the internet. What would be the impact of using word embeddings which had been trained on online comments specifically? A different route would be to see the impact of using a different methodology to develop the word embeddings. In many SotA solutions to NLP tasks, researchers have begun using transformer models such as Google's BERT to generate the word embeddings. 
   
   3. Leading on from this - can we achieve superior results in our traditional ML models using more complex word embeddings?
   
   4. Can we improve the models further by carrying out more exhaustive hyperparameter optimziation? I was significantly limited by available computing power, which meant that I had to comprimise on my hyperparameter optimization.
   
   5. Different neural network architectures. We would like to see the impact of adding different layers to our NN architecture such as an additional bidirectional LSTM layer. Outside of LSTMs, can we achieve similar results using CNN models which are generally much faster at training and inference? Transformer models would be a further area of research.
   

## Conclusions

Bias in machine learning will continue to exist as long as society and the data it produces exhibits bias. We have shown how using more complex models which have a greater ability to parse context of sentences can help alleviate this to an extent, but it is unlikely we will be able to fully remove bias without removing it from the data itself. 

One of the other key takeaways is how we may need to re-think standard NLP processes such as lemmatization and the removal of stop words. You will note that we did not do this for our LSTM model as the aim was to increase the number of words we had a word-embedding for. By simplifying our text using processes such as lemmatization are we removing important contextual information that we need to train models better at minimising bias?

Hopefully we can build on the improvements delivered by neural network architectures to build out ML models that promote inclusiveness and fairness for all members of society.
 
