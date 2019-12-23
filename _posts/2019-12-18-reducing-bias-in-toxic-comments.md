---
layout: post
title: Reducing bias in toxic comment classification
excerpt: "Capstone Project"
categories: [projects, NLP, Bias, LSTM ]
mathjax: true
comments: true
image:
  feature: https://images.unsplash.com/photo-1440635592348-167b1b30296f?crop=entropy&dpr=2&fit=crop&fm=jpg&h=475&ixjsv=2.1.0&ixlib=rb-0.3.5&q=50&w=1250
  credit: thomas shellberg
  creditlink: https://unsplash.com/photos/Ki0dpxd3LGc
---

As Machine Learning is increasingly used in our day to day lives, issues surrounding the reinforcement of existing biases against minority identities are increasingly important. In this project, I have explored this from the scope of online toxic comment classification. We have explored the short-comings of traditional machine learning models and NLP processes and have sought to train a better model which is better able to identify toxic comments while minimising bias against minority identites.  

This post provides an overview of my process and findings. You can find the code for this project on my [github](https://github.com/GovindSuresh/reducing-bias-in-toxicity-classification)

### How do traditional ML models and NLP processes reinforce bias?

Ultimately, ML models learn from the data they are trained on, therefore the models will pick up on biases that already exist in the data due to societal norms. When looking at online comments we unfortunately see that a large number of toxic comments are directed at a variety of minority groups (e.g. Black, Muslim, LGBTQ) and therefore may repeatedly contain mentions of these groups. 

Traditional ML classification models perform very well in terms of accuracy when identifying toxic comments. However due to the bias in the data described above, they have a tendency to view words that refer to a particular minority identity to be indicative of toxicity. This leads to a situation where even positive comments which happen to mention these identities being classified as toxic (false positives).

While having a strong overall accuracy is positive, this issue of false positives has the impact of excluding minority groups from talking about themselves in online communities and can therefore simply reinforce existing bias.

This is further exacerbated by standard NLP processing steps. It is common to reduce the complexity of the text data being passed in by going through certain steps such as lemmatization, and the removal of stop-words. Usually, this isn't a major issue given that we keep the majority of the useful information in the text. However, when we consider complex topic matters such as toxic comments, contextual clues given by any part of a comment can be useful. In addition, these processes reduce sentences down to key words, which in toxic comment cases may just be repeated mentions of a minority identity.

To help highlight this we have taken an example from our test dataset. This comment has been labelled as 'Non-Toxic' by the labellers of our dataset, but our logistic regression model has misclassified it as toxic. We can see from the image that the model has highlighted the words 'gay', 'lesbian', 'transgender' as key words indicating toxicity (shown in green below). This is the exact problem we have mentioned above and is what we are trying to solve.

![missclass](/assets/images/missclass-lgbt2.png "Misclassified LGBT comment")

 **Original Text:** *'I think your hearts in the right place but gay men and lesbians have no issues using the correctly gendered bathrooms. Sexual orientation and gender identity are two totally different things. A transgender person can be gay, lesbian, bi, straight or any other Sexual orientation.'*

### Training a better model:
Our aim is to train a neural network model, primarily an LSTM model, to classify toxic comments. RNN's such as LSTM are optimally suited to tackling this problem due to their ability to parse through sequences such as a comment. In other words, the LSTM can build contextual understanding of a word based on the words that have come before it (and even after it in the case of a bidirectional LSTM). This should give it an advantage over a traditional ML model such as Logistic Regression which have a tendency to look at how often individual words appear in toxic comments.  

![LSTM](/assets/images/LSTM_gif.gif "LSTM")

The LSTM achieves this via a series of 'gates' that are themselves feed-forward neural networks. The gates learn which parts of a sentence to forget, which to pass on to the next cell, and which to output. By doing this, the model can focus on what is actually important in answering the question at hand and therefore handle longer sequences than a standard RNN. I highly recommend reading Chris Olah's excellent [blog on LSTM's](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) to learn more about how the model works in detail.


### Dataset

The dataset used comes from the Kaggle Compeition [Jigsaw Unintended bias in toxicity classification.](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/description)

As a quick overview, the data contains online comments and a labelled target column indicating whether the comment is toxic. In addition to this, we are provided with identity labels, which show whether a particular comment had a specific mention to a certain identity.

The identity labels are particularly important. We use these labels to subset the data into comments that specifically mention certain identity groups and then assess the performance of our models based on the metrics which we have described in more detail further below. Labelling was done via human annotators and scores were averaged to get a final label. As we discuss in our final conclusions at the end of this post, even the use of human annotations introduces another layer of bias that needs to be considered, but the use of multiple annotators helps reduce this somewhat.

As with the Kaggle competition rules, we will be specifically focussing on the below subgroups. These subgroups have been chosen due to there being more than 500 examples of each case mentioned in our test set. 

   * Male
   * Female
   * Homosexual, Gay, or Lesbian
   * Christian
   * Jewish
   * Muslim
   * Psychiatric or Mental Illness

## How can we measure bias:

So how can we actually quantify bias? A lot of research and thought has gone into this and often this depends on the type of bias we are trying to reduce. In this case, Jigsaw AI have provided a 'final bias metric' as part of the Kaggle competition that borrows heavily from a standard classification metric, the Reciever Operating Characteristic - Area Uunder Curve (AUC). We split the data into the subgroups mentioned earlier and calculate the AUC for each subgroup. These are then combined with the overall AUC to give a final score. Effectively we weight the standard ROC-AUC with performance against specific identity subgroups. A high overall AUC but much lower weighted AUC suggests the model is heavily biased.  

If you are interested you can find Jigsaw AI's explanation of the metric [here.](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/evaluation) in addition to this, Jigsaw AI have written [this paper](https://arxiv.org/abs/1903.04561) which delves more deeply into the development of the metric we are using. 

Ultimately, the final metric is calculated as follows: 

$score = w_0 AUC_{overall} + \sum_{a=1}^{A} w_a M_p(m_{s,a})$

$A$ = number of submetrics (3)

$m_{s,a}$ = bias metric for identity subgroup $s$ using submetric $a$

$w_a$ = a weighting for the relative importance of each submetric; all four $w$ values set to 0.25

We will be comparing our models on the following:
   * Accuracy Score
   * F1 Score,
   * Final Bias Metric

 
## Process:
For this project our main aim is to train an LSTM model that is able to classify toxic comments. We will then compare this to a handful of traditional ML models where we have also applied standard NLP pre-processing methods to prepare the text data. If you are interested in looking at the code, I have mentioned the relevant notebook files for each part of the process. 

#### Traditional ML models
*see the ```ML_Models.ipynb```*
For the baseline models to compare I tested out the 3 classifiers below. T

   * Logistic Regression
   * XGBoost
   * Random Forest

 These were selected for a combination of speed and general performance at classification. In my view, logistic regression is always a great model to try out on a task given its ease of interpretability and generally strong performance. The other two models, XGBoost and random forest are both ensemble methods which generally also show very strong performance and speed. It was also our intention to try other models including SVM and Naieve Bayes, however we were limited by computing power and a relatively short timeframe. 

Hyperparameter optimization has been carried out for each model and calculate the metrics for each. Given the final bias metric cannot be easily loaded into Scikit-Learn's cross validation functions, we used the standard ROC-AUC metric as a proxy to find the best set of hyperparameters and regularization strength for our models. 

Text preprocessing can be found in the ```preprocessing.ipynb``` file. For our three models above we used the same NLP pipeline that covered the below:
   
   * Cased to lower
   * Expanded contractions
   * Tokenization
   * Removed punctuation
   * Removed stop words
   * Lemmatized words 

As the dataset was made up of online comments I had to factor in the numerous cases of non-standard language, such as deliberate miss-spellings, slang, emojis, and so on. To do so I took advantage of pre-made tools such as [NLTK's tweet tokenizer](https://www.nltk.org/api/nltk.tokenize.html). In a perfect world, there would have been more time to parse through the processed text to correct any other edge-cases but the size of the dataset made this unrealistic.

#### LSTM
*see ```NN_model.ipynb```*

We will also train a neural network to answer this problem. We will start with a basic LSTM model which will be made of:
    
   * LSTM layers to read through the data
   * Dense layers
   * Output layer using sigmoid for the classes

We will be using a Bidirectional LSTM layer in our model. The difference between this and a standard LSTM is that the model reads accross a given sequence both forwards and backwards and then combines the output of each pass through. The idea behind doing this is that the words which come after a given word also give useful context to a word, therefore we should build better understanding by reading through the sequence in each direction.

In terms of word embeddings, we will be using the pre-trained [GloVE Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors](https://nlp.stanford.edu/projects/glove/) word embeddings. These have been trained on a common crawl of the web, covering 2.2m different words and containing 840B tokens. These word embeddings have 300 dimensions, which would suggest that each word should be unique enough to capture contextual differences. 
Another important factor regarding word embeddings is where they have been trained on. One common source is to train word embeddings from Wikipedia. However we did not believe that this would be appropriate for our domain given the differences in how language is used on Wikipedia than in online comments. 

The model architecture will be written in TensorFlow 2.0 code.


## Results:

| Model | Accuracy | F1 | Final Bias Metric |
|:--------|:-------:|:--------:|:--------:|
| Logistic   | 94.7%   | 59.9%   | 71.3%   |
| XGBoost   | 94.4%   | 53.7%   | 66.6%   |
| Random Forest   | 94.2%   | 47.9%   | 61.1%   |
|=====
| **LSTM**   | **95.3%**   | **68.1%**   | **92.0%**   |   

From the results we can see that the LSTM model has outperformed the three traditional ML models that we have trained across all metrics tested. In particular, the result for the final bias metric was significantly better at 92%. This compares favorably to the next highest score of 71.3% and justifies our initial expectation that the LSTM model would perform better than the other models at minimising bias!

Delving into the metrics further we can see that all 4 models somewhat struggled with their recall. Once again, the  LSTM model was the best here, but the score was only 62.2%. We believe that one key reason for this is that the dataset contains a significant class imbalance, with only 8% of the ~1.8m comments being toxic. As a result, the models do not have enough toxic cases to train on that allow them to capture the different nuances in toxicity. We will mention this further in the section below.

Overall, we would consider our results to be a positive step forwards. The LSTM model has shown clear superiority over the other models, and especially for our stated aim of reducing bias in toxic comment classification. However there remains many further avenues to explore, leading us into our next section.

## Future avenues of exploration:

Our work here can be viewed as a first step in making less biased toxic comment classifiers. We have learned how we can combine RNNs with carefully selected word embeddings to achieve models with better contextual understanding of the comments they are classifying. We have identified further improvements/avenues of exploration for our project:

   1. Addressing class imbalance. We believe a driver of the lower than expected recall levels for the models we trained is the large class imbalance present. In the future we would like to re-run the models after using a combination of different techniques to address this imbalance such as up-sampling, down-sampling, and SMOTE. 
   
   2. The impact of different word embeddings. How text data is converted into numerical data is a key factor in any model. For our LSTM, we used word embeddings trained from a common crawl of the internet. What would be the impact of using word embeddings which had been trained on online comments specifically? A different route would be to see the impact of using a different methodology to develop the word embeddings. In many SotA solutions to NLP tasks, researchers have begun using transformer models such as Google's BERT to generate the word embeddings. 
   
   3. Leading on from this - can we achieve superior results in our traditional ML models using more complex word embeddings?
   
   4. Can we improve the traditional ML models further by carrying out more exhaustive hyperparameter optimziation? We were significantly limited by available computing power, which meant that we had to comprimise on our hyperparameter optimization.
   
   5. Different neural network architectures. We would like to see the impact of adding different layers to our NN architecture such as an additional bidirectional LSTM layer. Outside of LSTMs, can we achieve similar results using CNN models which are generally much faster at training and inference? Transformer models would be a further area of research.
   

## Conclusions

Bias in machine learning will continue to exist as long as society and the data it produces exhibits bias. We have shown how using more complex models which have a greater ability to parse context of sentences can help alleviate some of this issue, but it is unlikely we will be able to fully remove bias without removing it from the data itself. 

One of the other key takeaways is how we may need to re-think standard NLP processes such as lemmatization and the removal of stop words. You will note that we did not do this for our LSTM model as the aim was to increase the number of words we had a word-embedding for. By simplifying our text using processes such as lemmatization are we removing important contextual information that we need to train models better at minimising bias?

Hopefully we can build on the improvements delivered by neural network architectures to build out ML models that promote inclusiveness and fairness for all members of society.
 