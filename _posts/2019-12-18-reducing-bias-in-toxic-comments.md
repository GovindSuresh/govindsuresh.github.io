---
layout: post
title: Reducing bias in toxic comment classification
excerpt: "Capstone Project"
categories: [projects]
comments: true
image:
  feature: https://images.unsplash.com/photo-1440635592348-167b1b30296f?crop=entropy&dpr=2&fit=crop&fm=jpg&h=475&ixjsv=2.1.0&ixlib=rb-0.3.5&q=50&w=1250
  credit: thomas shellberg
  creditlink: https://unsplash.com/photos/Ki0dpxd3LGc
---

# Reducing bias in toxic comment classification

## Project Overview
As Machine Learning is increasingly used in our day to day lives, issues surrounding the reinforcement of existing biases against minority identities are increasingly important. We are exploring this from the scope of online toxic comment classification and how we can train models which perform better in terms of bias while maintaining strong accuracy.

This report provides an overview of our process and findings. Please explore the notebooks ```EDA.ipynb```,```preprocessing.ipynb```,```ML_models.ipynb```, and ```NN_model.ipynb``` for the code and further discussions on findings and process.

### The short-comings of traditional ML models and NLP processes

Ultimately, ML models learn from the data they are trained on, therefore the models will pick up on biases that already exist in the data due to societal norms. When looking at online comments we unfortunately see that a large number of toxic comments are directed at a variety of minority groups (e.g. Black, Muslim, LGBTQ).

Traditional ML classification models perform very well in terms of accuracy when identifying toxic comments. However due to the bias described above, they ds have a tendency to over-weight words that refer to a particular identity leading to non-toxic examples mentioning these identites being classified as toxic (False Positives). This issue is further exacerbated by standard NLP processing steps whereby sentences and words are stripped down to base representations and context is often lost 

While strong accuracy is a positive, models which are overly sensitive to minorities has the impact of excluding these groups from talking about themselves in online communities and can therefore simply reinforce existing bias. 

Below we have taken an example from our test dataset. This comment has been labelled as 'Non-Toxic' by the annotaters, but our logistic regression model has misclassified it as toxic. We can see from the image that the model has highlighted the words 'gay', 'lesbian', 'transgender' as key words indicating toxicity (shown in green below). This is the exact issue we have mentioned above and is what we are trying to solve.

![missclass](report_img/missclass-lgbt2.png "Misclassified LGBT comment")

 **Original Text:** *'I think your hearts in the right place but gay men and lesbians have no issues using the correctly gendered bathrooms. Sexual orientation and gender identity are two totally different things. A transgender person can be gay, lesbian, bi, straight or any other Sexual orientation.'*

### Training a better model:
Our aim is to train a neural network model, primarily an LSTM model, to classify toxic comments. RNN's such as LSTM are optimally suited to tackling this problem due to their ability to parse through sequences such as a comment. In other words, the LSTM can build contextual understanding of a word based on the words that have come before it (and even after it in the case of a bidirectional LSTM). This should give it an advantage over a traditional ML model such as Logistic Regression which have a tendency to look at how often individual words appear in toxic comments.  

## Process:
We will be training 4 different models in total, 3 standard ML models that follow a regular NLP pre-processing pipeline and 1 LSTM deep learning network.

#### Traditional ML models
This is primarily an NLP task, our X feature matrix will be based off the text from online comments. We have defined a pre-processing pipeline in the ```preprocessing.ipynb``` notebook to use for our our ML classifiers and a seperate pre-processing pipeline for the neural network models we are planning on training.

From the classic ML classifer models, we intend to use the following models - our base word embedding technique will be TF-IDF: 

   * Logistic Regression
   * XGBoost
   * Random Forest
   
We will carry out hyperparameter optimization for each model and calculate the metrics for each. The actual implentation of hyperparameter optimization will be restricted by our available computing power. Please see the ```ML_Models.ipynb``` for more details.

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

## Dataset and Metrics:

### Dataset

The dataset used comes from the Kaggle Compeition [Jigsaw Unintended bias in toxicity classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/overview/description)

As a quick overview, the data contains online comments and a labelled target column indicating whether the comment is toxic. In addition to this, we are provided with identity labels, which show whether a particular comment had a specific mention to a certain identity.

The identity labels are particularly important. We use these labels to subset the data into comments that specifically mention certain identity groups and then assess the performance of our models based on the metrics which we have described in more detail further below.

### Measuring Bias

As mentioned earlier, simply using accuracy is not enough for what we are trying to solve. As part of the Kaggle competition that the dataset is from, Jigsaw AI have provided a new metric that they have developed to assess how biased a model is against specified entities.

#### Overall ROC-AUC:

This is the standard ROC-AUC for the full evaluation set. In other words this is the area under the Reciever Operating Characteristic curve. It compares the true positive and false positive rates of a binary model.

#### Subgroup ROC-AUC:

Here, we restrict the data set to only the examples that mention the specific identity subgroup. A low value in this metric means the model does a poor job of distinguishing between toxic and non-toxic comments that mention the identity. No different to the standard AUC, just for a particular subgroup.

#### BPSN AUC:

BPSN (Background Positive, Subgroup Negative) AUC: Here, we restrict the test set to the non-toxic examples that mention the identity and the toxic examples that do not. A low value in this metric means that the model confuses non-toxic examples that mention the identity with toxic examples that do not, likely meaning that the model predicts higher toxicity scores than it should for non-toxic examples mentioning the identity.

#### BNSP AUC:

BNSP (Background Negative, Subgroup Positive) AUC: Here, we restrict the test set to the toxic examples that mention the identity and the non-toxic examples that do not. A low value here means that the model confuses toxic examples that mention the identity with non-toxic examples that do not, likely meaning that the model predicts lower toxicity scores than it should for toxic examples mentioning the identity.


#### Generalized Mean of Bias AUCs
To combine the per-identity Bias AUCs into one overall measure, we calculate their generalized mean as defined below:

$M_p(m_s) = \left(\frac{1}{N} \sum_{s=1}^{N} m_s^p\right)^\frac{1}{p}$

Where:

$M_p$ = the $p$th power-mean function

$m_s$ = the bias metric $m$ calulated for subgroup $s$

$N$ = number of identity subgroups

For this competition, JigsawAI use a $p$ value of -5 to encourage competitors to improve the model for the identity subgroups with the lowest model performance.

### Final Metric
We combine the overall AUC with the generalized mean of the Bias AUCs to calculate the final model score:

$score = w_0 AUC_{overall} + \sum_{a=1}^{A} w_a M_p(m_{s,a})$

$A$ = number of submetrics (3)

$m_{s,a}$ = bias metric for identity subgroup $s$ using submetric $a$

$w_a$ = $a$ weighting for the relative importance of each submetric; all four $w$ values set to 0.25


We will be comparing our models on the following:
    * Accuracy Score
    * F1 Score,
    * Final Bias Metric

By doing this we will be assessing our models ability to generally be correct at identifying toxic comments; how strong their precision and recall are; and their ability to do so without biasing against minority identities.

As with the actual Kaggle competition, we will be measuring the final bias metric against the following subgroups:

   * Male
   * Female
   * Homosexual, Gay, or Lesbian
   * Christian
   * Jewish
   * Muslim
   * Psychiatric or Mental Illness

These subgroups have been chosen due to there being more than 500 examples of each case mentioned in our test set. 


## Results:

![results_table2](report_img/results_table.png "Results Table")

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
 