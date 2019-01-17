---
layout: post
title: "automated detection of fake news using nlp models"
date: 2018-10-02
description: Finding out the news is fake or not using several machine learning models.
image: "background-image: url('https://daftengineer.github.io/media_assets/fakenews.jpg');"
image-sm: https://daftengineer.github.io/media_assets/fakenews.jpg
---

<div style="color:black;"><p></p>

<p style="text-align:justify;">With increase in polarisation of political landscape, there are abundant amount of fake news spreading all around us. For this project, I needed reliable data which was tough to find but I found one on a too-under-interacting kaggle competition. I will be using this data for this article. </p>
<p style="text-align:justify;">The basic logic, here, is to take the dataset and make a model for document embedding with paragraph vectors (100 dimensions) of articles given in dataset. Then assembling a vector, where title is paragraph vector with 25 dimensions and author is same paragraph vector with 25 dimensions. The reason here to make both title and author paragraph vector instead of one-hot embedding is both play important role in making news fake or true.(e.g. News is more likely to be fake if it is created by conspiracy theorist.) Then I will train these 150 dimensional vectors on 3-layer fully connected neural network using scikit-learn</p>
<p style="text-align:justify;">As always, source code for this article can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/FakeNewsDetection.ipynb">here</a></p>
<p style="text-align:justify;">First step is to import all the library which are required in the project</p>
<img src="https://daftengineer.github.io/media_assets/ml9p1.jpg" />
<p style="text-align:justify;">Now, I need the dataset which is saved in my personal google drive, which can be accessed from below direct links and I needed to make a pandas dataframe for it. So, the resultant dataset will look like below.</p>
<img src="https://daftengineer.github.io/media_assets/ml9p2.jpg" />
<p style="text-align:justify;">I will be needing to make paragraph vector of 100 dimensions so that we can use it to train the neural network. For this, I used gensim doc2vec model and saved the model so, I can use it in future and we will repeat the same process for title using 25 dimensions and author using 25 dimensions. All other hyperparameters are set as per needs be.</p>
<img src="https://daftengineer.github.io/media_assets/ml9p3.jpg" />
<p style="text-align:justify;">Now, I will load the trained models and clean the data and convert them into proper datatype so that I can process them using Multi Layer Perceptron. In other words, I am transforming the data into vectors using the model I trained earlier.</p>
<img src="https://daftengineer.github.io/media_assets/ml9p4.jpg" />
<p style="text-align:justify;">So resultant dataset would look like this.</p>
<img src="https://daftengineer.github.io/media_assets/ml9p5.jpg" />
<p style="text-align:justify;">And finally, I converted pandas columns to numpy array made appropriate changes and made the vectors processable (Vector Assembly) by converting them to list and the processed it using sklearn's MLPClassifier. And then tested against 25% of remaining data. The accuracy, I got here is astounding <b>95%</b></p>
<img src="https://daftengineer.github.io/media_assets/ml9p6.jpg" />

 </div>

