---
layout: post
title: "predicting stock prices considering market sentiment ml10"
date: 2018-10-12
description: Deep learning model to predict the market price of specific index on US stock market data from news sentiment analysis data.
image: "background-image: url('https://daftengineer.github.io/media_assets/stockmarket.jpg');"
image-sm: https://daftengineer.github.io/media_assets/stockmarket.jpg
---

<div style="color:black;"><p></p>

<p style="text-align:justify;">Predicting the stock market prices has been well studied subject in whole data analytics field but the problem with stock market is, it is highly relient on the collective human behaviour and sentiment towards the market. Now volatility in market is dependent on human sentiment which is harder to quantify. The most impactful thing to human sentiment about the stock is news. In this project I will try to predict the stock market with the help of news and sentiment around it. </p>
<p style="text-align:justify;">Code for the project can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/Market_Price_Prediction_with_Market_Sentiment.ipynb">here</a></p>
<p style="text-align:justify;">For this project, first thing I needed, was a reliable source of data which I found on a kaggle competition. It has two CSV files, one for market price data and another for everyday news regarding the specific index. All sentiment values are also analysed and provided with it. Here is sample of the data</p>
<img src="https://daftengineer.github.io/media_assets/ml10p1.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p2.jpg" />
<p style="text-align:justify;">Now, in order to tackle this problem I am going to be using CNN LSTM Model which I think is the best for prediction task. Here Keras is used with tensorflow as backend</p>
<p style="text-align:justify;">First Step is to import all the required library. </p>
<img src="https://daftengineer.github.io/media_assets/ml10p3.jpg" />
<p style="text-align:justify;">Now, I will clean the data so that they can be properly analysed</p>
<img src="https://daftengineer.github.io/media_assets/ml10p4.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p5.jpg" />
<p style="text-align:justify;">For analysing the headline data, I was required to convert it to machine understandable format which can be acquired using Document embedding. Code for which is given below</p>
<img src="https://daftengineer.github.io/media_assets/ml10p6.jpg" />
<p style="text-align:justify;">I also needed to vectorize the Subject and Audience columns. For which I wrote custom vectorization logic as there weren't anything for problem here present.</p>
<img src="https://daftengineer.github.io/media_assets/ml10p7.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p8.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p9.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p10.jpg" />
<p style="text-align:justify;">When functions are used it gives us this vectors</p>
<img src="https://daftengineer.github.io/media_assets/ml10p11.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p12.jpg" />
<p style="text-align:justify;">So while parsing the news I needed to make a vector of recent past market data. To fetch it I needed to make a QueryEngine which will be called on each news instance.</p>
<img src="https://daftengineer.github.io/media_assets/ml10p13.jpg" />
<p style="text-align:justify;">After that news were needed to be parsed for which below logic was used.</p>
<img src="https://daftengineer.github.io/media_assets/ml10p14.jpg" />
<img src="https://daftengineer.github.io/media_assets/ml10p15.jpg" />
<p style="text-align:justify;">Finally the data required was to be set with proper format so it can be computer using Keras.</p>
<img src="https://daftengineer.github.io/media_assets/ml10p16.jpg" />
<p style="text-align:justify;">Lastly, neural network was implemented using keras. It has first Embedding layer, second  Convolutional layer, third pooling layer, forth LSTM layer and last Dense Layer</p> 
<img src="https://daftengineer.github.io/media_assets/ml10p17.jpg" />
<p style="text-align:justify;">With this data I got 0.126 loss value which is quite small and which makes my algorithm highly accurate</p>
<img src="https://daftengineer.github.io/media_assets/ml10p18.jpg" />
</div>

