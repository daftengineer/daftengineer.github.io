---
layout: post
title: "COLLABORATIVE FILTERING ON NETFLIX DATA ML5"
date: 2018-09-06
description: predicting the behaviour of user depending on their movie choices
image: "background-image: url('https://daftengineer.github.io/media_assets/cf.jpg');"
image-sm: https://daftengineer.github.io/media_assets/cf.jpg
---

<div style="color:black;"><p></p>
  <p style="text-align:justify;">Collaborative filtering is method in which preference of an individual is predicted on basis of other individuals. We have several ways through which we can tackle this problem which include memory based and model based techniques. Here we will implement model based collaborative filter using Alternating Least Squares as regression model. </p>
  <p style="text-align:justify;">Before everything I needed to make the data usuable by spark so I wrote the scala script to change the data as per I wanted</p>
  <img src="https://daftengineer.github.io/media_assets/ml5p0.jpg" />
  <p style="text-align:justify;">In this project, I have considered grid of parameters which include regularization parameter and number of iteration. Using that we will be able to compute ALS and Predict the user rating.</p>
<img src="https://daftengineer.github.io/media_assets/ml5p2.jpg" />
  <p style="text-align:justify;">Using this model, we can recommend movies to every user (model has number of iteration 10 and regularization parameter 0.05). Which will look like below:</p>
<img src="https://daftengineer.github.io/media_assets/ml5p1.jpg" />
  <p style="text-align:justify;">And We can also see the Root mean square error of each model and use as per needs be. Here all models are good and variance is also close so any of the model can be used.</p>
<img src="https://daftengineer.github.io/media_assets/ml5p3.jpg" />
  <p style="text-align:justify;">Source code for this project can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/Collab.scala">here</a></p>
</div>
