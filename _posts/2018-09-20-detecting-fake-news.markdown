---
layout: post
title: "detecting fake news using nlp models ml9"
date: 2018-09-20
description: Finding out the news is fake or not using several machine learning models.
image: "background-image: url('https://daftengineer.github.io/media_assets/fakenews.jpg');"
image-sm: https://daftengineer.github.io/media_assets/fakenews.jpg
---

<div style="color:black;"><p></p>

<p style="text-align:justify;">With increase in polarisation of political landscape, there are abundant amount of fake news spreading all around us. For this project, I needed reliable data which was tough to find but I found one on a too-under-interacting kaggle competition. I will be using this data for this article. </p>
<p style="text-align:justify;">The basic logic, here, is to take the dataset and make a model for document embedding with paragraph vectors (100 dimensional) of articles given in dataset. Then assembling a vector, where title is paragraph vector with 25 dimensions and author is glove of 50 dimension (Pretrained/glove-wiki-gigaword-50). Then training these 175 dimensional vectors on 3-layer fully connected neural network using keras (tensorflow backend).</p>
<p style="text-align:justify;"></p>
<p style="text-align:justify;"></p>
<p style="text-align:justify;"></p>
<p style="text-align:justify;"></p>
<p style="text-align:justify;"></p>
 </div>

