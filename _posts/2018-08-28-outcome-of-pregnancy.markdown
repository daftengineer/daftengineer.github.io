---
layout: post
title: "PREDICTING THE OUTCOME OF PREGNANCY FROM AHS DATASET ML3"
date: 2018-08-28
description: Predicting the outcome of pregnancy using data from annual health survey of India
image: "background-image: url('https://daftengineer.github.io/media_assets/preg.jpg');"
image-sm: https://daftengineer.github.io/media_assets/preg.jpg
---

<div style="color:black;"><p></p>

<p style="text-align:justify;">Census Bureau of India, annually orchestrates Annual Health Survey in Different States. The Raw data about this survey is available online for public use so I used it to train classification model and predict the outcome of pregnancy.</p>
<p style="text-align:justify;">As always source code for this project can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/PregnancyOutcome.scala">here</a></p>
<p style="text-align:justify;">According to dataset available with us Outcome of pregnancy is in binary format. And we have Pipe Seperated Value file with more than 200 columns. This time, the classification is tackled using grid of parameters on gradient boosted tree algorithm which will give use different result which will give us the idea about which parameter to choose. Schema for the dataset is given below.</p>
<img></img>
<p style="text-align:justify;">Here, I will be explaining the code which has been developed using IntelliJ IDEA. We will start by loading the data and cleaning it. Though, this is a csv file it is pipe seperated so we will pass an option of delimited by pipe(|). </p>
<p style="text-align:justify;"></p>
<p style="text-align:justify;"></p>
</div>
