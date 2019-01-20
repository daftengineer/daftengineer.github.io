---
layout: post
title: "predicting the outcome of pregnancy from ahs dataset using apache spark"
date: 2018-08-31
description: Predicting the outcome of pregnancy using data collected from annual health survey of India
image: "background-image: url('https://daftengineer.github.io/media_assets/preg.jpg');"
image-sm: https://daftengineer.github.io/media_assets/preg.jpg
---

<div style="color:black;"><p></p>

<p style="text-align:justify;">Census Bureau of India, annually orchestrates Annual Health Survey in different states. The Raw data about this survey is available online for public use so I used it to train classification model and predict the outcome of pregnancy.</p>
<p style="text-align:justify;">According to dataset available with us outcome of pregnancy is in binary format. And we have 
"Pipe Seperated Value" file with more than 200 columns. This time, the classification is tackled using grid of parameters on gradient boosted tree algorithm which will give use different result which will give us the idea about which parameter to choose. Schema for the dataset is given below (Beware: It is huge).</p>
<img src = "https://daftengineer.github.io/media_assets/ml3p1.jpg" />
<p style="text-align:justify;">Here, I will be explaining the code which has been developed using IntelliJ IDEA. We will start by loading the data and cleaning it. Though, this is a csv file it is pipe seperated so we will pass an option of delimited by pipe(|). We need to use only the few of the fields in order to train our model. But we will need those in "double" datatype so first we will cast them to double. Then, we will select and cache the fields which are needed for the gradient boosting. And after it, we will fill all null values with 0</p>
<img src = "https://daftengineer.github.io/media_assets/ml3p2.png" />
<p style="text-align:justify;">Now, we need our data to be in LabelPoint Format(each row : (Label, Vector of Features)). So we made an RDD of it. <br /> Our label data has 3 values:<br /> <span style="text-align:left;">1) nulls (which are now filled with 0s)<br /> 1) "1" : Positive Outcome <br />2) "2" : Negative Outcome</span><br />So we have to remove those zeros from our data as they are not giving us any information. Data is then split and gets converted to LabelPoint RDD.</p>
<img src = "https://daftengineer.github.io/media_assets/ml3p3.jpg" />

<p style="text-align:justify;">We have taken 5 different values for "Number of Iteration" and "Depth of Tree" parameters so that we can get 25 different combination of parameters using which we can check which one suits best with our data. We will evaluate them using MSE, They get from every iteration of modeling and testing. This is not the proper way to do it so we will just use 20% of dataset which is saved as FILENAME_SMALL for our calculation.</p>
<img src = "https://daftengineer.github.io/media_assets/ml3p4.png" />
<p style="text-align:justify;">So, we can see below that lowest MSE score we got was 0.0159922495298564 from iteration 18 depth 25 which shows these are the best parameters for our data. </p>
<img src = "https://daftengineer.github.io/media_assets/ml3p5.jpg" />
<p style="text-align:justify;">As always source code for this project can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/PregnancyOutcome.scala">here</a></p>

</div>
