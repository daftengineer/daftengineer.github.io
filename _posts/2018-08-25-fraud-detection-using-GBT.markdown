---
layout: post
title: "GRADIENT BOOSTED TREE ML2"
date: 2018-08-25
description: Fraud Detection using Gradient Boosted Trees Algorithm
image: "background-image: url('https://daftengineer.github.io/media_assets/Fraud2.jpg');"
image-sm: https://daftengineer.github.io/media_assets/Fraud2.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">Here, we are going to tackle the same problem as ML1 but with different algorithm. The algorithm is called Gradient Boosted Trees. It is type of decision tree algorithm optimising differential loss function by using boosting. Unlike last project, I will be using IntelliJ IDEA for development purposes. And the application will be submittable to spark cluster. So unlike last time, I won't be able to show you, the output of individual lines of code so instead I will be explaining the code and and show the output at the end.</p>
<p style="text-align:justify;">Here is the development environment looked like while Developing the program.</p>
  <img src="https://daftengineer.github.io/media_assets/ml2p1.png" />
<p style="text-align:justify;">Now, I will explain every line of code here. First, it is important to start spark session for any spark submittable application. And then, steps for cleaning our data start. We have data in CSV format so we will load them here. we need implicit functions (like col) to run so we will need spark implicits to be imported. Our Label data is "isFraud" column which requires to be in double in order to work with our algorithm so in last step we changed its datatype.</p>
  <img src="https://daftengineer.github.io/media_assets/ml2p2.jpg" />
<p style="text-align:justify;">We need to remove all other unnecessary column which won't be required for the process of analysis. But we have type column which is in string format and we need that in double as well so we will use StringIndexer. And then using variable "cleanandTransformed" we will make a dataframe to make a single entity and cache it on the memory.</p>
   <img src="https://daftengineer.github.io/media_assets/ml2p3.jpg" />
<p style="text-align:justify;">Now, in order to input the data in Gradient Boosting Algorithm available in spark, we need the data in format of LabeledPoint and in RDD format so next, we will do exactly the same and split the data in two variables, one for training and other for testing.</p>
   <img src="https://daftengineer.github.io/media_assets/ml2p4.jpg" />
  
<p style="text-align:justify;">After that we will map the data to LabeledPoint format RDD and Train the model using training data. Here, appropriate hyperparameters are considered in order to get proper result. Then test data is predicted in code and mean squared error for test data is also calculated.</p>
 <img src="https://daftengineer.github.io/media_assets/ml2p5.jpg" />
 <p style="text-align:justify;">Here MSE is 5.487364620938629E-4 which is quite good as we just used sample(10MB) data and not the full data. This shows how effective the gradient boosted tree algorithm is when it comes to classification.</p>
   <img src="https://daftengineer.github.io/media_assets/ml2p6.jpg" />
 <p style="text-align:justify;"> In last line of code we saved the model for future use. The saved files are compressed with snappy and serialized using parquet. Which can be found in HDFS as shown below</p>
   <img src="https://daftengineer.github.io/media_assets/ml2p7.jpg" />

<p style="text-align:justify;">The code in this article can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/FraudDetectionWithGBT.scala">here.</a></p>
</div>
