---
layout: post
title: "Fraud Detection ML1"
date: 2018-08-20
description: This is simple fraud Detection Program in Spark
image: "background-image: url('https://daftengineer.github.io/media_assets/fraud.jpg');"
image-sm: https://daftengineer.github.io/media_assets/fraud.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">This is the implementation of a typical Fraud Detection Algorithm using Apache Spark and MLlib. The algorithm is not widely used because of its simplicity but we can use it to compare it with other different classification algorithm and we can benchmark what needs to be done and what is required to be changed. The aim of this implementation is, to make a spark programmer get used to typical MLlib processing pipeline of cleaning/transforming/analysing data</p>
<p style="text-align:justify;">The algorithm is called Logistic Regression and There are enough article available online so I wouldn't be explaining it here. In a nutshell, This is go to algorithm when it comes to binary classification. Here all the script have been made is in spark shell using scala.</p> 
<p style="text-align:justify;">Now, any batch mode analysis has at least 3 stages: <a href="https://en.wikipedia.org/wiki/Extract,_transform,_load">Extract, Transform/Analyse, Load</a>. All these step are considered here. The data, here, is simulated data, acquired from <a href="https://www.researchgate.net/profile/Stefan_Axelsson4/publication/313138956_PAYSIM_A_FINANCIAL_MOBILE_MONEY_SIMULATOR_FOR_FRAUD_DETECTION/links/5890f87e92851cda2568a295/PAYSIM-A-FINANCIAL-MOBILE-MONEY-SIMULATOR-FOR-FRAUD-DETECTION.pdf">Payment Simulator</a> from Modeling and Simulation Conference.</p>
  <p style="text-align:justify;">The first step is to extract the data. Here data is stored in hdfs and is in csv file so we will use dataframe api to make a dataframe of CSV file(Around 500MB).<font size="1">Note: (Here I have assumed that all the required libraries are imported already.)</font></p>
  <img src = "https://daftengineer.github.io/media_assets/ml1p1.jpg" />
    <p style="text-align:justify;">Now, we are in transform section. After Loading, we need to cleaning out all the column which aren't necessary for the analysis </p><img src = "https://daftengineer.github.io/media_assets/ml1p2.jpg" />
    <p style="text-align:justify;">String datatype don't get analysed by the algorithm so we need to make the index of these string values. Here we know, there are 5 strings so we can take the advantage of this fact but we will use StringIndexer to be cautious</p><img src = "https://daftengineer.github.io/media_assets/ml1p3.jpg" />
    <p style="text-align:justify;">After transformation, We would be able to see the dataframe like below: </p><img src = "https://daftengineer.github.io/media_assets/ml1p4.jpg" />
    <p style="text-align:justify;">In order to put values through algorithm we need vector of the row values so we will vectorize the row. The point here to be noted, is that, as we go on through different stages of pipeline, we remove all the null rows we have so that doesn't impact our last stages.</p><img src = "https://daftengineer.github.io/media_assets/ml1p5.jpg" />
  <p style="text-align:justify;">So, this is the step in which we will normalize the vector so we can analyse it without huge overhead and without spiking to too low or too high value as normalization will keep those values in 0 to 1 interval</p>
  <img src = "https://daftengineer.github.io/media_assets/ml1p6.jpg" />
  <img src = "https://daftengineer.github.io/media_assets/ml1p7.jpg" />
  <p style="text-align:justify;">Now, we will Split the dataframe in to parts one is training(80%) and another is testing(20%). After that we will apply Logistic Regression, Which requires model to be built first and then getting tested. After the test we will store the result and Compare it Using the matrices</p>
  <img src = "https://daftengineer.github.io/media_assets/ml1p8.jpg" />
  <img src = "https://daftengineer.github.io/media_assets/ml1p9.jpg" />
  <p style="text-align:justify;">Finally, We can print the result</p>
  <img src = "https://daftengineer.github.io/media_assets/ml1p10.jpg" />
  <p style="text-align:justify;">Considering its Simplicity, AUC of 0.74 is quite good which indicates effectiveness of the algorithm </p>
<p>&nbsp;</p>
</div>
