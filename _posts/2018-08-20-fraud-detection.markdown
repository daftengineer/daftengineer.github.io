---
layout: post
title: "Fraud Detection ML1"
date: 2018-08-20
description: This is simple fraud Detection Program in Spark
image: "background-image: url('https://daftengineer.github.io/media_assets/fraud.jpg');"
image-sm: https://daftengineer.github.io/media_assets/fraud.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">This is the implementation a typical Fraud Detection Algorithm. It is not widely used because of its simplicity but we can use it to compare it with other different prediction algorithm and we can benchmark what needs to be done and what is required to be changed. The aim of this implementation is make a spark programer used to typical MLlib work around and process of cleaning/transforming/analysing data</p>
<p style="text-align:justify;">The algorithm is called Logistic Regression and There are enough article available online so I wouldn't be explaining it here. In a nutshell, This is go to algorithm when it comes to binary classification. Here all the script have been made is in spark shell using scala.</p> 
<p style="text-align:justify;">Now any batch mode analysis has at least 3 stages: <b>Extract, Transform/Analyse, Load</b>. All these step are considered here. The data here is simulated data acquired from <a href="https://www.researchgate.net/profile/Stefan_Axelsson4/publication/313138956_PAYSIM_A_FINANCIAL_MOBILE_MONEY_SIMULATOR_FOR_FRAUD_DETECTION/links/5890f87e92851cda2568a295/PAYSIM-A-FINANCIAL-MOBILE-MONEY-SIMULATOR-FOR-FRAUD-DETECTION.pdf">Payment Simulator</a> from Modeling and Simulation Conference.</p>
  <p style="text-align:justify;">The first step is to extract the data. Here data is stored in hdfs and is in csv file so we will use dataframe api to make a dataframe of CSV file.Note: (Here I have assumed that all the required libraries are imported already.</p>
  <img src = "https://daftengineer.github.io/media_assets/ml1p1.jpg" />
    <p style="text-align:justify;">Now we are transform section. After Loading we need to cleaning out all the column which arent necessary for the analysis </p><img src = "https://daftengineer.github.io/media_assets/ml1p2.jpg" />
    <p style="text-align:justify;">Now we need to make index the string to make it work with algorithm</p><img src = "https://daftengineer.github.io/media_assets/ml1p3.jpg" />
    <p style="text-align:justify;">After transformation We would be able to see the dataframe like below: </p><img src = "https://daftengineer.github.io/media_assets/ml1p4.jpg" />
    <p style="text-align:justify;"></p>
<p>&nbsp;</p>
</div>
