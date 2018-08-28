---
layout: post
title: "GRADIENT BOOSTED TREE ML2"
date: 2018-08-25
description: Fraud Detection using Gradient Boosted Trees Algorithm
image: "background-image: url('https://daftengineer.github.io/media_assets/Fraud2.jpg');"
image-sm: https://daftengineer.github.io/media_assets/Fraud2.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">Here, we are going to tackle the same problem as ML1 but with different algorithm. The algorithm is called Gradient Boosted Trees. It is type of decision tree algorithm optimising differential loss function by using boosting. Unlike last Project I will be using IntelliJ IDEA for development purposes. And the application will be submittable to spark cluster. So unlike last time I wont be able to show you the output of individual lines of code so instead I will be explaining the code and and show the output at the end.</p>
<p style="text-align:justify;">Here is the development environment looked like while Developing the program</p>
  <img src="https://daftengineer.github.io/media_assets/ml2p1.png" />
<p style="text-align:justify;">Now I will explain every line of code here. First it is important to start spark session for any spark submittable application. And then steps for cleaning our data start. We have data in CSV Format so we will load them here. we need implicit functions(like col) to run so we might will need spark implicits to be imported</p>
  <p style="text-align:left;color:white;"><span style="background-color:rgb(39, 124, 163);">
    val spark = SparkSession<br />
      .builder()<br />
      .appName("Fraud Detection With GBT")<br />
      .master("yarn")<br />
      .getOrCreate()<br />
    import spark.implicits._<br />
val df2=spark.read.format("csv").option("header","true").option("inferSchema","true")<br />
    .load("hdfs://nn01.itversity.com:8020/user/pratiksheth/frauddetectiondata.csv")<br />
    val df = df2.withColumn("isFraud", col("isFraud").cast("double"))<br /></span></p>
  
<p>&nbsp;</p>
</div>
