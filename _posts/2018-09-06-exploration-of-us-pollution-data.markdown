---
layout: post
title: "exploratory analysis of huge usa pollution data using apache spark"
date: 2018-09-06
description: Exploring the USA pollution data provided to the public by EPA
image: "background-image: url('https://daftengineer.github.io/media_assets/pollution.jpg');"
image-sm: https://daftengineer.github.io/media_assets/pollution.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">Environmental Protection Agency of United States conducts air pollution measurements of many areas in order to keep track on pollution level. In this project, I will be doing an explorative analysis of it. It will be in Spark REPL. There are 4 goals for the exploration which are as below:</p>
<p>1) Which state had highest pollution yearly?<br />
   2) Which cities were most polluted yearly?<br />
   3) Highest concentration of pollutant in the state.<br />
   4) Which observation method was most used yearly?<br /></p>
   <p style="text-align:justify;">The available data, we have, has around 75 lakh rows so it is huge dataset. Some of the data might not be sorted, this is because spark needs to shuffle data as per needs be in intermediate stages and sorting them afterward inceases overhead on workers. Schema of the data is given below. </p>
   <img src = "https://daftengineer.github.io/media_assets/ml4p1.jpg"/>
   <p style="text-align:justify;">So the first question, we want to explore is which state had highest pollution yearly. This seemingly simple question has tricky answer as there are many pollutants and they are measured using different units. So we need to find a way to make sure that we get the highest pollution yearly and we respect the existence of different units. And the answer I found was to have the average of individual pollutants within a state per year and find the euclidean distance of the vector formed using the pollutant from origin. For that I needed to write User Defined Aggregation Function and then using spark we got the highest polluted state yearly.</p>
   <img src = "https://daftengineer.github.io/media_assets/ml4p2.jpg"/>
   <img src = "https://daftengineer.github.io/media_assets/ml4p10.jpg"/>
   <p style="text-align:justify;">So the answer we got is given below:</p>
   <img src = "https://daftengineer.github.io/media_assets/ml4p3.jpg"/>
   <p style="text-align:justify;">We can apply exactly the same logic to find most polluted city areas.</p>
  <img src = "https://daftengineer.github.io/media_assets/ml4p4.jpg"/>
    <p style="text-align:justify;">Now, In order to find the maximum concentration of individual pollutants, we can not do euclidean distance. Here, we have to have the units set to a standard and then only we can compare. So, here I have converted all the units to microgram per meter cube. In order to convert from nanogram to microgram we just needed it to divide by 1000. But to convert parts per billion carbon to microgram we need to know which pollutant we are talking about, find its molar mass and then multiply it by constant and value in ppb. So after that we have the result like below.</p>
    <img src = "https://daftengineer.github.io/media_assets/ml4p5.jpg"/>
    <img src = "https://daftengineer.github.io/media_assets/ml4p6.jpg"/>
     <p style="text-align:justify;">Finding the most methods used yearly was easiest task as we did not require any of the preprocessing and just used basic spark tools to get to the result.</p>
     <img src = "https://daftengineer.github.io/media_assets/ml4p7.jpg"/>
      <img src = "https://daftengineer.github.io/media_assets/ml4p8.jpg"/>
     <img src = "https://daftengineer.github.io/media_assets/ml4p9.jpg"/>
      <p style="text-align:justify;">This was the brief exploration of dataset provided to us by EPA. There are many more exploration goal we can achieve from the same data but we will keep them for the future. Source code for this article can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/USPollution.scala">here</a></p>
</div>
