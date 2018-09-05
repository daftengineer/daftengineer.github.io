---
layout: post
title: "EXPLORATION OF USA POLLUTION DATA ML4"
date: 2018-08-31
description: Exploring the USA pollution data provided to the public by EPA
image: "background-image: url('https://daftengineer.github.io/media_assets/pollution.jpg');"
image-sm: https://daftengineer.github.io/media_assets/pollution.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">Environmental Protection Agency of United States conducts air pollution measurements of many areas in order to keep track on pollution level. In this project, I will be doing an explorative analysis of it. It will be in Spark REPL. There are 5 goals for the exploration which are as below:</p>
<p>1) Which state had highest pollution yearly?<br />
   2) Which cities were most polluted yearly?<br />
   3) Highest concentration of pollutant in the state.<br />
   4) Which observation method was most used yearly?<br /></p>
   <p style="text-align:justify;">The available data, we have, has around 75 lakh rows so it is huge dataset. Schema of the data is given below. </p>
   <img />
   <p style="text-align:justify;">So the first question we want to explore is which state had highest pollution yearly. This seemingly simple question has tricky answer as there are many pollutants and they are measured using different units. So we need to find a way to make sure that we get the highest pollution yearly and we respect the existence of different units. ANd the answer i found was to have the average of individual pollution within a state per year and find the euclidean distance of the vector formed using the pollutant from origin. For that i needed to write User Defined Aggregation Function. And then using spark we got the highest polluted state yearly.</p>
   <img />
   <img />
   <p style="text-align:justify;">So the answer we got is given below:</p>
   
   <p style="text-align:justify;"></p>
</div>
