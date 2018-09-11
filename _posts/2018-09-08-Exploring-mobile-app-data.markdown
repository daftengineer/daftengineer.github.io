---
layout: post
title: "EXPLORING MOBILE APP DATA ML6"
date: 2018-09-08
description: EXPLORATION OF MOBILE APP DATA AVAILABLE ON GOOGLE PLAY STORE APPS ML6
image: "background-image: url('https://daftengineer.github.io/media_assets/apps.jpg');"
image-sm: https://daftengineer.github.io/media_assets/apps.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">In this article, I am going to use more of conventional tools in order to perform the explorative ananlysis on android apps data. This conventional tools include using python, pandas, numpy and matplotlib. The big disadvantage of using these tools is that we usually are not able to compute extremely large batches of data as these tools work on single system without any logic to distribute the data and algorithm. So this tools are good for doing analysis which can be done on a single machine. This means we usually are not able to use this analytical tools on distributed environments but that doesnt mean no one has been working on it. Even Apache Spark is available on python language but that is for totally different use case.</p>
<p style="text-align:justify;">Source code for this article can be found <a href = "https://github.com/daftengineer/MachineLearningProjects/blob/master/Exploring_Android_App_Data.ipynb">here</a></p>
<p style="text-align:justify;">So, In this article, I will be exploring below questions:</p>
 <b> <ol>
  <li>Pricing Trends</li>
  <li>Number of Downloads Trends</li>
  <li>App Size Trends</li>
  <li>Most Popular and rated app monthly</li>
  <li>App with Highest Revenue</li>
  </ol></b>
 <h1>Pricing Trend</h1>
<p style="text-align:justify;">Using pandas is best tool for us for this explorative task. First I imported all the library which are required. And then upload the CSV file into the notebook for Analysis</p>
<p style="text-align:justify;">Now, for finding pricing trends we need to clean the date column as it has many typos. For that i wrote a function which we can pass on every row to clean the date as required.</p>
<p style="text-align:justify;">Then, I grouped data and showed pricing trend (mean):</p>
 <h1>Number of Downloads Trends<h1>
<p style="text-align:justify;">Using exactly the same logic I plotted the trend for Number of downloads. But point here to be noted is that graph here is on logarithmic axis so changes are actually more drastic than it seems.</p>
<h1>App Size Trends</h1>
<p style="text-align:justify;">Here, size of app is in form of "2,3M" (2.3 Million) so we need to change it to float value from current string value using which we can work on it. So I wrote below function for it.</p>
<p style="text-align:justify;">Then it is became easy to plot it.</p>
  <h1>Most Popular app Monthly</h1>
<p style="text-align:justify;">For this I needed a definition for Popularity. So in context of data we have available we can define popularity as number of downloads times rating. This results in this kinda app list but we had merge the data as groupby removes every other column which are not subject to grouping</p>
<p style="text-align:justify;"></p>

 </div>

