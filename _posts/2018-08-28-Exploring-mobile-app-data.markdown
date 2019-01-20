---
layout: post
title: "exploring mobile app data"
date: 2018-08-28
description: EXPLORATION OF MOBILE APP DATA AVAILABLE ON GOOGLE PLAY STORE APPS ML6
image: "background-image: url('https://daftengineer.github.io/media_assets/apps.jpg');"
image-sm: https://daftengineer.github.io/media_assets/apps.jpg
---

<div style="color:black;"><p></p>
<p style="text-align:justify;">In this article, I am going to use more of conventional tools in order to perform the explorative ananlysis on android apps data. These conventional tools include using python, pandas, numpy and matplotlib. The big disadvantage of using these tools is that we usually are not able to compute extremely large batches of data as these tools work on single system without any logic to distribute the data and algorithm. So these tools are good for doing analysis which can be done on a single machine. This means we usually are not able to use these analytical tools on distributed environments but that doesn't mean no one has been working on it.</p>
<p style="text-align:justify;">Source code for this article can be found <a href = "https://github.com/daftengineer/MachineLearningProjects/blob/master/Exploring_Android_App_Data.ipynb">here</a></p>
<p style="text-align:justify;">So, In this article, I will be exploring below questions:</p>
 <b> <ol>
  <li>Pricing Trends</li>
  <li>Number of Downloads Trends</li>
  <li>App Size Trends</li>
  <li>Most Popular and rated app monthly</li>
  <li>App with Highest Revenue</li>
  </ol></b>
 <h2>Pricing Trend</h2><br />
<p style="text-align:justify;">Pandas is best tool for us for this explorative task. First, I imported all the library which are required and then uploaded the CSV file into the notebook for analysis</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p1.jpg" />
<p style="text-align:justify;">Now, for finding pricing trends, we need to clean the date column as it has many typing mistakes. For that I wrote a function, which we can pass on every row to clean the date as required.</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p2.jpg" />
<p style="text-align:justify;">Then, I have grouped data and showed pricing trend (mean):</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p3.jpg" />
 <h2>Number of Downloads Trends</h2><br />
<p style="text-align:justify;">Using exactly the same logic I plotted the trend for Number of downloads. But point here to be noted, is that graph here, is on logarithmic axis so changes are actually more drastic than it seems.</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p4.jpg" />
<h2>App Size Trends</h2><br />
<p style="text-align:justify;">Here, size of app is in form of "2,3M" (2.3 Million) so we need to change it to float value from current string value using which we can work on it. So I wrote below function for it.</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p5.jpg" />
<p style="text-align:justify;">Then it becomes easy to plot it.</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p6.jpg" />
  <h2>Most Popular app Monthly</h2><br />
<p style="text-align:justify;">For this, I needed a definition for Popularity. So, in context of data we have available, we can define popularity as number of downloads times rating. This results in this type of app list but we had merged the data as groupby removes every other column which are not subject to grouping</p>
 <img src= "https://daftengineer.github.io/media_assets/ml6p7.jpg" />
 <img src= "https://daftengineer.github.io/media_assets/ml6p8.jpg" />
 <h2>App with Highest Revenue Monthly</h2><br />
<p style="text-align:justify;">Using the above logic we can also find apps with most revenue. Revenue, here, is defined as number of downloads times price. In earlier years, there werent monetization option on apps so I removed all those who might have earned from advertisement but not directly from play store.</p>
<img src= "https://daftengineer.github.io/media_assets/ml6p9.jpg" />
 </div>

