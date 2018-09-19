---
layout: post
title: "finding accident hotspots in bangalore ml8"
date: 2018-09-15
description: exploring different crimes in india using data publicly available about it.
image: "background-image: url('https://daftengineer.github.io/media_assets/accident.jpg');"
image-sm: https://daftengineer.github.io/media_assets/accident.jpg
---

<div style="color:black;"><p></p>

 <p style="text-align:justify;">In this project, I will try to find out concentration of accident in banglore. </p>
 <p style="text-align:justify;">Finding an accident hotspot is really important task for any urban planner. It provides us with information that which are has unusal amount of traffic accidents. I have sensor data available with me which measures speed and gps location of accident. Using that I found 36 places where there is unusal number of traffic accidents.</p>
 <p style="text-align:justify;">Finding accident hotspot is basically a clustering problem. Here accidents tend to cluster in some location and those needs to be in focus of attention. For this task, I have use "Density-based Spatial Clustering of Applications with Noise" (DBSCAN) Algorithm. The only reason I used this algorithm is that it is scalable and K-Means is too common (Still widely used).</p>
 <p style="text-align:justify;">Source code for this article can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/Accidents_in_Bangalore.ipynb">here</a></p>
 <p style="text-align:justify;"><h1>-></h1>First of all we will import all the library which are required and upload the required csv file.</p>
 <img src="https://daftengineer.github.io/media_assets/ml8p1.jpg" />
 <p style="text-align:justify;"><h1>-></h1>Here is the preview of file. And after that I have converted all data in dataframe to numpy array so we can use it in scikit learn model.</p>
 <img src="https://daftengineer.github.io/media_assets/ml8p2.jpg" />
 <p style="text-align:justify;"><h1>-></h1>Now we will define and fit the dbscan model. Here value of epsilon is taken using trial and error basis. There are different techniques to find the optimal value but here in our usecase we wont be requiring them. Same is true for minimum sample size. After the first line, all logic is for ploting the data which has been done using matplotlib.</p>
  <img src="https://daftengineer.github.io/media_assets/ml8p3.jpg" />

 <p style="text-align:justify;"><h1>-></h1>Finally the plot looks like this:</p>
  <img src="https://daftengineer.github.io/media_assets/ml8p4.jpg" />

 <p style="text-align:justify;"><h1>-></h1>Here are one location for each hotspot. Others will be at maximum of epsilon distance from this location.</p>
  <img src="https://daftengineer.github.io/media_assets/ml8p5.jpg" />
 </div>

