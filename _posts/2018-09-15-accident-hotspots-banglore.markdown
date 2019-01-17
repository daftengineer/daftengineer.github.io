---
layout: post
title: "finding accident hotspots in bangalore with clustering algorithm"
date: 2018-09-15
description: With the use of Clustering Algorithm, finding accident hotspots in bangalore.
image: "background-image: url('https://daftengineer.github.io/media_assets/accident.jpg');"
image-sm: https://daftengineer.github.io/media_assets/accident.jpg
---

<div style="color:black;"><p></p>

 <p style="text-align:justify;">In this project, I will try to find out concentration of accident in bangalore. </p>
 <p style="text-align:justify;">Finding an accident hotspot is really important task for any urban planner. It provides us with information that, which area has unusal amount of traffic accidents. I have sensor data available with me which has speed and gps location of accident. Using that I found 36 places where there is unusal number of traffic accidents.</p>
 <p style="text-align:justify;">It is basically a clustering problem. Here accidents tend to cluster in some location and those needs to be in focus of attention. For this task, I have used "Density-based Spatial Clustering of Applications with Noise" (DBSCAN) Algorithm. The only reason, I used this algorithm is that, it is scalable and K-Means is too common (Still widely used though).</p>
 <p style="text-align:justify;">Source code for this article can be found <a href="https://github.com/daftengineer/MachineLearningProjects/blob/master/Accidents_in_Bangalore.ipynb">here</a></p>
 <p style="text-align:justify;">First of all, we will import all the library which are required and upload the required csv file.</p>
 <img src="https://daftengineer.github.io/media_assets/ml8p1.jpg" />
 <p style="text-align:justify;">Here is the preview of file. And after that, I have converted all data in dataframe to numpy array so we can use it in scikit-learn model.</p>
 <img src="https://daftengineer.github.io/media_assets/ml8p2.jpg" />
 <p style="text-align:justify;">Now, we will define and fit the DBSCAN model. Here value of epsilon is taken on trial and error basis. There are different techniques to find the optimal value but here, in our usecase, we wont be requiring them. Same is true for minimum sample size. After the first line, all logic is for ploting the data which has been done using matplotlib.</p>
  <img src="https://daftengineer.github.io/media_assets/ml8p3.jpg" />

 <p style="text-align:justify;">Finally the plot looks like this:</p>
  <img src="https://daftengineer.github.io/media_assets/ml8p4.jpg" />

 <p style="text-align:justify;">Here is one location for each hotspot. Others will be at maximum of epsilon distance from this location.</p>
  <img src="https://daftengineer.github.io/media_assets/ml8p5.jpg" />
 </div>

