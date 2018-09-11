---
layout: post
title: "explanation of deep learning using tea"
date: 2018-07-02
description: Analogy for DEEP LEARNING with Tea 
image: "background-image:url('https://daftengineer.github.io/media_assets/2.jpg');"
image-sm: 
---

<div style="color:black;">
  
  <p></p>
  <p style="text-align:justify;">
    Deep Learning is most consequential algorithm of 21<sup>st</sup> century, there is no doubt about it but majority of newcomers in Machine learning field, find it really hard to grasp the concept of it. Here is just a small post in which I would like to explain the basics of deep learning and little bit of the detail about future of it, all with the example of making a tea. This might be alot more intuitive than a typical architectural example used by most of books out in market.
  </p>
  <img src="https://daftengineer.github.io/media_assets/process.jpg" style="" />
  <p style="text-align:justify;">
    Here I have oversimplified the concept but the gist of it is accurate.
  </p>
  <p style="text-align:justify;">
    We do optimisation in every decision of our life. In layman's term optimisation is trade off. We do trade off between work or life, spending or saving, present benefits vs future benefits etc. Now, to get the best results, we need to find the balance among all these. In a nutshell, all the machine learning algorithms are optimising the result we have got to the result we want. Now we are clear on what optimisation is and what the goal of machine learning algorithm is, we can start with deep learning.
  </p>
  <p style="text-align:justify;">
    Imagine we are making world's best tea and we are given whole kitchen, full of all ingredients, whether we need them or not for making tea. Now imagine, Person who is making tea doesn't know what all ingredients are for and how much to use them. Only way to know the ingredients is by tasting the tea made by him. Now chef would mix random things and give it to us and we will say whether it was good or bad. We have list of things, we want from a tea to have in order to consider it the best. those things are:
  </p>
  <ul>
    <li>Color</li>
    <li>Taste</li>
    <li>Consistency</li>
    <li>Temperature</li>
  </ul>
 <p style="text-align:justify;">
   We will judge the tea with these matrix and tell chef what he needs to work on. So chef goes back to kitchen considers what we told him and tries to recall what he added and what he needs to do in order to make those specific changes to make good tea. Now if this thing happens thousands of time eventually we will get to better tea we want. In deep learning we exactly do this, we input the data and compare the output with ideal source, this is called training phase. Now to make appropriate changes we need to use optimisation algorithm like <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent">stochastic gradient descent</a>. In tea making process, what chef did was to recall what he added and made changes, these changes are done using optimisation algorithm and the process of recalling the ingredient is called <a href="https://en.wikipedia.org/wiki/Backpropagation">backpropogation</a>. We do not know what chef has used or what process he took to make it but we only know what a tea tastes or looks like. The fact that process is hidden to us is known as hidden layer in deep learning. The process of comparing the end result (tea) with accpetable version of tea, is called <a href="https://en.wikipedia.org/wiki/Loss_function">Cost Function</a>. Comparing the current tea with the best tea ever is called objective function.
  </p>
  <p style="text-align:justify;">
    Many a times, It might happen that chef forgets overtime what he added on first go. It happens with deep learning algorithms too, it is called <a href="https://en.wikipedia.org/wiki/Vanishing_gradient_problem">vanishing gradient</a>. So what he does, is to make batches of sequence of tries and writes it on paper so that he doesn't forget it. In deep learning terms, the batches are called epochs and writing on paper is spilling to nonvolatile memory. Now if we repeat this process thousands of time on hundreds of epochs we will get best tea that particular chef can make. Here we didn't use all the ingredients available in kitchen because we didn't need everyone of them. The fact that we used specific ingredients is called <a href="https://en.wikipedia.org/wiki/Activation_function">activation</a> of neurons in deep learning. This is a typical working of <a href="https://en.wikipedia.org/wiki/Convolutional_neural_network">Convolutional neural network</a>.
  </p>
  <p style="text-align:justify;">
    We can intuitively say that this is not the best way to make tea, we need alot of trial and error as we don't know the structure to follow while making a tea. Working of CNN is like making a tea with ingredients in it put without any order like chef added milk, sugar, ginger, first and when it came to boil then he added tea-leaves. This way of optimising the tea would definitely get you better tea but it might not be the best way. We might need to analyse the process of tea making and its order. This idea is called <a href="https://en.wikipedia.org/wiki/Capsule_neural_network">capsule networks</a> in deep learning. 
  </p>
  <p style="text-align:justify;">
    Here we barely scratched the surface of what is done using deep learning but it will definitely give you basic idea about how deep learning algorithms work and how they can be used in real life problems.
  </p>
  
</div>
