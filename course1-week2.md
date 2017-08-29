


## What We Will Cover
* Vecotrization:  When working with neural networks and lots of data, you probably do not want to use
a for loop to go over each example in your training set during the training phase.
There are better ways -- vecotrized ways!
* Forward and Backward Propagation

These ideas will be conveyed using logistic regression, which is a typical way of introducing
neural networks and how to optmize them.  

## Logistic Regression
LogReg is a type of binary classifier, e.g., {cat, notCat}.

QUESTION:  If you have an image of cats and you want to classify whether or not the image has cats in it,
what do you consider your features?
  
In a 32x32 B&W image, one has 16x16 = 256 pixels, ranging in blackness from 0 to 255.  Each pixel
can be considered an input feature.  A single pixel's value likely has little to say about whether or
not a cat is in the picture---but consider a small neighborhood of pixels, e.g., a patch of pixels big
enough to capture a cat's ear.  Each pixel by itself hold very little information, but taken in the context
of its neighbors holds quite a bit (in a way, the whole is greater than the sum of its parts). 

As a human, we know that pixels are best considered in context, which is why convolutional neural networks
shine in the image recognition context.  However, context can be derived even with simple densely connected
neural networks.  In this setting, the computer is given a 256-dimensional vector.  Each pixel is presumed
to be independent of the other pixels.  But during the training phase, each node (decision maker) in the network
will begin to learn some context --- that certain pixels matter more than others to a particular decision being 
made.  

The same is true of color/RGB images.  An 16x16 RGB image has 3 16x16 layers instead of one, and so has
3x16x16 = 768 pixel features.  Again, though not the best method, a densely-connected NN can still learn
quite a bit about classification indicators in color/pixel neighborhoods for binary classification on small images.
Of course, as the image size grows in pixel area, this becomes more and more difficult to realize using
this approach.

---------------------------

## Notation
* Data Set has points (x,y), where x inhabits the feature space and y inhabits the target space
* Let m denote the number of data points
  - m_training is number of training points
  - m_test is number of test points
  - etc
* Feature Vector:  x in R^Nx
  - actually, this is notational convenience
  - e.g., when talking about B&W images, if U := {0, ..., 255} < Z, then x is in U^Nx
* Feature Matrix: X in (R^Nx)x(R^m)
  - X is a "horizontal stack" of feature vectors
  - NOTE:  this feature matrix has **Columnar Records** and each row is a feature, whereas in TensorFlow and the Udacity course, the data matrix has row records and each feature is a column
    * Ng says the columnar record approach makes implementing neural networks much easier
  - note that this changes the equations
    * Row Records (TF, Udacity):  xW + b --> y
    * Column Records (Ng):   Mu + d --> z 
    * CONVERSION:   (xW+b)^T = (xW)^T + b^T = (W^T)(x^T) + b^T = Mu + d
  - also note that in other classes, X[i,j] denotes the jth feature of the ith data point
    * in this class, X[i,j] denotes the ith feature of the jth data point
  - however, when using only one index, the notation is the same for both col/row record conventions:  X[i] stands for the ith data point
* Target Vector: y in {0,1}
* Target Matrix:  Y = [y[1], ..., y[m]]
  - in this course, Y is a "horizontal stack" of target vectors, whereas in the Udacity/TF course it was a vertical stack
  - since the target vectors in this lecture are 1-dimensional, the target matrix is 1xm

Just keep the convention in mind when programming since things like the components of input.shape
take on different meanings in the two conventions:
* Row Records:  input.shape = (num_rows, num_features)
* Col Records:  input.shape = (num_features, num_rows)

----------------------------------

## Logistic Regression (Theory)
* Given x, we want to know p = P(y=1|x), where p is an estimate of the probability that y=1 given x.
* Given that x is a feature vector from R^Nx, the parameter vector we want to learn will inhabit R^(Nx+1)
  - one weight coefficient for each feature, and one bias term
* In linear regression, we would have p = (w^T)x + b, but this cannot be used as an estimate of probability
* In logistic regression, we instead have p = sigmoid((w^T)x + b)

![](./images/sigmoid-fcn.png)

What does the sigmoid function do for us in terms of decision making?

If you look at the sigmoid equation, it says that large positive z values should indicate a "yes" (~1), 
wherease large negative z values should likely indicate a "no" (~0).  So in a regular logistic regression,
we have one node making the decision; this decision is made by learning the "best assumptions," i.e., the
values for w and b that will help the node estimate the probability.

Reminder:
* X[i,j] in this course stands for the ith feature of the jth data point
* if only one index is referenced, X[i] stands for the ith data point

## The LogReg Loss Function
### MSE
You can use the mean-squared error (MSE), however this produces a nonconvex optmization problem in the 
context of logistic regression.  That is, a detected minimum loss over the parameter space is not necessarily
the global minimum (gradient descent will not necessarily work well).  

In the follwing image "y hat" is the probability estimate (what I refer to as p above).

<img src=./images/mse.png width=200>

In complex neural networks, one might have to deal with this possibility.  However, in
a simple network like logistic regression, one can identify a loss function that does make for a convex
optimization problem (the detected minimum is the global minimum).

### Cross Entropy
In the follwing image "y hat" is the probability estimate (what I refer to as p above).
<img src=./images/cross-entropy.png width=200>

If you look at the equation, the prediction ("y hat") and the actual ("y") have separate roles.  
* the actual's role is to determine which term in the sum should be considered
* the prediction's role is to be compared to how close it is what its supposed to be

This observation tells us that this loss function is not a distance metric: it is not symmetric in its input arguments.

If you think about entropy from physics, this is kind of like saying that you can increasingly
add order to a partially-disordered system, but can never again add order to a fully-disordered system.
In terms of information, it is like saying that a partially-ignorant can always strive towards
minimizing its ignorance (e.g., using deduction, past experience, etc), but that a fully-ignorant agent
will always be fully ignorant (e.g., has no knowledge of deductive reasoning, has no memory of past experience, has
no ability to link two bits of information, etc).  Any of that help?  

Another thought: the role played by the actual is that of perfect information. For simplicity,
consider a parabola z = w^2.  Getting to z=0 can only be done in one way -- continuously head
towards zero.  That is, to achieve a state of perfect information, continously reduce your
ignorance.  However, to get to any other state, z=a, at its simplest one must choose one of
two directions to proceed from perfection.  That is, getting to imperfection from perfection
is not well-defined like it is to get to perfection from imperfection!  

Thinking this through, one can actually realize that the roles of the prediction and the actual
in symmetric loss functions are different as well, but the symmetry hides this insight a bit.

## The LogReg Cost Function
Andrew Ng follow the convention I typically like, though it is not universal.  The loss function
refers to how you measure the error at a single data point, while the cost function refers to
how you measure the error over a group of data points (sometimes the entire data set, but more freqently
in "mini-batches" of the data set).

<img src=./images/cost-fcn.png width=200>

In physics speak, this is somewhat similar to defining a Lagrangian, then defining the action of that Lagrangian, and
finally minimizing the action to solve for the equations of motion.


