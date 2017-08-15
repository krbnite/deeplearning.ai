

Electricity transformed countless industries: AI will bring about an equally-powerful transformation

## Specialization
* Course 1:  Neural Networks and Deep Learning 
  - Covers fundamentals/foundations
  - Learn to build/train deep NNs
  - Will build a cat classifier
* Course 2: Improving DNNs
  - Hyperparameter tuning
  - Regularization
  - Optimization (Adam, RMSProp, etc)
* Course 3: Structuring your Machine Learning Project
  - best practices
  - dealing w/ differences in data distributions inherent in split data sets
  - end-to-end deep learning: should you use it?
* Course 4:  Convolutional Neural Networks (CNNs)
* Course 5:  NLP / Sequence Models
  - RNN, LSTM, etc
  

## What is a Neural Network?
* DL just refers to building "deep" neural networks

### Housing Price Prediction
* Simple Neural Network
  - if your intuition says, we can just apply a linReg, then you would be right
    * in a way, you just built your first neural network (w/ an identity activation)
  - but can get fancier, e.g., we know price can never be negative
  - so, for example, can define any house w/ size < s to have price 0
  - this is a simple neural network: it has one neuron, N = xLA
    * that is, the univariate input layer, x, is linearly transformed, z=xL, and activated, y=zA
  
* Relu (REctified Linear Units)
  - the threshold on house size is similar to a step function
    * step(x-s) = {0, x-s < 0; 1, x-s >= 0}
  - in fact, it's like we multiplied the linear equation by a step function
    * f(x) = step(x-s)*line(x)
  - this can be rewritten as a relu function
    * f(x) = max(0, line(x))
  
 * Wider Neural Networks
  - the simplest increase in complexity is to use a wider input layer
  - we can also widen that second layer: just stack neurons like the one we built 
    * 
    * but this would be a vectorial output, so we can add one additional layer -- call it the output layer
    * in the image, Ng shows examples in the hidden layer, but that is to just give you an idea of how the network might combine variables to extract relevant info
      - the hidden layer is determined by the network coefficients which is optimized during training
    * we say that the hidden layer is fully/densely connected b/c each neuron is connected to every input feature 
    
    
 
 