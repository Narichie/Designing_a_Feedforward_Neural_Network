# Designing a Feedforward Neural Network
## Introduction
Context: The dataset was created to predict admissions to the University of California, Los Angeles (UCLA), which will help students select universities based on their profiles. The predicted outcome will give students a fair indication of their
chances of acceptance.
## Objective: 
To build a classification model using neural networks to predict a student's chances of admission to UCLA.
## Observation: 
All libraries and datasets were loaded successfully, and the target variable is the “Chance of Admit.” To fulfill the classification task, I will convert the target variable into categorical using a .80% threshold. If the “chance of admit” is more than 80%, then Admit would be 1(i.e., yes); otherwise, it would be 0(i.e., no). Therefore, a new variable was created, “Admit;” the “Chance of Admit” column was removed, and the “serial No” was also removed since it would not add any value to the analysis.
### The first model (model_1)
Model_1 consists of one hidden layer with 16 neurons, which utilizes the ReLU activation function. This is followed by a second hidden layer containing 32 neurons, employing ReLU. 

model_1 = Sequential()

model_1.add(Dense(16, activation='relu', input_shape=(7,)))

model_1.add(Dense(32, activation='relu'))

model_1.add(Dense(1, activation='sigmoid'))

The output layer features a single neuron with a Sigmoid activation function. A learning rate of 0.001 was used, and the training was conducted over 100 epochs, with a 10% validation split and a batch size of 32. The model achieved an accuracy of 96% on the test dataset, with a loss of 0.1219. There were 705 parameters to estimate, and the complete training time was 3.70 seconds.
### The modify model (model_2)
The model_2 comprises a first hidden layer containing 64 neurons, followed by a second layer with 32 neurons, and then two additional layers, each with 64 neurons. The fifth and sixth layers comprise 256 neurons, all utilizing the ReLU activation function. The output layer features a
single neuron with a Sigmoid activation function.

model_2 = Sequential()

model_2.add(Dense(64, activation='relu', input_shape=(7,)))

model_2.add(Dense(32, activation='relu'))

model_2.add(Dense(64, activation='relu'))

model_2.add(Dense(64, activation='relu'))

model_2.add(Dense(256, activation='relu'))

model_2.add(Dense(256, activation='relu'))

model_2.add(Dense(1, activation='sigmoid'))

I used the Adam optimizer with a learning rate of 0.001. The training was conducted over 100 epochs, with a 10% validation split and a batch size of 32. The model achieved an accuracy of 99% on the test dataset, with a loss of 0.0696. There were 91553 parameters to estimate, and the complete training time was 4.10 seconds.

### The Modify model (model_3)
Model_3 comprises one hidden layer containing 8 neurons with the ReLU activation function and one output layer featuring a single neuron with a sigmoid activation function.

model_3 = Sequential()

model_3.add(Dense(8, activation='relu', input_shape=(7,)))

model_3.add(Dense(1, activation='sigmoid'))

I used a learning rate of 0.001. The training was conducted over 100 epochs, with a 10% validation split and a batch size of 32. The model achieved an accuracy of 95% on the test dataset, with a loss of 0.2909. There were 73 parameters to estimate, and the complete training time was 2.02 seconds.

## Conclusion:
Achieving an effective feedforward neural network requires careful consideration of several factors, including the balance between architectural complexity, performance, training time, and the number of parameters. This balance is crucial for accurately modeling complex patterns
across a variety of datasets. By optimizing these elements, one can enhance the network's ability to learn and generalize from diverse data inputs.
