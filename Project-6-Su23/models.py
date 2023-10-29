import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # Initialize batch size
        batch_size = 1
        while True:
            accurate_classification = True
            # Iterate through all data points and update our model if needed
            for x, y in dataset.iterate_once(batch_size):
                # Obtain the classification for one data point
                prediction = self.get_prediction(x)
                # Update parameter if wrong prediction
                if prediction != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    accurate_classification = False
            # End iteration if clear pass
            if accurate_classification:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Define a two-layer network with recommended hyperparameters
        self.w0 = nn.Parameter(1, 512)
        self.w1 = nn.Parameter(512, 1)
        self.b0 = nn.Parameter(1, 512)
        self.b1 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Run layer by layer as instructed in the project writeup.
        xw0 = nn.Linear(x, self.w0)
        h1 = nn.ReLU(nn.AddBias(xw0, self.b0))
        xw1 = nn.Linear(h1, self.w1)
        result = nn.AddBias(xw1, self.b1)
        return result

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Squared loss used as required
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Define recommended hyperparameters.
        batch_size = 200
        alpha = 0.05
        # Run until loss is less than 0.02 as instructed
        while True:
            for x, y in dataset.iterate_once(batch_size):
                # Obtain the gradients as each iteration for parameter updates
                loss = self.get_loss(x, y)
                grad_wrt_w0, grad_wrt_w1, grad_wrt_b0, grad_wrt_b1 = nn.gradients(loss, 
                    [self.w0, self.w1, self.b0, self.b1])
                # Update parameters as needed
                self.w0.update(grad_wrt_w0, -alpha)
                self.w1.update(grad_wrt_w1, -alpha)
                self.b0.update(grad_wrt_b0, -alpha)
                self.b1.update(grad_wrt_b1, -alpha)
            # Compute loss again to see if stop threshold is met
            loss = nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y)))
            # print(loss)
            if loss <= 0.02:
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Define a two-layer network with recommended hyperparameters
        self.w0 = nn.Parameter(784, 200)
        self.w1 = nn.Parameter(200, 10)
        self.b0 = nn.Parameter(1, 200)
        self.b1 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Run layer by layer as instructed in the project writeup.
        xw0 = nn.Linear(x, self.w0)
        h1 = nn.ReLU(nn.AddBias(xw0, self.b0))
        xw1 = nn.Linear(h1, self.w1)
        result = nn.AddBias(xw1, self.b1)
        return result


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Use softmaxLoss as required
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Define recommended hyperparameters.
        batch_size = 200
        alpha = 0.5
        # Run until accuracy is greater than 98.5% as instructed (added some buffer here)
        while True:
            for x, y in dataset.iterate_once(batch_size):
                # Obtain the gradients as each iteration for parameter updates
                loss = self.get_loss(x, y)
                grad_wrt_w0, grad_wrt_w1, grad_wrt_b0, grad_wrt_b1 = nn.gradients(loss, 
                    [self.w0, self.w1, self.b0, self.b1])
                # Update parameters as needed
                self.w0.update(grad_wrt_w0, -alpha)
                self.w1.update(grad_wrt_w1, -alpha)
                self.b0.update(grad_wrt_b0, -alpha)
                self.b1.update(grad_wrt_b1, -alpha)
            # Compute accuracy to see if stop threshold is met
            accuracy = dataset.get_validation_accuracy()
            # print(accuracy)
            if accuracy >= 0.98:
                break

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # Set the size of hidden layers to be 200 (sufficiently large maybe)
        # The output should have dimensionality of length of languages
        self.w0 = nn.Parameter(self.num_chars, 200)
        self.w1 = nn.Parameter(200, 200)
        self.w2 = nn.Parameter(200, len(self.languages))
        self.b0 = nn.Parameter(1, 200)
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, len(self.languages))

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Run the first input to initialize f and h respectively
        f_init = nn.Linear(xs[0], self.w0)
        h = nn.ReLU(nn.AddBias(f_init, self.b0))
        z = h
        # Iterate through the rest of inputs and compute the output of each layer 
        for char in xs[1:]:
            # First sub-network
            f = nn.Linear(z, self.w1)
            h = nn.ReLU(nn.AddBias(f, self.b1))
            # Second sub-network
            f_x = nn.Linear(char, self.w0)
            h_x = nn.ReLU(nn.AddBias(f_x, self.b0))
            z = nn.Add(h, h_x)
        # Final layer computation
        f_final = nn.Linear(z, self.w2)
        return nn.AddBias(f_final, self.b2)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        # Use softmax loss in this case
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # Define hyperparameters.
        batch_size = 200
        alpha = 0.1
        # Run until accuracy is greater than 87% as instructed (add some buffer here to pass autograder)
        while True:
            for x, y in dataset.iterate_once(batch_size):
                # Obtain the gradients as each iteration for parameter updates
                loss = self.get_loss(x, y)
                grad_wrt_w0, grad_wrt_w1, grad_wrt_w2, grad_wrt_b0, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, 
                    [self.w0, self.w1, self.w2, self.b0, self.b1, self.b2])
                # Update parameters as needed
                self.w0.update(grad_wrt_w0, -alpha)
                self.w1.update(grad_wrt_w1, -alpha)
                self.w2.update(grad_wrt_w2, -alpha)
                self.b0.update(grad_wrt_b0, -alpha)
                self.b1.update(grad_wrt_b1, -alpha)
                self.b2.update(grad_wrt_b2, -alpha)
            # Compute accuracy to see if stop threshold is met
            accuracy = dataset.get_validation_accuracy()
            # print(accuracy)
            if accuracy >= 0.87:
                break
