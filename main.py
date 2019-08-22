import numpy
import scipy.special

#TEST TEST TEST TEST

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.5):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.input_to_hidden_weights = numpy.random.normal(0.0, pow(self.input_nodes, -0.5),
                                                        (self.hidden_nodes,self.input_nodes))
        self.hidden_to_output_weights = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5),
                                                         (self.output_nodes, self.hidden_nodes))

        self.activationFunction = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate sigansl into hidden layer
        hidden_inputs = numpy.dot(self.input_to_hidden_weights, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.hidden_to_output_weights, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activationFunction(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.hidden_to_output_weights.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.hidden_to_output_weights += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0-final_outputs),
                                                                         numpy.transpose(hidden_outputs)))
        # update the weights for the links between the input and hidden layers
        self.input_to_hidden_weights += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),
                                                                       numpy.transpose(inputs))

    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layers
        hidden_inputs = numpy.dot(self.input_to_hidden_weights, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activationFunction(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.hidden_to_output_weights, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activationFunction(final_inputs)

        return final_outputs


def main():
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.5

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(network.query([1.0, 0.5, -1.5]))


if __name__ == '__main__':
    main()
    quit()