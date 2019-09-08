import numpy
import scipy.special
import matplotlib.pyplot


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
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.5

    network = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    training_data_file = open("mnist_dataset/mnist_train_100.csv",'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    for record in training_data_list:
        # split values by commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        network.train(inputs, targets)
        pass


if __name__ == '__main__':
    main()
    quit()