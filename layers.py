import numpy as np

class Linear:
    def __init__(self, in_feat, out_feat, name='linear'):
        self.name = name

        self.lim = np.sqrt(6.0 / (in_feat + out_feat + 1.0))

        self.weights = np.random.uniform(low= -self.lim, high=self.lim, size=(in_feat, out_feat))
        self.bias = np.random.uniform(low= -self.lim, high=self.lim, size=out_feat)

        self.trainable = True

        self.parameters = {'weights': self.weights, 'bias': self.bias}

    def __call__(self, inputs):
        return np.matmul(inputs, self.weights) + self.bias

    def backward(self, inputs, grad_loss, learning_rate):
        new_grad_loss = np.dot(grad_loss, self.weights.transpose())

        dw = np.dot(inputs.transpose(), grad_loss).reshape(self.weights.shape)
        db = (grad_loss.mean(axis=0) * inputs.shape[0]).reshape(self.bias.shape)

        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db

        return new_grad_loss

class ReLU:
	def __init__(self, name='relu'):
		self.name = name
		self.trainable = False

	def __call__(self, inputs):
		return np.maximum(0, inputs)

	def backward(self, inputs, grad_out, learning_rate):
		out = inputs > 0
		return grad_out * out

class LeakyReLU:
    def __init__(self, alpha=0.2, name='leaky_relu'):
        self.alpha = alpha
        self.name = name

        self.trainable = False
    
    def __call__(self, inputs):
        return np.maximum(inputs, inputs * self.alpha)

    def backward(self, inputs, grad_loss, learning_rate):
        out = np.ones_like(inputs)
        out[inputs<=0] = self.alpha
        return grad_loss * out