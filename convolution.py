from layer import Layer
import numpy as np
import math


#get the windows/slices for the convolution using as_strided

def getWindows(input, output_size, kernel_size, padding=0, stride=1, dilate=0):
    working_input = input
    working_pad = padding
    # dilate the input if necessary
    if dilate != 0:
        working_input = np.insert(working_input, range(1, input.shape[2]), 0, axis=2)
        working_input = np.insert(working_input, range(1, input.shape[3]), 0, axis=3)

    # pad the input if necessary
    if working_pad != 0:
        working_input = np.pad(working_input, pad_width=((0,), (0,), (working_pad,), (working_pad,)), mode='constant', constant_values=(0.,))

    in_b, in_c, out_h, out_w = output_size
    out_b, out_c, _, _ = input.shape
    batch_str, channel_str, kern_h_str, kern_w_str = working_input.strides

    return np.lib.stride_tricks.as_strided(
        working_input,
        (out_b, out_c, out_h, out_w, kernel_size, kernel_size),
        (batch_str, channel_str, stride * kern_h_str, stride * kern_w_str, kern_h_str, kern_w_str)
    )

class Convolution(Layer):
    def __init__(self, input_channel = 3, kernel_size = 3, no_of_filters = 5, stride = 1, padding = 0, f = None):
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.no_of_filters = no_of_filters
        self.stride = stride
        self.padding = padding
        self.W = None
        self.b = None
        self.dw = None
        self.db = None
        self.cache = None
        self.init_weights()

        self.f = f



    def init_weights(self):
        self.W = np.random.randn(self.no_of_filters, self.input_channel, self.kernel_size, self.kernel_size) * math.sqrt(2.0 / (self.input_channel * self.kernel_size * self.kernel_size))
        self.b = np.random.randn(self.no_of_filters,)


    
    def forward(self, a, train = True):
        batch_size, input_channel, input_height, input_width = a.shape
        output_height = (input_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_width = (input_width - self.kernel_size + 2 * self.padding) // self.stride + 1


        windows = getWindows(a, (batch_size, self.no_of_filters, output_height, output_width), self.kernel_size, self.padding, self.stride)


        out =  np.einsum('bihwkl,oikl->bohw', windows, self.W)

        out += self.b[None, :, None, None]

        self.cache = a, windows

        self.print_params(self.f)

        return out


    def backward(self, dz):

        a, windows = self.cache

        #get how much to pad the input

        padding = self.kernel_size - 1 if self.padding == 0 else self.padding

        dz_windows = getWindows(dz, a.shape, self.kernel_size, padding=padding, stride=1, dilate=self.stride - 1)
        rotted_kernel = np.rot90(self.W, 2, (2, 3))

        self.db = np.sum(dz, axis=(0, 2, 3))
        self.dw = np.einsum('bihwkl,bohw->oikl', windows, dz)
        da = np.einsum('bohwkl,oikl->bihw', dz_windows, rotted_kernel)


        self.print_params(self.f)

        return da


    def update(self, learning_rate):
        self.W -= learning_rate * self.dw
        self.b -= learning_rate * self.db


    def __str__(self):
        return "Convolution Layer"


    def print_params(self,f):
        f.write("W: " + str(self.W) + "\n\n")
        f.write("b: " + str(self.b) + "\n\n")

        if self.dw is not None:
            f.write("dw: " + str(self.dw) + "\n\n")
            f.write("db: " + str(self.db) + "\n\n")



#test the code
if __name__ == "__main__":
    conv1 = Convolution(3,3,5,1,1)

    #make a random input of size shape (3,16,3,28,28)


    a = np.random.randn(16,3,28,28)

    #forward pass

    z = conv1.forward(a)

    print("z shape: ", z.shape)

    #backward pass

    #make a random dz of the same shape as z

    dz = np.random.randn(16,5,28,28)

    da = conv1.backward(dz)
    conv1.update(0.01)

    print("da shape: ", da.shape)

    f = open("convolution.txt", "w")
    conv1.print_params(f)












        