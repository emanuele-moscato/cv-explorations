from tensorflow.keras.layers import Conv2D, MaxPool2D, Layer


class VGGLayer(Layer):
    """
    Implementation of a VGG-16 layer (as a Keras `Layer` object), as described
    at
        https://medium.com/@siddheshb008/vgg-net-architecture-explained-71179310050f
    """
    def __init__(self, config):
        """
        Class constructor. Blocks are defined as lists of lists of layers
        containing the same number of convolutional layers and each
        terminating with a maxpooling operation. The config dictates the
        number of filters for the convolutional layers in each block.
        """
        # Call the parent class' constructor.
        super().__init__()

        # Define all the blocks of convolutional layers.
        self.conv_blocks = [
            [
                Conv2D(
                    filters=n_filters,
                    kernel_size=3,
                    activation='relu',
                    # padding='same'
                )
                for n_filters in n_filters_list
            ]
            for n_filters_list in config['n_filters_conv_blocks']
        ]


        self.maxpool = MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid'
        )


    def call(self, x):
        """
        Forward pass of the model.
        """
        # Loop through the blocks.
        for block in self.conv_blocks:
            # Loop through each convolutional layer within the block.
            for layer in block:
                x = layer(x)

            # Apply maxpooling at the end of each block.
            x = self.maxpool(x)

        return x
