import torch
import torch.nn as nn
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: list,
        pool_every: int,
        hidden_dims: list,
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params
        self.class_in_h = 0,
        self.class_in_w = 0

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Use only dimension-preserving 3x3 convolutions.
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        channels = self.channels

        # For dimension calculations
        self.class_in_h, self.class_in_w = in_h, in_w
        padding = 0
        if type(self.conv_params) is dict and 'padding' in self.conv_params.keys():
            padding = self.conv_params['padding']

        stride = 1
        if type(self.conv_params) is dict and 'stride' in self.conv_params.keys():
            stride = self.conv_params['stride']

        kernel_size = 3
        if type(self.conv_params) is dict and 'kernel_size' in self.conv_params.keys():
            kernel_size = self.conv_params['kernel_size']

        pool_kernel_size = 2
        if type(self.pooling_params) is dict and 'kernel_size' in self.pooling_params.keys():
            pool_kernel_size = self.pooling_params['kernel_size']

        lrelu_slope = 0
        if type(self.activation_params) is dict and 'negative_slope' in self.pooling_params.keys():
            lrelu_slope = self.pooling_params['negative_slope']

        # Iterating over feature channels
        for i in range(len(channels)):

            # calculate class input dimensions
            self.class_in_h = int((self.class_in_h + 2 * padding - kernel_size)/stride) + 1
            self.class_in_w = int((self.class_in_w + 2 * padding - kernel_size)/stride) + 1

            layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels[i], kernel_size=kernel_size, 
            padding=padding, stride=stride))
            # activation type
            if self.activation_type == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(**self.activation_params))
            # pool every P times
            if (i+1) % self.pool_every == 0:

                # Calculating dims with regards to pooling
                self.class_in_h = int((self.class_in_h - pool_kernel_size)/pool_kernel_size) + 1
                self.class_in_w = int((self.class_in_w - pool_kernel_size)/pool_kernel_size) + 1

                if(self.pooling_type == "max"):
                    layers.append(nn.MaxPool2d(pool_kernel_size))
                else:
                    layers.append(nn.AvgPool2d(pool_kernel_size))

            

            in_channels = channels[i]
        # raise NotImplementedError()
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        num_of_features = self.channels[-1]*self.class_in_h*self.class_in_w
        for dim in self.hidden_dims:

            layers.append(nn.Linear(num_of_features, dim))
            num_of_features = dim
            if self.activation_type == "relu":
                layers.append(nn.ReLU())
            else:
                layers.append(nn.LeakyReLU(**self.activation_params))
        layers.append(nn.Linear(num_of_features, self.out_classes))
        # raise NotImplementedError()
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)
        out = self.classifier(features)
        # raise NotImplementedError()
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: list,
        kernel_sizes: list,
        batchnorm=False,
        dropout=0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        main_layers = []
        # dims = []
        # in_w = 0
        # in_h = 0

        curr_in_channels = in_channels

        # block layers
        for i, channel in enumerate(channels[:-1]):

            main_layers.append(nn.Conv2d(in_channels=curr_in_channels, out_channels=channel,
                                         kernel_size=kernel_sizes[i], padding=int((kernel_sizes[i]-1)/2)))
            if dropout > 0:
                main_layers.append(nn.Dropout2d(dropout))

            if batchnorm:
                main_layers.append(nn.BatchNorm2d(channel))
            if activation_type == "relu":
                main_layers.append(nn.ReLU())
            else:
                main_layers.append(nn.LeakyReLU(**activation_params))

            curr_in_channels = channel

        main_layers.append(nn.Conv2d(in_channels=curr_in_channels, out_channels=channels[-1],
                                     kernel_size=kernel_sizes[-1], padding=int((kernel_sizes[-1] - 1) / 2)))

        shortcut = nn.Sequential()
        # skip connection using 1x1 kernel
        if in_channels != channels[-1]:
            shortcut = nn.Sequential(nn.Conv2d(in_channels, channels[-1], kernel_size=1, bias=False))


        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = shortcut
        # raise NotImplementedError()
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        pool_func = POOLINGS[self.pooling_type]
        channels = self.channels
        self.class_in_h = in_h
        self.class_in_w = in_w
        pool_size = self.pooling_params["kernel_size"]
        pool_pad = self.pooling_params.get("padding", 0)
        pool_stride = self.pooling_params.get("stride", pool_size)
        

        for i in range(int(len(channels)/self.pool_every)):

            curr_channels = channels[i*self.pool_every:(i+1)*self.pool_every]

            layers.append(ResidualBlock(in_channels = in_channels,
                                        channels = curr_channels, kernel_sizes = [3]*len(curr_channels),
                                        batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, 
                                        activation_params = self.activation_params))

            layers.append(pool_func(**self.pooling_params))

            in_channels = channels[(i+1)*self.pool_every - 1]

            # Classifier diminesions
            self.class_in_h = (self.class_in_h-pool_size+2*pool_pad)//pool_stride + 1
            self.class_in_w = (self.class_in_w-pool_size+2*pool_pad)//pool_stride + 1

        if len(channels)%self.pool_every > 0:

            curr_channels = channels[(len(channels) - len(channels)%self.pool_every):]
            layers.append(ResidualBlock(in_channels=in_channels, 
                                        channels=curr_channels,kernel_sizes=[3]*len(curr_channels), 
                                        batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type, 
                                        activation_params = self.activation_params))
        # raise NotImplementedError()
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, channels, pool_every, hidden_dims, **kwargs):
        super().__init__(in_size, out_classes, channels, pool_every, hidden_dims, **kwargs)

    # TODO: Change whatever you want about the ConvClassifier to try to
    #  improve it's results on CIFAR-10.
    #  For example, add batchnorm, dropout, skip connections, change conv
    #  filter sizes etc.
    # ====== YOUR CODE: ======
    
    def _make_feature_extractor(self):
        # For dimension calculations
        # self.class_in_h, self.class_in_w = in_h, in_w
        padding = 0
        if type(self.conv_params) is dict and 'padding' in self.conv_params.keys():
            padding = self.conv_params['padding']

        stride = 1
        if type(self.conv_params) is dict and 'stride' in self.conv_params.keys():
            stride = self.conv_params['stride']

        kernel_size = 3
        if type(self.conv_params) is dict and 'kernel_size' in self.conv_params.keys():
            kernel_size = self.conv_params['kernel_size']

        pool_kernel_size = 2
        if type(self.pooling_params) is dict and 'kernel_size' in self.pooling_params.keys():
            pool_kernel_size = self.pooling_params['kernel_size']

        lrelu_slope = 0
        if type(self.activation_params) is dict and 'negative_slope' in self.pooling_params.keys():
            lrelu_slope = self.pooling_params['negative_slope']
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        length = len(self.channels)
        Nmodulep = int(length % self.pool_every)
        Ndividedp = int(length / self.pool_every)
        p_channels = in_channels
        for i in range(Ndividedp):
            out_channels = self.channels[i * self.pool_every: i * self.pool_every + self.pool_every]
            kernel_sizes = [3] * self.pool_every
            layers.append(ResidualBlock(in_channels = p_channels, channels = out_channels, kernel_sizes = kernel_sizes,
                batchnorm=True, dropout=0.1, activation_type="lrelu", activation_params = self.activation_params))
            layers.append(nn.MaxPool2d(kernel_size=2))
            p_channels = out_channels[-1]

        if Nmodulep > 0:
            out_channels = self.channels[Ndividedp * self.pool_every:]
            kernel_sizes = [3] * Nmodulep
            layers.append(ResidualBlock(in_channels=p_channels, channels=out_channels, kernel_sizes=kernel_sizes,
                                 batchnorm=True, dropout=0.4))

        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        length = len(self.channels)
        in_features = self.channels[-1] * (in_w // (2 ** (int(length / self.pool_every)))) * (in_h // (2 ** (int(length / self.pool_every))))

        length = len(self.hidden_dims)

        for k in range(length):
            layers.append(nn.Linear(in_features=in_features, out_features=self.hidden_dims[k]))
            layers.append(nn.LeakyReLU(**self.activation_params))
            layers.append(nn.Dropout2d(p=0.4))
            in_features = self.hidden_dims[k]

        layers.append(nn.Linear(in_features=self.hidden_dims[-1], out_features=self.out_classes))
        seq = nn.Sequential(*layers)
        return seq

    # ========================
