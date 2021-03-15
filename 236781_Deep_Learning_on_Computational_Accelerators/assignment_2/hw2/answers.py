r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""

1. For $in\_features=1024$ and $out\_features = 2048$, with batch size of $N= 128$. 
All outputs would have a correspondingly size to the input batch size, in addition 
to the input features. The total Jacobian size for the aforementioned parameters is 
$[1024, 2048, 128]$ with a total number of parameters being $1024 \times 2048 \times 128=268435456$

2. Given that each element is 32 bits long, let us calculate the total number of bits 
necessary to store the Jacobian. $268435456 \times 32 = 8589934592 bits$ and in gigabytes. 
$1 byte = 8 bits$, total bytes required $1073741824 = 1.07 GB$.

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.025
    reg = 0.01
    # raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 8
    lr_vanilla = 0.026
    lr_momentum = 0.0075
    lr_rmsprop = 0.0002
    reg = 0.0012
    # raise NotImplementedError()
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.8
    lr = 0.004
    # raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

1. For the case of dropout=0 we can see that the training accuracy is greater than 
for cases with a dropout rate. As we increase the dropout rate we notice the models 
training accuracy increases. We can attribute this to the model being more "detailed"
for smaller dropout rates, giving us a more "fitted" structure which in turn gives us
a higher accuracy for the "Training Set". The counter point for such a detailed model 
is the higher chance of "over-fitting", which we indeed can see in the "Test Set". 
Where the models with the higher dropout rate outperform those with a lower dropout 
rate.

2. In q1.1 (the previous question) we went into great detail of how the graphs acted



"""

part2_q2 = r"""

Looking at the cross-entropy loss formula, we can see the class scores for a sample 
may get various values throughout the epochs, with the correct class predicted each time. 
Let's look at an example, let us say that for an image of 5 we give the correct class 
as 5 with a score of 0.8, this one being the highest of all the other scores. In the 
next epoch for we give the 5 a score of 0.6 while all the other scores are lower. Thus 
we have simultaneously increased both the loss and the accuracy.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

1. Number of parameters for a given cnn layer is $Input\_feature\_map\times Output\_feature\_map\times 
Size\_of\_each\_filter$. 

 Strandard block - $256\times 256\times 3\times 3 + 256\times 256\times 3\times 3 \approx 1.18M$

 Bottleneck block - $256\times 64\times 1\times 1 + 64\times 64\times 3=times 3 + 
 64\times 256\times 1\times 1 \approx 69K$

 The bottleneck block has much few parameters.

2. In comparison with multiplication, all other operations are negligible, we will exclude 
activation functions performed and bias addition.
 Total number of multiplications performed are $Output_Height\times Output_Width\times 
 Input\_feature\_map\times Output\_feature\_map\times Size\_of\_each\_filter$
 In an attempt to retain map dimensions we will use padding.

 Strandard Block - $Height\times Width\times 256\times 256\times 3\times 3 + Height\times Width\times 256\times 256\times 3\times 3 \approx 
 1.18M\ Height\times Width$
 Bottleneck block - $Height\times Width\times 256\times 64\times 1\times 1 + 64\times 64\times 3=times 3 + 
 64\times 256\times 1\times 1 \approx 69K\ Height\times Width$
 Here too we can see a reduction in the amount of operations needed for the bottleneck 
 block

3. For the standard block the receptive field is larger, when using two nested $3\times3$ 
filters, we get a $5\times5$ receptive field.
The bottleneck performs on a $3\times3$ area. Across feature maps for both types, we 
use the filters on all the input channels and thus are able to combine the channels 
onto a given spatial area.

"""

part3_q2 = r"""

The graph indicates that we got the best result for $L=4$, this is true for all $K's$. 
As we saw from the paper discussing Residual Networks, our intuition would claim that 
a deeper network would perform at least as good as a shallower network. But this is 
shown not to be the case. For $L=8$ the model underperforms in comparison with $L=4$. 
We believe that the reason for the decrease in performance can be related to the fact 
that as our model gets deeper we begin to suffer from "vanishing gradients". Another 
possible explanation can be due to the increase in number of parameters needed to 
learn, we can see this clearly when comparing for a $L=4$ between $K=32$ and $K=64$ 
For really deep networks, the predictions where as successful as randomly guessing, 
$ACC=10%$, likelyhood of picking the correct number out ot 10 possible numbers. This 
is the case we got for $L=16$.
It may be possible to improve our models by using skip connections, similar to the 
ones used in Residual Blocks, this would tackle the "vanishing gradient" problem. 
Another possible solution can be to use Batch Normalization Layers, which may help 
keep the gradient more stable as it propagates through the model.

"""

part3_q3 = r"""

First off we can see that the number of epochs it takes to train a model is lower the 
fewer the layers there are, which is interesting. Due to the early stopping, we believe 
this to be due to the fact the deeper models suffer more from vanishing gradient and 
in turn over time will have diminishing inceases to the models performance. Something 
else to take note is the fact that models with fewer Kernels outperformed for shorter 
models (i.e lower L's) but this does not hold for deeper models, this is indeed curious. 
It is interesting to see that there is no truly one rule of thumb that can be deducted 
across all models, and the importance of testing multiple models and comparing them. 
It is difficult to determine ahead which would give the best results.

"""

part3_q4 = r"""

The first thing that we notice is that the deep layers simply are not stable and cannot 
be trained (perhaps in the future with better computing power and mathematical technique 
this can be overcome). Something else that stands out is the fact that in some cases 
lower L's in this experiments gave better results than the previous experiments, obviously 
the expanding K's are assumed to attribute to this.

"""

part3_q5 = r"""

We can immediately notice that for the models with fixed $K=32$ using the skip connection 
we were able to get the deeper $L\geq16$ models to converge. We assume this is to the fact 
that the deeper models using the skip connection had the ability to learn the identity matrix 
more easily. Another factor we noticed that supports our conclusion is the fact that the deeper 
models did not out perform each other by much (in the test phase)

"""

part3_q6 = r"""

1. We opted to use a model which implemented most of the elements we played around with during this exercise. 
After multiple attempts to test different builds we found that it was very difficult to select the best model 
to give us the highest accuracy. We ended using the **ResNet** model as our base, seeing as it gave a satisfactory 
performance throughout the previous experiment. We built the residual blocks using:
 A standard Convolution layer
 Followed by a Dropout layer
 Then a batch normalization layer
 Then a leaky relu layer
 Finishing off with a Convolution layer

 We implemented a standard skip connection layer to circumvent this block
 Following each block we inserted a pooling layer, we test both Avg pooling and Max pooling
 We felt that in some cases there may be rule of thumbs to follow when implementing our model, but by and large 
 we found it was a lot of trying different things and testing for best results.

2. We did find that we were able to "defeat" vanishing gradient better than in our previous experiments. In our YCN 
we choose to fully implement the classifier part, hoping that it may bear better results, we found this to be partially 
the case but still lacking satisfactory results. Our best configuration which is the one detailed above gave us above $80% ACC$ 
on the test set, where as we were unable to achieve this in the previous experiments, we can therefore claim this have been a success.
Though we believe that through further testing it may be possible to even outdo that.

"""
# ==============
