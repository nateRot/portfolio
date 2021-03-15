r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
There are 3 main pools of data we use when slelecting the full model we believe will give us the best prediction overall. Train, validation,
and Test. The training pool allows us to select the best parameters for a selected model of our choosing, the validation pool allows us to select the
best hyper-parameters. Lastly the Test pool is used to compare different models that have been trained, with both their parameters and hyper-parameters 
configured. Which in turn allows us to select the overall best model.
The ideal situation would be that each pool is unique, with regard to the data. Yet in cases were not enough data is on hand (though considered bad
 practice) sometimes, engineers will reuse training data as test data (not recommended).
The selection of the test set is to be as broad as possible across the different smaples we may encounter in the real world. We will also try to select
the outlying data samples, to see how our model would predict such samples.
"""

part1_q2 = r"""
**Your answer:**
For cases where the selected model has no hyper-parameters we can forgo the validation set and use all data for training. By and large we prefer to keep
the data within each set unique. 

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
Either too large or too small a k, will not deliver good results for unseen data. If we examine the extremes k = 1, would not give enough "diversity" and
out lying data would get clustered with more reliable data, otherwise known as **noise**. As for too large a k, we would get "overfitting" where we would be unable to connect clumps
of closely related data. This is consistent with our results


"""

part2_q2 = r"""
**Your answer: **

1. If we were to compare our models by the accuracy they achieved on the *train* set, we would have a high probability of selecting a more train-fitted model.
Though we have no assurance that it is the best for general *unseen* data. We use *cross-validation* to find the model which gives best results for data 
unknown to the model.

2. If we were to train and select the model based on the test set, we would pass over **hyperparameter optimization**. This could lead to missing a more ideal
model which can give better results for unseen general data.

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

If we were to attempt to use SVM loss without regularization, the delta would impact our weights. For a large delta we would get the same weights just by a 
larger magnitude. When using regularization, the weights will get "punished" for growing to large and the lambda and delta will battle each other out.
Therefore we can select arbitrary values for delta and select the *correct* lambda to recieve the best result.

"""

part3_q2 = r"""
**Your answer:**

1. We can see from the images which pixels are more dominant in determining what number the hand written image is. The weights learned are the pixels that
give the best repressentation for determining what number is being predicted. Number 1 for example the dominant pixels are more focused in a perpendicular
line through the center of the image. We can see the model will have more difficulty distinguishing between numbers with similar weight distributions, i.e 
7 and 9's dominant weights are concentrated in the upper right corner. We can expect mislabeling each other.

2. The k-nn implementation we did in this assignment (there is a different version, storing k points averaging each set, saves the need to store all the data).
Does a more "pooling" approach, we compare each new data with existing data trying to see which group claims greatest proximity to it. The model we built here
attempts to give a generalized representation learned from labelled data. All new data is compared with this "representation" we created. The comparison
determines what prediction we give the data.

"""

part3_q3 = r"""
**Your answer:**

1. We beleive our learning rate was good, we can see a sharp increase early on in the training session, after it begins to taper down, a signle we reach near
"optimal" solution quickly, meaning we didn't undershoot. In addition we can't see to much spiking in the graph meaning we remaind in near proximity to the
minima we found, signaling our learning rate was not to great.

Had the learning rate been to small, we would notice a near constant steady increase throughout the training session, with no clear indication we plateaued
near the optimal model.
On the other hand had our learning rate been too high, we would have noticed spiking in our model. Should that our model was not able to stabalize near a 
minima.

2. We beleive our model to be slightly over fitted to our training data. We reached over 90% accuracy for our training data towards the end of the training
session, yet our test data returned an accuracy of slightly less than 90%.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**

Using the residual plot we can get an idea of of the orientation of our model and whether or not I does a good job of "capturing" the data. A majority of the 
data should sit between the borders (dotted lines) with perhaps a few outlying items. An indication for a "good" model would be one one that uses features
to partion the test data in a similar way to the training data, as we can see we accomplished! Increasing the data's degree gives us insight up until a certain
point, after which it seems to make the system more sensitive to noise. The top 5 features have two strongly fitted features with the following 3 a bit more
sporadic. Though the final model seems to show a good indication with clustered data. 

"""

part4_q2 = r"""
**Your answer:**

1. Even after using non-linear features our model remains a linear model. The model itself is linear, meaning we apply linear operations to the data's features
2. Yes, we can fit any non-linear function of the original features. We would transform the features from it's domain using a non-linear operator and finally
apply the linear model, which is itself a linear operation
3. The hyperplane is just a composition of linear operations, meaning that as noted before appyling non-linear features would not alter the form of the 
hyperplane. Though the decision boundries may change! As the features not are given a different representation that before, this would most likely cause the
boundries to be realligned accordingly. 

"""

part4_q3 = r"""
**Your answer:**

1. Choosing `np.logspace` allows trying different scales of regularization values, so can find the "sweet" spot between optimal generalization and estimation
error. Larger Lambdas will benifit generalization, while smaller Lambdas will improve estimation error. Finding the ideal hyperparameters can be difficult
(time consuming) utilizing `np.logspace` as opposed to `np.linspace` allows us to test a greater range of values. 

2. A total of `k folds` x `num lambdas` x `num degree values`
for each pair we trained k times.
For a grand total of: 3 x 3 x 20 = 180

"""

# ==============
