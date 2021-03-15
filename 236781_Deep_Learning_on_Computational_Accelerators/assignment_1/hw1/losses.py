import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).

        loss = None
        # ====== YOUR CODE: ======

        correctly_labelled = y_predicted == y
        y_ = y.unsqueeze(1)
        correct_values = x_scores.gather(1,y_)

        M = x_scores - correct_values
        M += self.delta
        tally = torch.sum(torch.clamp(M, min=0))
        delta_ = x_scores.shape[0]*self.delta
        loss = (tally -  delta_) / float(x_scores.shape[0])
        # raise NotImplementedError()
        # ========================

        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        self.grad_ctx = {
            'samples': x,
            'labels': y,
            'M': M,
        }
        # ========================

        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        grad = None
        # ====== YOUR CODE: ======
        x = self.grad_ctx['samples']
        y = self.grad_ctx['labels']
        M = self.grad_ctx['M']


                
        y_ = y.unsqueeze(1)
        label_mask = torch.zeros(M.shape).scatter_(dim=1, index=y_, src=torch.ones(M.shape))

        wrong_label_mask = torch.logical_not(label_mask, out=torch.empty(label_mask.size(), dtype=torch.int16))
        
        gt_zero_index = torch.gt(M, torch.zeros(*M.shape))

        incorrect_label_indx = wrong_label_mask * gt_zero_index.float()

        correct_labels = y.view(M.shape[0], 1)
        
        tally_classes = torch.sum(incorrect_label_indx, dim=1, keepdim=True)

        final_tally = torch.zeros(*M.shape).scatter_(dim=1, index=correct_labels, src=tally_classes)


        G = incorrect_label_indx - final_tally

        
        grad = torch.mm(x.T, G) / float(M.size(0))
        # raise NotImplementedError()
        # ========================

        return grad
