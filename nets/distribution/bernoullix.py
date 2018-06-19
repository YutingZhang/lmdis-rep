import nets.distribution.bernoulli

BaseFactory = nets.distribution.bernoulli.Factory


class Factory(BaseFactory):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # use cross-entropy for nll
    def nll(self, dist_param, samples):
        gt_dist_param = self.parametrize(samples, None)  # no need for latent dim for bernoulli
        xnll = self.cross_entropy(gt_dist_param, dist_param)
        return xnll, True

