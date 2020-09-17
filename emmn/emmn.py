import torch
from torch.optim.optimizer import Optimizer


class EMMN(Optimizer):
    u"""Implements EMMN algorithm with SLSBoost algorithm

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1)
        beta (float, optional): discount factor of EMA (default: 0.95)
        sigma(float | tuple): setting for target standard deviation. (default: (0.3, 0.99))
            | float: fixed target standard deviation will be used for normalized residual gradient (recommended value = 2e-4)
            | tuple(float, float): relative target standard deviation will be used. (λ, β_v) of λ√EMA(mean(Var(θ'))| β_v))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): factor of one step weight decay (i.e. one_step_weight_decay = weight_decay * lr) (default: 0)
        sls_boost (tuple:(float, float) | None, optional): (mixing ratio of boosting vector, discount factor of EMA(updater)) (default: None)
    """

    def __init__(self, params, lr=1, beta=0.95, sigma=(0.3, 0.99), eps=1e-8, weight_decay=0, sls_boost=None):
        if lr <= 0.0:
            raise ValueError("Argument lr must be 0.0 < lr, but lr={}".format(lr))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Argument beta must be 0.0 <= beta <= 1.0, but beta={}".format(beta))
        if not (isinstance(sigma, float) or (isinstance(sigma, tuple) and len(sigma) == 2)):
            raise ValueError("Argument sigma must be float or tuple of (float, float), but type(sigma)={}".format(type(sigma)))
        if isinstance(sigma, float) and sigma <= 0.0:
            raise ValueError("Argument sigma must be 0.0 < sigma, but sigma={}".format(sigma))
        if isinstance(sigma, tuple) and sigma[0] <= 0.0:
            raise ValueError("Argument sigma[0] must be 0.0 < sigma[0], but sigma[0]={}".format(sigma[0]))
        if isinstance(sigma, tuple) and not 0.0 <= sigma[1] <= 1.0:
            raise ValueError("Argument sigma[1] must be 0.0 <= sigma[1] <= 1.0, but sigma[1]={}".format(sigma[1]))
        if eps <= 0.0:
            raise ValueError("Argument eps must be 0.0 < eps, but eps={}".format(eps))
        if weight_decay < 0:
            raise ValueError("Argument weight_decay must be 0.0 <= weight_decay, but weight_decay={}".format(weight_decay))
        if sls_boost is not None:
            if not 0.0 <= sls_boost[0] <= 1.0:
                raise ValueError("Argument sls_boost[0] must be 0.0 <= sls_boost[0] <= 1.0, but sls_boost[0]={}".format(sls_boost[0]))
            if not 0.0 <= sls_boost[1] <= 1.0:
                raise ValueError("Argument sls_boost[1] must be 0.0 <= sls_boost[1] <= 1.0, but sls_boost[1]={}".format(sls_boost[1]))

        defaults = dict(lr=lr, beta=beta, sigma=sigma, eps=eps, weight_decay=weight_decay, sls_boost=sls_boost)
        super(EMMN, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('EMMN does not support sparse gradients')
                state = self.state[p]

                if isinstance(group['sigma'], float):
                    sigma = group['sigma']
                    mode = 'fix'
                else:
                    sigma = group['sigma'][0]
                    beta_var = group['sigma'][1]
                    mode = 'relative'
                beta = group['beta']
                sls_boost = group['sls_boost']

                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p.data)
                    state['ema_sqr_grad'] = torch.zeros_like(p.data)
                    if mode == 'relative':
                        state['ema_mean_var'] = torch.zeros(1).to(p.data.device)

                state['step'] += 1
                step = state['step']
                ema_grad = state['ema_grad']
                ema_sqr_grad = state['ema_sqr_grad']
                ema_grad.mul_(beta).add_(1 - beta, grad)
                ema_sqr_grad.mul_(beta).addcmul_(1 - beta, grad, grad)
                var_grad = ema_sqr_grad.addcmul(-1, ema_grad, ema_grad)
                std_grad = var_grad.sqrt().add(group['eps'])
                grad_adjusted = grad * (1 - beta**step)
                if step == 1:
                    emma_grad = grad_adjusted
                else:
                    residual = grad_adjusted - ema_grad

                    if mode == 'relative':
                        ema_mean_var = state['ema_mean_var']
                        ema_mean_var.mul_(beta_var).add_(1 - beta_var, var_grad.mean())
                        ema_mean_var_bc = ema_mean_var / (1 - beta_var**(step-1))
                        residual *= sigma * ema_mean_var_bc.sqrt() / std_grad
                    else: # if mode == 'fix'
                        residual *= sigma / std_grad
                    emma_grad = ema_grad + residual

                    if sls_boost is not None:
                        if 'ema_u' not in state:
                            ema_u = state['ema_u'] = torch.clone(emma_grad).detach() * (1 - sls_boost[1])
                        else:
                            ema_u = state['ema_u']
                        target_vec = ema_u * emma_grad.norm() / ema_u.norm()
                        boost_vec = sls_boost[0] * (target_vec - emma_grad)
                        updater = emma_grad + boost_vec
                        # updater *= emma_grad.norm() / updater.norm()
                        ema_u.mul_(sls_boost[1]).add_(updater, alpha=1 - sls_boost[1])
                        emma_grad = updater

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                p.data.add_(-group['lr'], emma_grad)

        return loss


class EMMN_PDCBoost(Optimizer):
    u"""Implements EMMN algorithm with PDCBoost algorithm

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1)
        beta (float, optional): discount factor of EMA (default: 0.95)
        sigma(float | tuple): setting for target standard deviation. (default: (0.3, 0.99))
            | float: fixed target standard deviation will be used for normalized residual gradient (recommended value = 2e-4)
            | tuple(float, float): relative target standard deviation will be used. (λ, β_v) of λ√EMA(mean(Var(θ'))| β_v))
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-8)
        weight_decay (float, optional): factor of one step weight decay (i.e. one_step_weight_decay = weight_decay * lr) (default: 0)
        pdc_boost (tuple:(float, float) | None, optional): (mixing ratio of boosting vector, discount factor of EMA(updater)) (default: None)
    """

    def __init__(self, params, lr=1, beta=0.95, sigma=(0.3, 0.99), eps=1e-8, weight_decay=0, pdc_boost=None):
        if lr <= 0.0:
            raise ValueError("Argument lr must be 0.0 < lr, but lr={}".format(lr))
        if not 0.0 <= beta <= 1.0:
            raise ValueError("Argument beta must be 0.0 <= beta <= 1.0, but beta={}".format(beta))
        if not (isinstance(sigma, float) or (isinstance(sigma, tuple) and len(sigma) == 2)):
            raise ValueError("Argument sigma must be float or tuple of (float, float), but type(sigma)={}".format(type(sigma)))
        if isinstance(sigma, float) and sigma <= 0.0:
            raise ValueError("Argument sigma must be 0.0 < sigma, but sigma={}".format(sigma))
        if isinstance(sigma, tuple) and sigma[0] <= 0.0:
            raise ValueError("Argument sigma[0] must be 0.0 < sigma[0], but sigma[0]={}".format(sigma[0]))
        if isinstance(sigma, tuple) and not 0.0 <= sigma[1] <= 1.0:
            raise ValueError("Argument sigma[1] must be 0.0 <= sigma[1] <= 1.0, but sigma[1]={}".format(sigma[1]))
        if eps <= 0.0:
            raise ValueError("Argument eps must be 0.0 < eps, but eps={}".format(eps))
        if weight_decay < 0:
            raise ValueError("Argument weight_decay must be 0.0 <= weight_decay, but weight_decay={}".format(weight_decay))
        if pdc_boost is not None:
            if not 0.0 <= pdc_boost[0] <= 1.0:
                raise ValueError("Argument pdc_boost[0] must be 0.0 <= pdc_boost[0] <= 1.0, but pdc_boost[0]={}".format(pdc_boost[0]))
            if not 0.0 <= pdc_boost[1] <= 1.0:
                raise ValueError("Argument pdc_boost[1] must be 0.0 <= pdc_boost[1] <= 1.0, but pdc_boost[1]={}".format(pdc_boost[1]))

        defaults = dict(lr=lr, beta=beta, sigma=sigma, eps=eps, weight_decay=weight_decay, pdc_boost=pdc_boost)
        super(EMMN_PDCBoost, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('EMMN does not support sparse gradients')
                state = self.state[p]

                if isinstance(group['sigma'], float):
                    sigma = group['sigma']
                    mode = 'fix'
                else:
                    sigma = group['sigma'][0]
                    beta_var = group['sigma'][1]
                    mode = 'relative'
                beta = group['beta']
                pdc_boost = group['pdc_boost']

                if len(state) == 0:
                    state['step'] = 0
                    state['ema_grad'] = torch.zeros_like(p.data)
                    state['ema_sqr_grad'] = torch.zeros_like(p.data)
                    if mode == 'relative':
                        state['ema_mean_var'] = torch.zeros(1).to(p.data.device)

                state['step'] += 1
                step = state['step']
                ema_grad = state['ema_grad']
                ema_sqr_grad = state['ema_sqr_grad']
                ema_grad.mul_(beta).add_(1 - beta, grad)
                ema_sqr_grad.mul_(beta).addcmul_(1 - beta, grad, grad)
                var_grad = ema_sqr_grad.addcmul(-1, ema_grad, ema_grad)
                std_grad = var_grad.sqrt().add(group['eps'])
                grad_adjusted = grad * (1 - beta**step)
                if step == 1:
                    emma_grad = grad_adjusted
                else:
                    residual = grad_adjusted - ema_grad

                    if mode == 'relative':
                        ema_mean_var = state['ema_mean_var']
                        ema_mean_var.mul_(beta_var).add_(1 - beta_var, var_grad.mean())
                        ema_mean_var_bc = ema_mean_var / (1 - beta_var**(step-1))
                        residual *= sigma * ema_mean_var_bc.sqrt() / std_grad
                    else: # if mode == 'fix'
                        residual *= sigma / std_grad
                    emma_grad = ema_grad + residual

                    if pdc_boost is not None:
                        if 'ema_u' not in state:
                            ema_u = state['ema_u'] = torch.clone(emma_grad).detach() * (1 - pdc_boost[1])
                        else:
                            ema_u = state['ema_u']
                        target_vec = ema_u.sign()
                        target_vec *= emma_grad.norm() / target_vec.norm()
                        boost_vec = pdc_boost[0] * (target_vec - emma_grad)
                        updater = emma_grad + boost_vec
                        # updater *= emma_grad.norm() / updater.norm()
                        ema_u.mul_(pdc_boost[1]).add_(updater, alpha=1 - pdc_boost[1])
                        emma_grad = updater

                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                p.data.add_(-group['lr'], emma_grad)

        return loss
