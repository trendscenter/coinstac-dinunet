import os as _os

import numpy as _np
import torch as _torch

from coinstac_dinunet.distrib.learner import COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer


def check(logic, k, v, kw):
    phases = []
    for site_vars in kw.values():
        phases.append(site_vars.get(k) == v)
    return logic(phases)


def power_iteration_BC(B, C, rank=10, numiterations=20, device='cuda', tol=0.0):
    CC = _torch.mm(C.T, C)
    BCC = _torch.mm(B, CC)

    def zero_result():
        sigma = _torch.tensor(0.0, device=device)
        b_k = _torch.zeros(B.shape[0], device=device)
        c_k = _torch.zeros(C.shape[0], device=device)
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": b_k}

    def eigenvalue(B, v):
        Bv = _torch.mv(B.T, v)
        return _torch.sqrt(Bv.dot(_torch.mv(CC, Bv)))

    def past_values(computed_eigs):
        bb = _torch.stack([x['b'] for x in computed_eigs], 0)
        vv = _torch.stack([x['v'] for x in computed_eigs], 0)
        return bb, vv

    def iterations(computed_eigs=[], is_sigma=1):
        if not is_sigma: return zero_result()
        # start with one of the columns
        b_k = _torch.rand(B.shape[0], device=device)
        # b_k = B[:, 0]  # np.random.randn(B.shape[0])
        if computed_eigs:
            bb, vv = past_values(computed_eigs)
        for _ in range(numiterations):
            adjuster = _torch.tensor(0.0, device=device)
            if computed_eigs:
                adjuster = _torch.mv(vv.T, _torch.mv(bb, b_k))
            # calculate the matrix-by-vector product (BC'CB' - adjusting_matrix)b
            b_k1 = _torch.mv(BCC, _torch.mv(B.T, b_k)) - adjuster
            # calculate the norm of b
            b_k1_norm = _torch.norm(b_k1)
            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        sigma = eigenvalue(B, b_k)
        if _torch.isnan(sigma): return zero_result()
        c_k = _torch.mv(C, _torch.mv(B.T, b_k)) / sigma
        if len(computed_eigs) > 1 and _torch.norm(b_k - computed_eigs[-1]['b']) / _torch.norm(
                computed_eigs[-1]['b']) < tol:
            r = zero_result()
            computed_eigs[-1]['b'] = r['b']
            computed_eigs[-1]['c'] = r['c']
            computed_eigs[-1]['sigma'] = r['sigma']
            return zero_result()
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": sigma * sigma * b_k}

    eigs = [{"sigma": _torch.tensor(1.0, device=device)}]
    for i in range(rank):
        eigs += [iterations(computed_eigs=eigs[1:], is_sigma=eigs[-1]["sigma"])]
        if eigs[-1]["sigma"] == 0.0:
            break
    eigs = eigs[1:-2]
    return (
        _torch.stack([x["sigma"] * x["b"] for x in eigs], 1),
        _torch.stack([x["c"] for x in eigs], 1),)


class DADParallel(_torch.nn.Module):
    def __init__(self, module, reduction_rank=5, num_pow_iters=1):
        super(DADParallel, self).__init__()
        self.module = module
        self.reduction_rank = reduction_rank
        self.num_pow_iters = num_pow_iters
        self._reset()

    def _reset(self):
        self.fw_hooks_handle = []
        self.bk_hooks_handle = []
        self.activation_ctx = {}
        self.local_grads_ctx = {}

    def _hook_fn(self, hook_type, layer):
        def get(m, in_grad, out_grad):
            if hook_type.lower() == 'forward':
                for i, b in enumerate(in_grad):
                    if b is not None:
                        self.activation_ctx[layer] = b
                    break
            if hook_type.lower() == 'backward':
                for i, c in enumerate(out_grad):
                    if c is not None:
                        self.local_grads_ctx[layer] = c
                    break

        return get

    def _hook(self):
        if self.training:
            for layer, ch in list(self.module.named_children()):
                self.fw_hooks_handle.append(
                    ch.register_forward_hook(self._hook_fn('forward', layer))
                )
                self.bk_hooks_handle.append(
                    ch.register_backward_hook(self._hook_fn('backward', layer))
                )

    def _unhook(self):
        for hk in self.fw_hooks_handle:
            hk.remove()
        for hk in self.bk_hooks_handle:
            hk.remove()

    def train(self, mode=True):
        self.module.train(mode)
        self._hook()
        return self

    def eval(self):
        self.module.eval()
        self._unhook()
        return self

    def forward(self, *inputs, **kwargs):
        if self.training:
            self._reset()
        output = self.module(*inputs, **kwargs)
        return output


class DADLearner(COINNLearner):

    def step(self) -> dict:
        out = {}
        out['save_state'] = False
        if self.input.get('dad_step'):
            fk = list(self.trainer.nn.keys())[0]
            dad_params = dict([(k, v) for k, v in self.trainer.nn[fk].module.named_parameters()])
            dad_children = dict([(k, v) for k, v in self.trainer.nn[fk].module.named_children()])
            for layer in list(dad_children.keys())[::-1]:
                act_tall = _torch.FloatTensor(
                    _np.load(self.state['baseDirectory'] + _os.sep + f"{layer}_activation.npy"),
                    device=self.trainer.device['gpu'])
                local_grad_tall = _torch.FloatTensor(
                    _np.load(self.state['baseDirectory'] + _os.sep + f"{layer}_local_grads.npy"),
                    device=self.trainer.device['gpu'])
                dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
                if dad_params.get(f"{layer}.bias") is not None:
                    dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        out['save_state'] = True
        return out

    def forward(self):
        out = {}

        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        its = []
        # for _ in range(self.cache['local_iterations']):
        its = []
        for _ in range(self.cache['local_iterations']):
            batch, nxt_iter_out = self.trainer.data_handle.next_iter()
            it = self.trainer.iteration(batch)
            it['loss'].backward()
            its.append(it)
            out.update(**nxt_iter_out)
            """Cannot use grad accumulation with DAD at the moment"""
            break
        return out, self.trainer.reduce_iteration(its)

    def to_reduce(self):
        out, it = {}, {}
        fk = list(self.trainer.nn.keys())[0]
        self.trainer.nn[fk] = DADParallel(self.trainer.nn[fk])

        if len(self.cache.get('dad_layers', [])) == 0:
            out, it = self.forward()
            self.cache['dad_layers'] = [k for k, v in self.trainer.nn[fk].module.named_children()]
            out['last_layer'] = True

        if len(self.cache['dad_layers']) > 0:
            self.cache['dad_layer'] = self.cache['dad_layers'].pop()
            out['activation_file'] = f"{self.cache['dad_layer']}-act.npy"
            out['local_grads_file'] = f"{self.cache['dad_layer']}-local_grads.npy"

            local_grad, act = power_iteration_BC(
                self.trainer.nn[fk].local_grads_ctx[self.cache['dad_layer']].T,
                self.trainer.nn[fk].activation_ctx[self.cache['dad_layer']].T,
                rank=self.cache.get('dad_reduction_rank', 10),
                numiterations=self.cache.get('dad_num_pow_iters', 20),
                device=self.trainer.device['gpu']
            )

            _np.save(f"{self.state['transferDirectory']}{_os.sep}{out['activation_file']}", act.T.numpy())
            _np.save(f"{self.state['transferDirectory']}{_os.sep}{out['local_grads_file']}", local_grad.T.numpy())

            out['dad_layer'] = self.cache['dad_layer']
        out['to_step'] = len(self.cache['dad_layers']) == 0
        out['reduce'] = True
        return out, it


class DADReducer(COINNReducer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def reduce_VGG(self):
        out = {}
        h, grad_prev = [], []
        for site, site_vars in self.input.items():
            h.append(
                _np.load(self.state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['activation_file'])
            )
            grad_prev.append(
                _np.load(self.state['baseDirectory'] + _os.sep + site + _os.sep + site_vars['local_grads_file'])
            )

        _np.save(self.state['transferDirectory'] + _os.sep + site_vars['dad_layer'] + "_activation.npy",
                 _np.concatenate(h))
        _np.save(self.state['transferDirectory'] + _os.sep + site_vars['dad_layer'] + "_local_grads.npy",
                 _np.concatenate(grad_prev))

        out['update'] = True
        out['dad_step'] = check(all, 'to_step', True, self.input)
        return out

    def reduce(self):
        return self.reduce_VGG()
