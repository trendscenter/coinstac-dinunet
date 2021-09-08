import os as _os

import numpy as _np
import torch as _torch

from coinstac_dinunet.distrib.learner import COINNLearner
from coinstac_dinunet.distrib.reducer import COINNReducer

_SKIP_NORM_Layers = [_torch.nn.BatchNorm1d, _torch.nn.LayerNorm, _torch.nn.GroupNorm]


def _dad_trainable_module(module):
    a_norm_layer = any([isinstance(module, k) for k in _SKIP_NORM_Layers])
    if a_norm_layer:
        return False

    """Has trainable parameters"""
    return len(list(module.parameters())) > 0


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
        self.is_dad_module = {}
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
        def _hook_recursive(module_name, module):
            children = list(module.named_children())
            self._is_dad_module[module_name] = False
            if len(children) > 0:
                for children_name, child in children:
                    _hook_recursive(
                        self._hierarchy_key(module_name, children_name),
                        child
                    )

            elif _dad_trainable_module(module):
                self._is_dad_module[module_name] = True
                self.fw_hooks_handle.append(
                    module.register_forward_hook(self._hook_fn('forward', module_name))
                )
                self.bk_hooks_handle.append(
                    module.register_backward_hook(self._hook_fn('backward', module_name))
                )

        if self.training:
            for ch_name, ch in list(self.module.named_children()):
                _hook_recursive(ch_name, ch)

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
        fk = list(self.trainer.nn.keys())[0]
        dad_params = dict([(k, v) for k, v in self.trainer.nn[fk].module.named_parameters()])
        dad_children = dict([(k, v) for k, v in self.trainer.nn[fk].module.named_children()])
        for layer in list(dad_children.keys())[::-1]:
            act_tall = _torch.FloatTensor(
                _np.load(self.state['baseDirectory'] + _os.sep + self.input['tall_activation_file'][layer]),
                device=self.trainer.device['gpu'])
            local_grad_tall = _torch.FloatTensor(
                _np.load(self.state['baseDirectory'] + _os.sep + self.input['tall_grads_file'][layer]),
                device=self.trainer.device['gpu'])
            dad_params[f"{layer}.weight"].grad.data = (act_tall.T.mm(local_grad_tall)).T.contiguous()
            if dad_params.get(f"{layer}.bias") is not None:
                dad_params[f"{layer}.bias"].grad.data = local_grad_tall.sum(0)

        first_optim = list(self.trainer.optimizer.keys())[0]
        self.trainer.optimizer[first_optim].step()
        return out

    def forward(self):
        out = {}

        first_model = list(self.trainer.nn.keys())[0]
        first_optim = list(self.trainer.optimizer.keys())[0]

        self.trainer.nn[first_model].train()
        self.trainer.optimizer[first_optim].zero_grad()

        its = []
        for _ in range(self.cache['local_iterations']):
            batch, nxt_iter_out = self.trainer.data_handle.next_iter()
            it = self.trainer.iteration(batch)
            it['loss'].backward()
            its.append(it)
            out.update(**nxt_iter_out)
            """Cannot use grad accumulation with DAD at the moment"""
            break
        return self.trainer.reduce_iteration(its), out

    def to_reduce(self):
        it, out = {}, {}
        fk = list(self.trainer.nn.keys())[0]
        if not isinstance(self.trainer.nn[fk], DADParallel):
            self.trainer.nn[fk] = DADParallel(self.trainer.nn[fk])

        it, fw_out = self.forward()
        out.update(**fw_out)

        def _backward(module_name, module):
            dad_params = dict(list(module.named_parameters())[::-1])
            dad_children = dict(list(module.named_children())[::-1])

            if len(dad_children) > 0:
                for child_name, child in dad_children.items():
                    _backward(self._hierarchy_key(module_name, child_name), child)

            elif self.trainer.nn[fk].is_dad_module.get(module_name):
                """ Update and sync weights """
                dad_params = dict(list(module.named_parameters())[::-1])
                dad_children = dict(list(module.named_children())[::-1])
                out['dad_layers'] = dict([(k, dict()) for k in dad_children.keys()][::-1])

                for layer_name in out['dad_layers']:
                    out['dad_layers'][layer_name]['activation_file'] = f"{layer_name}-act.npy"
                    out['dad_layers'][layer_name]['grads_file'] = f"{layer_name}-grads.npy"

                    _np.save(f"{self.state['transferDirectory']}{_os.sep}{out['dad_layers'][layer_name]['activation_file']}",
                             self.trainer.nn[fk].activation_ctx[layer_name].detach().numpy())
                    _np.save(f"{self.state['transferDirectory']}{_os.sep}{out['dad_layers'][layer_name]['grads_file']}",
                             self.trainer.nn[fk].local_grads_ctx[layer_name].detach().numpy())

        out['dad_layers'] = {}
        for ch_name, ch in list(self.trainer.nn[fk].module.named_children())[::-1]:
            _backward(ch_name, ch)

        # def _backward(module):
        #     dad_params = dict(list(module.named_parameters())[::-1])
        #     dad_children = dict(list(module.named_children())[::-1])
        #     out['dad_layers'] = dict([(k, dict()) for k in dad_children.keys()][::-1])
        #
        #     for layer_name in out['dad_layers']:
        #         out['dad_layers'][layer_name]['activation_file'] = f"{layer_name}-act.npy"
        #         out['dad_layers'][layer_name]['grads_file'] = f"{layer_name}-grads.npy"
        #
        #         _np.save(f"{self.state['transferDirectory']}{_os.sep}{out['dad_layers'][layer_name]['activation_file']}",
        #                  self.trainer.nn[fk].activation_ctx[layer_name].detach().numpy())
        #         _np.save(f"{self.state['transferDirectory']}{_os.sep}{out['dad_layers'][layer_name]['grads_file']}",
        #                  self.trainer.nn[fk].local_grads_ctx[layer_name].detach().numpy())


        out['reduce'] = True
        return it, out

    def _hierarchy_key(self, *args):
        return ".".join([f"{a}" for a in args])


class DADReducer(COINNReducer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def reduce(self):
        out = {}
        out['tall_grads_file'] = {}
        out['tall_activation_file'] = {}
        site_var = list(self.input.values())[0]
        for layer_name, layer in site_var['dad_layers'].items():
            h, delta = [], []
            out['tall_activation_file'][layer_name] = f"{layer_name}_activation_tall.npy"
            out['tall_grads_file'][layer_name] = f"{layer_name}_grads_tall.npy"
            for site, site_var in self.input.items():
                _layer = site_var['dad_layers'][layer_name]
                h.append(
                    _np.load(self.state['baseDirectory'] + _os.sep + site + _os.sep + _layer['activation_file'])
                )
                delta.append(
                    _np.load(self.state['baseDirectory'] + _os.sep + site + _os.sep + _layer['grads_file'])
                )
            _np.save(self.state['transferDirectory'] + _os.sep + out['tall_activation_file'][layer_name],
                     _np.concatenate(h))
            _np.save(self.state['transferDirectory'] + _os.sep + out['tall_grads_file'][layer_name],
                     _np.concatenate(delta))
        out['update'] = True
        return out
