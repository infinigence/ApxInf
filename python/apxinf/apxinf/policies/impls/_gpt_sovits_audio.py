"""GPU audio preprocessing adapted from the official GPT-SoVITS frontend.

Source: RVC-Boss/GPT-SoVITS, commit b4570b876f5c086819ba84e3215472733c6a7b70,
GPT_SoVITS/inference_webui.py, _resampy_gpu_filter and resample(match_librosa=True).
Only the audio filter is reused; no reference model is loaded or executed.
"""

# MIT License
#
# Copyright (c) 2024 RVC-Boss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def _resampy_gpu_filter(sr0, sr1, device, filter_cache):
    import numpy as np
    import torch
    key = "%s-%s-%s" % (sr0, sr1, str(device))
    if key not in filter_cache:
        import resampy
        interp_win, precision, rolloff = resampy.filters.get_filter("kaiser_best")
        interp_win = torch.from_numpy(interp_win.astype(np.float32)).to(device)
        interp_delta = torch.diff(interp_win, append=interp_win[-1:])
        filter_cache[key] = (interp_win, interp_delta, int(precision))
    return filter_cache[key]


def resample_kaiser_best(audio_tensor, sr0, sr1, filter_cache):
    """Match the official GPU HuBERT preprocessing, including reduction order."""
    import torch

    if int(sr0) == int(sr1):
        return audio_tensor.clone()
    ratio = float(sr1) / float(sr0)
    scale = min(1.0, ratio)
    n_orig = audio_tensor.shape[-1]
    n_out = int(n_orig * ratio)
    interp_win, interp_delta, num_table = _resampy_gpu_filter(sr0, sr1, audio_tensor.device, filter_cache)
    if ratio < 1.0:
        interp_win = interp_win * ratio
        interp_delta = interp_delta * ratio
    index_step = int(scale * num_table)
    nwin = interp_win.numel()
    t_all = torch.arange(n_out, device=audio_tensor.device, dtype=torch.float32) / ratio
    out = torch.zeros((audio_tensor.shape[0], n_out), device=audio_tensor.device, dtype=audio_tensor.dtype)
    chunk = 4096
    max_taps = (nwin + index_step - 1) // index_step
    for begin in range(0, n_out, chunk):
        t = t_all[begin:begin + chunk]
        n = torch.floor(t).long()
        frac = scale * (t - n.to(t.dtype))
        index_frac = frac * num_table
        offset = torch.floor(index_frac).long()
        eta = index_frac - offset.to(index_frac.dtype)
        ii = torch.arange(max_taps, device=audio_tensor.device).view(1, -1)
        li = offset.view(-1, 1) + ii * index_step
        lmax = torch.minimum(n + 1, (nwin - offset) // index_step).view(-1, 1)
        lm = ii < lmax
        li_safe = li.clamp(0, nwin - 1)
        lw = interp_win[li_safe] + eta.view(-1, 1) * interp_delta[li_safe]
        left_idx = (n.view(-1, 1) - ii).clamp(0, n_orig - 1)
        left = (lw * lm).to(audio_tensor.dtype)
        lv = audio_tensor[:, left_idx]
        frac_r = scale - frac
        index_frac = frac_r * num_table
        offset = torch.floor(index_frac).long()
        eta = index_frac - offset.to(index_frac.dtype)
        ri = offset.view(-1, 1) + ii * index_step
        rmax = torch.minimum(n_orig - n - 1, (nwin - offset) // index_step).view(-1, 1)
        rm = ii < rmax
        ri_safe = ri.clamp(0, nwin - 1)
        rw = interp_win[ri_safe] + eta.view(-1, 1) * interp_delta[ri_safe]
        right_idx = (n.view(-1, 1) + ii + 1).clamp(0, n_orig - 1)
        right = (rw * rm).to(audio_tensor.dtype)
        out[:, begin:begin + t.numel()] = (
            (lv * left.unsqueeze(0)).sum(-1)
            + (audio_tensor[:, right_idx] * right.unsqueeze(0)).sum(-1)
        )
    return out
