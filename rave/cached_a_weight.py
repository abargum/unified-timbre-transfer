import logging
import os
from typing import Optional, Callable

import torch as tr
from torch import Tensor
from torch import nn
from typing import Tuple

from torchaudio.transforms import Spectrogram

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))

class CircularInplaceTensorQueue:
    def __init__(self, n_ch: int, max_size: int, use_debug_mode: bool = True) -> None:
        """
        Creates a FIFO queue designed for audio data that does not allocate any memory during normal use and performs
        as few memory operations as possible. The queue is also compatible with converting to TorchScript.
        """
        self.use_debug_mode = use_debug_mode
        self.max_size = max_size
        self.queue = tr.zeros((n_ch, max_size))
        self.start_idx = 0
        self.end_idx = 0
        self.size = 0

    def _calc_push_indices(self, in_n: int) -> Tuple[int, int, int, int]:
        """
        Calculates the indices to place new data of length in_n into the queue. Since it's a circular queue this can
        mean wrapping around once past the end of the queue depending on the contents of the queue at that moment in
        time. As a result, we define two possible index ranges for pushing data: start_1:end_1 and start_2:end_2
        if wrapping occurs, otherwise end_1 == start_2 == end_2

        Returns:
            Tuple[int, int, int, int]: start_1, end_1, start_2, end_2
        """
        if self.use_debug_mode:
            assert 0 < in_n < self.max_size
        start_1 = self.end_idx
        if start_1 == self.max_size:
            start_1 = 0
        end_2 = start_1 + in_n
        if end_2 > self.max_size:
            end_2 = end_2 % self.max_size
        end_1 = end_2
        start_2 = end_2
        if end_2 < start_1:
            end_1 = self.max_size
            start_2 = 0
        return start_1, end_1, start_2, end_2

    def push(self, x: Tensor) -> None:
        """
        Pushes the contents of x to the end of the queue. If the queue does not have adequate space left, the contents
        of the queue will be overwritten, starting at the head of the queue.
        """
        if self.use_debug_mode:
            assert x.ndim == self.queue.ndim
            assert x.size(0) == self.queue.size(0)
        in_n = x.size(1)
        if in_n >= self.max_size:
            self.queue[:, :] = x[:, -self.max_size :]
            self.start_idx = 0
            self.end_idx = self.max_size
            self.size = self.max_size
            return
        if in_n < 1:
            return
        start_1, end_1, start_2, end_2 = self._calc_push_indices(in_n)
        n_1 = end_1 - start_1
        self.queue[:, start_1:end_1] = x[:, 0:n_1]
        if n_1 < in_n:
            self.queue[:, start_2:end_2] = x[:, n_1:]
        self.end_idx = end_2
        self.size = min(self.size + in_n, self.max_size)
        if self.size == self.max_size:
            self.start_idx = self.end_idx

    def _calc_pop_indices(self, out_n: int) -> Tuple[int, int, int, int]:
        """
        Calculates the indices to pop data of length out_n from the queue. Since it's a circular queue this can
        mean wrapping around once past the end of the queue depending on the contents of the queue at that moment in
        time. As a result, we define two possible index ranges for popping data: start_1:end_1 and start_2:end_2
        if wrapping occurs, otherwise end_1 == start_2 == end_2

        Returns:
            Tuple[int, int, int, int]: start_1, end_1, start_2, end_2
        """
        out_n = min(out_n, self.size)
        if self.use_debug_mode:
            assert out_n > 0
        start_1 = self.start_idx
        if start_1 == self.max_size:
            start_1 = 0
        end_2 = start_1 + out_n
        if end_2 > self.max_size:
            end_2 = end_2 % self.max_size
        end_1 = end_2
        start_2 = end_2
        if end_2 <= start_1:
            end_1 = self.max_size
            start_2 = 0
        return start_1, end_1, start_2, end_2

    def pop(self, out: Tensor) -> int:
        """
        Attempts to fill the out tensor with data popped from the head of the queue. Begins filling the out tensor at
        index 0. If the out tensor is bigger than the number of items in the queue, fills the tensor as much as
        possible.

        Returns:
            int: the number of items successfully popped from the queue.
        """
        # TODO(cm): remove duplicate code using fill
        if self.use_debug_mode:
            assert out.ndim == self.queue.ndim
            assert out.size(0) == self.queue.size(0)
        if self.is_empty():
            return 0
        out_n = out.size(1)
        if out_n < 1:
            return 0
        start_1, end_1, start_2, end_2 = self._calc_pop_indices(out_n)
        n_1 = end_1 - start_1
        n_2 = end_2 - start_2
        removed_n = n_1 + n_2
        if self.use_debug_mode:
            assert 0 < n_1 <= self.size
            assert 0 <= n_2 < self.size
            assert removed_n <= self.size
        out[:, 0:n_1] = self.queue[:, start_1:end_1]
        if n_2 > 0:
            out[:, n_1:removed_n] = self.queue[:, start_2:end_2]
        self.start_idx = end_2
        self.size -= removed_n
        if self.use_debug_mode:
            if self.size == 0:
                assert self.start_idx == self.end_idx
        return removed_n

    def fill(self, out: Tensor) -> int:
        """
        Attempts to fill the out tensor with data from the head of the queue. Begins filling the out tensor at index 0.
        If the out tensor is bigger than the number of items in the queue, fills the tensor as much as possible. Does
        not remove any elements from the queue.

        Returns:
            int: the number of items successfully filled from the queue.
        """
        if self.use_debug_mode:
            assert out.ndim == self.queue.ndim
            assert out.size(0) == self.queue.size(0)
        if self.is_empty():
            return 0
        out_n = out.size(1)
        if out_n < 1:
            return 0
        start_1, end_1, start_2, end_2 = self._calc_pop_indices(out_n)
        n_1 = end_1 - start_1
        n_2 = end_2 - start_2
        filled_n = n_1 + n_2
        if self.use_debug_mode:
            assert 0 < n_1 <= self.size
            assert 0 <= n_2 < self.size
            assert filled_n <= self.size
        out[:, 0:n_1] = self.queue[:, start_1:end_1]
        if n_2 > 0:
            out[:, n_1:filled_n] = self.queue[:, start_2:end_2]
        return filled_n

    def is_empty(self) -> bool:
        return self.size == 0

    def is_full(self) -> bool:
        return self.size == self.max_size

    def reset(self) -> None:
        self.start_idx = 0
        self.end_idx = 0
        self.size = 0
        

class Cached_A_Weight(nn.Module):
    def __init__(
        self,
        sr: int = 44100,
        n_ch: int = 1,
        n_fft: int = 2048,
        hop_len: int = 1024,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        window_fn: Callable[..., Tensor] = tr.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        center: bool = True,
        use_debug_mode: bool = True,
    ) -> None:
        """
        Creates a Mel spectrogram that supports streaming of a centered, non-causal
        Mel spectrogram operation that uses zero padding. Using this will result in
        audio being delayed by (n_fft / 2) - hop_len samples. When calling forward,
        the input audio block length must be a multiple of the hop length.

        Parameters:
            sr (int): Sample rate of the audio
            n_ch (int): Number of audio channels
            n_fft (int): STFT n_fft (must be even)
            hop_len (int): STFT hop length (must divide into n_fft // 2)
            f_min (float): Minimum frequency of the Mel filterbank
            f_max (float): Maximum frequency of the Mel filterbank
            n_mels (int): Number of mel filterbank bins
            window_fn (Callable[..., Tensor]): A function to create a window tensor
            power (float): Exponent for the magnitude spectrogram (must be > 0)
            normalized (bool): Whether to normalize the mel spectrogram or not
            center (bool): Whether to center the mel spectrogram (must be True)
            use_debug_mode (bool): Whether to use debug mode or not
        """
        super().__init__()
        assert center, "center must be True, causal mode is not supported yet"
        assert n_fft % 2 == 0, "n_fft must be even"
        assert (n_fft // 2) % hop_len == 0, "n_fft // 2 must be divisible by hop_len"
        self.n_ch = n_ch
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.use_debug_mode = use_debug_mode
        self.spec = Spectrogram(
            n_fft=n_fft,
            hop_length=hop_len,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            center=False,  # We use a causal STFT since we do the padding ourselves
        )
        
        self.sr = sr
        self.padding_n_samples = self.n_fft - self.hop_len
        self.cache = CircularInplaceTensorQueue(
            n_ch, self.padding_n_samples, use_debug_mode
        )
        
        self.register_buffer("padding", tr.zeros((n_ch, self.padding_n_samples)))
        self.cache.push(self.padding)

    def forward(self, x: Tensor) -> Tuple[tr.Tensor, tr.Tensor]:
        """
        Computes the Mel spectrogram of the input audio tensor. Supports streaming as
        long as the input audio tensor is a multiple of the hop length.
        """
        if self.use_debug_mode:
            assert x.ndim == 2, "input audio must have shape (n_ch, n_samples)"
            assert x.size(0) == self.n_ch, "input audio n_ch is incorrect"
            assert (
                x.size(1) % self.hop_len == 0
            ), "input audio n_samples must be divisible by hop_len"
        # Compute the Mel spec
        n_samples = x.size(1)
        n_frames = n_samples // self.hop_len
        padded_x = tr.cat([self.padding, x], dim=1)
        padded_spec = self.spec(padded_x)
        spec = padded_spec[:, :, -n_frames:]

        mag_stft = tr.abs(spec)
        num_freqs = mag_stft.shape[-1]
        freq_axis = tr.linspace(0, self.sr /2, num_freqs)
    
        f_sq = freq_axis ** 2
        const = tr.tensor([12194.217, 20.598997, 107.65265, 737.86223]) ** 2.0
    
        weights = 2.0 + 20.0 * (
            tr.log10(const[0])
            + 2 * tr.log10(f_sq)
            - tr.log10(f_sq + const[0])
            - tr.log10(f_sq + const[1])
            - 0.5 * tr.log10(f_sq + const[2])
            - 0.5 * tr.log10(f_sq + const[3])
        )
    
        weights = weights.view(1, 1, -1)
    
        perceptual_stft = 10 * tr.log10(mag_stft**2 + 1e-10) + weights
        loudness = tr.log10(tr.mean(tr.pow(10, perceptual_stft/10), dim=1) + 1e-5)

        loudness_norm = 10 ** (loudness / 20)

        # Update the cache and padding
        padding_idx = min(n_samples, self.padding_n_samples)
        self.cache.push(x[:, -padding_idx:])
        self.cache.fill(self.padding)
        return loudness, loudness_norm

    def prepare_for_inference(self) -> None:
        """
        Prepares the cached Mel spectrogram for inference by disabling debug mode.
        """
        self.cache.use_debug_mode = False
        self.use_debug_mode = False

    @tr.jit.export
    def get_delay_samples(self) -> int:
        """
        Returns the number of samples of delay of the cached Mel spectrogram.
        """
        return (self.n_fft // 2) - self.hop_len

    @tr.jit.export
    def get_delay_frames(self) -> int:
        """
        Returns the number of frames of delay of the cached Mel spectrogram.
        """
        return self.get_delay_samples() // self.hop_len

    @tr.jit.export
    def reset(self) -> None:
        """
        Resets the cache and padding of the cached Mel spectrogram.
        """
        self.cache.reset()
        self.padding.zero_()
        self.cache.push(self.padding)