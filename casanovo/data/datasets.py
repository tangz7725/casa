"""A PyTorch Dataset class for annotated spectra."""

from typing import Optional, Tuple

import depthcharge
import numpy as np
import spectrum_utils.spectrum as sus
import torch
from torch.utils.data import Dataset
# import pymc3 as pm



class SpectrumDataset(Dataset):
    """
    Parse and retrieve collections of MS/MS spectra.

    Parameters
    ----------
    spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None`
        retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude
        TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the
        base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the
        precursor mass.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        spectrum_index: depthcharge.data.SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

    def __len__(self) -> int:
        """The number of spectra."""
        return self.n_spectra

    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, float, int, Tuple[str, str]]:
        """
        Return the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        spectrum_id: Tuple[str, str]
            The unique spectrum identifier, formed by its original peak file and
            identifier (index or scan number) therein.
        """
        mz_array, int_array, precursor_mz, precursor_charge = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        return (
            spectrum,
            precursor_mz,
            precursor_charge,
            self.get_spectrum_id(idx),
        )

    def get_spectrum_id(self, idx: int) -> Tuple[str, str]:
        """
        Return the identifier of the MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the MS/MS spectrum within the SpectrumIndex.

        Returns
        -------
        ms_data_file : str
            The peak file from which the MS/MS spectrum was originally parsed.
        identifier : str
            The MS/MS spectrum identifier, per PSI recommendations.
        """
        with self.index:
            return self.index.get_spectrum_id(idx)

    def _process_peaks(
        self,
        mz_array: np.ndarray,
        int_array: np.ndarray,
        precursor_mz: float,
        precursor_charge: int,
    ) -> torch.Tensor:
        """
        Preprocess the spectrum by removing noise peaks and scaling the peak
        intensities.

        Parameters
        ----------
        mz_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak m/z values.
        int_array : numpy.ndarray of shape (n_peaks,)
            The spectrum peak intensity values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.

        Returns
        -------
        torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        """

        # Deconvolution step
        mz_array, int_array = self.deconvolve_spectrum(mz_array, int_array)
        spectrum = sus.MsmsSpectrum(
            "",
            precursor_mz,
            precursor_charge,
            mz_array.astype(np.float64),
            int_array.astype(np.float32),
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError
            spectrum.scale_intensity("root", 1)
            intensities = spectrum.intensity / np.linalg.norm(
                spectrum.intensity
            )
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError:
            # Replace invalid spectra by a dummy spectrum.
            return torch.tensor([[0, 1]]).float()


    def deconvolve_spectrum(self, mz_array: np.ndarray, int_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deconvolve the spectrum to separate overlapping peaks.

        Parameters:
        mz_array (numpy.ndarray): The m/z values of the spectrum.
        int_array (numpy.ndarray): The intensity values of the spectrum.

        Returns:
        tuple: A tuple containing the deconvoluted m/z and intensity arrays.
        """
        # Implement your deconvolution logic here
        # For example, using a simple peak-finding algorithm
        from scipy.signal import find_peaks

        # Find peaks using a simple peak-finding algorithm
        peaks, _ = find_peaks(int_array, height=0.01 * max(int_array))

        # Extract deconvoluted peaks
        mz_deconv = mz_array[peaks]
        int_deconv = int_array[peaks]

        return mz_deconv, int_deconv

    def deconvolve_spectrum_bayes(self, mz_array: np.ndarray, int_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Deconvolve the spectrum using Bayesian deconvolution.

        Parameters:
        mz_array (numpy.ndarray): The m/z values of the spectrum.
        int_array (numpy.ndarray): The intensity values of the spectrum.

        Returns:
        tuple: A tuple containing the deconvoluted m/z and intensity arrays.
        """
        # 数据预处理
        int_array = int_array / np.max(int_array)  # 归一化强度

        # 贝叶斯解卷积模型
        with pm.Model() as model:
            # 定义先验分布
            # 峰的位置先验：均匀分布
            mu = pm.Uniform('mu', lower=0, upper=len(int_array))
            # 峰的高度先验：正态分布
            height = pm.Normal('height', mu=0.5, sigma=0.2)
            # 峰的宽度先验：指数分布
            width = pm.Exponential('width', lam=1.0)

            # 定义似然函数
            # 假设观测数据服从正态分布，均值为峰的位置，标准差为峰的宽度
            likelihood = pm.Normal('likelihood', mu=mu, sigma=width, observed=int_array)

            # 进行采样
            trace = pm.sample(1000, tune=1000, cores=1)

        # 提取采样结果
        mu_samples = trace['mu']
        height_samples = trace['height']
        width_samples = trace['width']

        # 计算最可能的峰位置和强度
        peak_indices = np.round(mu_samples).astype(int)
        peak_counts = np.bincount(peak_indices, minlength=len(int_array))
        peak_indices = np.where(peak_counts > 0)[0]

        # 提取解卷积后的峰
        mz_deconv = mz_array[peak_indices]
        int_deconv = int_array[peak_indices]

        return mz_deconv, int_deconv

    @property
    def n_spectra(self) -> int:
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self) -> depthcharge.data.SpectrumIndex:
        """The underlying SpectrumIndex."""
        return self._index

    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)


class AnnotatedSpectrumDataset(SpectrumDataset):
    """
    Parse and retrieve collections of annotated MS/MS spectra.

    Parameters
    ----------
    annotated_spectrum_index : depthcharge.data.SpectrumIndex
        The MS/MS spectra to use as a dataset.
    n_peaks : Optional[int]
        The number of top-n most intense peaks to keep in each spectrum. `None`
        retains all peaks.
    min_mz : float
        The minimum m/z to include. The default is 140 m/z, in order to exclude
        TMT and iTRAQ reporter ions.
    max_mz : float
        The maximum m/z to include.
    min_intensity : float
        Remove peaks whose intensity is below `min_intensity` percentage of the
        base peak intensity.
    remove_precursor_tol : float
        Remove peaks within the given mass tolerance in Dalton around the
        precursor mass.
    random_state : Optional[int]
        The NumPy random state. ``None`` leaves mass spectra in the order they
        were parsed.
    """

    def __init__(
        self,
        annotated_spectrum_index: depthcharge.data.SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            remove_precursor_tol=remove_precursor_tol,
            random_state=random_state,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int, str]:
        """
        Return the annotated MS/MS spectrum with the given index.

        Parameters
        ----------
        idx : int
            The index of the spectrum to return.

        Returns
        -------
        spectrum : torch.Tensor of shape (n_peaks, 2)
            A tensor of the spectrum with the m/z and intensity peak values.
        precursor_mz : float
            The precursor m/z.
        precursor_charge : int
            The precursor charge.
        annotation : str
            The peptide annotation of the spectrum.
        """
        (
            mz_array,
            int_array,
            precursor_mz,
            precursor_charge,
            peptide,
        ) = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        return spectrum, precursor_mz, precursor_charge, peptide
