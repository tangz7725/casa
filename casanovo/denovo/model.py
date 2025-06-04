"""A de novo peptide sequencing model."""

import collections
import heapq
import logging
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import depthcharge.masses
import einops
import torch
import numpy as np
import lightning.pytorch as pl
from torch.utils.tensorboard import SummaryWriter
from depthcharge.components import ModelMixin, PeptideDecoder, SpectrumEncoder

from . import evaluate
from .. import config
from ..data import ms_io

logger = logging.getLogger("casanovo")

# whh：
from Bio import pairwise2
import re
from Bio.pairwise2 import format_alignment

class Spec2Pep(pl.LightningModule, ModelMixin):
    """
    A Transformer model for de novo peptide sequencing.

    Use this model in conjunction with a pytorch-lightning Trainer.

    Parameters
    ----------
    dim_model : int
        The latent dimensionality used by the transformer model.
    n_head : int
        The number of attention heads in each layer. ``dim_model`` must be
        divisible by ``n_head``.
    dim_feedforward : int
        The dimensionality of the fully connected layers in the transformer
        model.
    n_layers : int
        The number of transformer layers.
    dropout : float
        The dropout probability for all layers.
    dim_intensity : Optional[int]
        The number of features to use for encoding peak intensity. The remaining
        (``dim_model - dim_intensity``) are reserved for encoding the m/z value.
        If ``None``, the intensity will be projected up to ``dim_model`` using a
        linear layer, then summed with the m/z encoding for each peak.
    max_length : int
        The maximum peptide length to decode.
    residues : Union[Dict[str, float], str]
        The amino acid dictionary and their masses. By default ("canonical) this
        is only the 20 canonical amino acids, with cysteine carbamidomethylated.
        If "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    max_charge : int
        The maximum precursor charge to consider.
    precursor_mass_tol : float, optional
        The maximum allowable precursor mass tolerance (in ppm) for correct
        predictions.
    isotope_error_range : Tuple[int, int]
        Take into account the error introduced by choosing a non-monoisotopic
        peak for fragmentation by not penalizing predicted precursor m/z's that
        fit the specified isotope error:
        `abs(calc_mz - (precursor_mz - isotope * 1.00335 / precursor_charge))
        < precursor_mass_tol`
    min_peptide_len : int
        The minimum length of predicted peptides.
    n_beams : int
        Number of beams used during beam search decoding.
    top_match : int
        Number of PSMs to return for each spectrum.
    n_log : int
        The number of epochs to wait between logging messages.
    tb_summarywriter : Optional[str]
        Folder path to record performance metrics during training. If ``None``,
        don't use a ``SummaryWriter``.
    train_label_smoothing : float
        Smoothing factor when calculating the training loss.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    out_writer : Optional[str]
        The output writer for the prediction results.
    calculate_precision : bool
        Calculate the validation set precision during training.
        This is expensive.
    **kwargs : Dict
        Additional keyword arguments passed to the Adam optimizer.
    """

    def __init__(
        self,
        dim_model: int = 512,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 9,
        dropout: float = 0.0,
        dim_intensity: Optional[int] = None,
        max_length: int = 100,
        residues: Union[Dict[str, float], str] = "canonical",
        max_charge: int = 5,
        precursor_mass_tol: float = 50,
        isotope_error_range: Tuple[int, int] = (0, 1),
        min_peptide_len: int = 6,
        n_beams: int = 1,
        top_match: int = 1,
        n_log: int = 10,
        tb_summarywriter: Optional[
            torch.utils.tensorboard.SummaryWriter
        ] = None,
        train_label_smoothing: float = 0.01,
        warmup_iters: int = 100_000,
        cosine_schedule_period_iters: int = 600_000,
        out_writer: Optional[ms_io.MztabWriter] = None,
        calculate_precision: bool = False,
        # whh
        # gene_filename: Optional[str] = None,
        gene_sequence: Optional[List] = None,
        **kwargs: Dict,
    ):
        super().__init__()
        # whh
        self.gene_sequence = gene_sequence
        self.save_hyperparameters()

        # Build the model.
        self.encoder = SpectrumEncoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            dim_intensity=dim_intensity,
        )
        self.decoder = PeptideDecoder(
            dim_model=dim_model,
            n_head=n_head,
            dim_feedforward=dim_feedforward,
            n_layers=n_layers,
            dropout=dropout,
            residues=residues,
            max_charge=max_charge,
        )
        self.softmax = torch.nn.Softmax(2)
        self.celoss = torch.nn.CrossEntropyLoss(
            ignore_index=0, label_smoothing=train_label_smoothing
        )
        self.val_celoss = torch.nn.CrossEntropyLoss(ignore_index=0)
        # Optimizer settings.
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        # `kwargs` will contain additional arguments as well as unrecognized
        # arguments, including deprecated ones. Remove the deprecated ones.
        for k in config._config_deprecated:
            kwargs.pop(k, None)
            warnings.warn(
                f"Deprecated hyperparameter '{k}' removed from the model.",
                DeprecationWarning,
            )
        self.opt_kwargs = kwargs

        # Data properties.
        self.max_length = max_length
        self.residues = residues
        self.precursor_mass_tol = precursor_mass_tol
        self.isotope_error_range = isotope_error_range
        self.min_peptide_len = min_peptide_len
        self.n_beams = n_beams
        self.top_match = top_match
        self.peptide_mass_calculator = depthcharge.masses.PeptideMass(
            self.residues
        )
        self.stop_token = self.decoder._aa2idx["$"]

        # Logging.
        self.calculate_precision = calculate_precision
        self.n_log = n_log
        self._history = []
        if tb_summarywriter is not None:
            self.tb_summarywriter = SummaryWriter(tb_summarywriter)
        else:
            self.tb_summarywriter = tb_summarywriter

        # Output writer during predicting.
        self.out_writer = out_writer

    # whh：forward是模型的前向传播函数，其调用beam_search_decode来预测肽序
    # 在验证和预测的函数中都调用了它
    def forward(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Predict peptide sequences for a batch of MS/MS spectra.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide predictions. A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        return self.beam_search_decode(
            spectra.to(self.encoder.device),
            precursors.to(self.decoder.device),
        )

    # whh：transformer中的decoder，实际肽序预测发生的地方
    # 在得到了预测的肽序列（多个可能的预测结果）后，其调用了_cache_finished_beams函数计算每个肽序的得分
    # 其调用了c函数，该函数从decoder得到的束(beams)中返回最高分的肽序列
    def beam_search_decode(
        self, spectra: torch.Tensor, precursors: torch.Tensor
    ) -> List[List[Tuple[float, np.ndarray, str]]]:
        """
        Beam search decoding of the spectrum predictions.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.

        Returns
        -------
        pred_peptides : List[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        memories, mem_masks = self.encoder(spectra)

        # Sizes.
        batch = spectra.shape[0]  # B
        length = self.max_length + 1  # L
        vocab = self.decoder.vocab_size + 1  # V
        beam = self.n_beams  # S

        # Initialize scores and tokens.
        scores = torch.full(
            size=(batch, length, vocab, beam), fill_value=torch.nan
        )
        scores = scores.type_as(spectra)
        tokens = torch.zeros(batch, length, beam, dtype=torch.int64)
        tokens = tokens.to(self.encoder.device)

        # Create cache for decoded beams.
        pred_cache = collections.OrderedDict((i, []) for i in range(batch))

        # Get the first prediction.
        pred, _ = self.decoder(None, precursors, memories, mem_masks)
        tokens[:, 0, :] = torch.topk(pred[:, 0, :], beam, dim=1)[1]
        scores[:, :1, :, :] = einops.repeat(pred, "B L V -> B L V S", S=beam)

        # Make all tensors the right shape for decoding.
        precursors = einops.repeat(precursors, "B L -> (B S) L", S=beam)
        mem_masks = einops.repeat(mem_masks, "B L -> (B S) L", S=beam)
        memories = einops.repeat(memories, "B L V -> (B S) L V", S=beam)
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")

        # whh：解码器的循环部分
        # The main decoding loop.
        for step in range(0, self.max_length):
            # Terminate beams exceeding the precursor m/z tolerance and track
            # all finished beams (either terminated or stop token predicted).
            (
                finished_beams,
                beam_fits_precursor,
                discarded_beams,
                # ***************************************************whh:被注释的基因组评分***************************************************
                # gene_score,
            ) = self._finish_beams(tokens, precursors, step)
            # Cache peptide predictions from the finished beams (but not the
            # discarded beams).

            # whh：这里的_cache_finished_beams函数实现了计算分数的后处理，即m/z filter这一部分
            self._cache_finished_beams(
                tokens,
                scores,
                step,
                finished_beams & ~discarded_beams,
                beam_fits_precursor,
                pred_cache,
                # ***************************************************whh:被注释的基因组评分***************************************************
                # gene_score,
            )

            # Stop decoding when all current beams have been finished.
            # Continue with beams that have not been finished and not discarded.
            finished_beams |= discarded_beams
            if finished_beams.all():
                break
            # Update the scores.
            scores[~finished_beams, : step + 2, :], _ = self.decoder(
                tokens[~finished_beams, : step + 1],
                precursors[~finished_beams, :],
                memories[~finished_beams, :, :],
                mem_masks[~finished_beams, :],
            )
            # Find the top-k beams with the highest scores and continue decoding
            # those.
            tokens, scores = self._get_topk_beams(
                tokens, scores, finished_beams, batch, step + 1
            )

        # Return the peptide with the highest confidence score, within the
        # precursor m/z tolerance if possible.
        whh_peptide_list = list(self._get_top_peptide(pred_cache))
        # whh_list_candidate = self.get_candidate_peptide_sequences(pred_cache)
        return whh_peptide_list
        # whh：这里返回的list是已经完成的了，不存在候选项
        # return list(self._get_top_peptide(pred_cache))

    # whh：首先其作用是追踪所有已完成的束
    def _finish_beams(
        self,
        tokens: torch.Tensor,
        precursors: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Track all beams that have been finished, either by predicting the stop
        token or because they were terminated due to exceeding the precursor
        m/z tolerance.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.

        Returns
        -------
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams have been
            finished.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating if current beams are within precursor m/z
            tolerance.
        discarded_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams should be
            discarded (e.g. because they were predicted to end but violate the
            minimum peptide length).
        """
        # Check for tokens with a negative mass (i.e. neutral loss).
        aa_neg_mass = [None]
        for aa, mass in self.peptide_mass_calculator.masses.items():
            if mass < 0:
                aa_neg_mass.append(aa)
        # Find N-terminal residues.
        n_term = torch.Tensor(
            [
                self.decoder._aa2idx[aa]
                for aa in self.peptide_mass_calculator.masses
                if aa.startswith(("+", "-"))
            ]
        ).to(self.decoder.device)

        # ***************************************************whh:被注释的基因组评分***************************************************
        # # whh：gene_score基因组评分初始化，初值为1，进行扣分制
        # # 这里是检查是否有传入基因组数据
        # if len(self.gene_sequence) == 0:
        #     gene_score = torch.zeros(
        #         tokens.shape[0], dtype=torch.float
        #     ).to(self.encoder.device)
        # else:
        #     gene_score = torch.ones(
        #         tokens.shape[0], dtype=torch.float
        #     ).to(self.encoder.device)

        beam_fits_precursor = torch.zeros(
            tokens.shape[0], dtype=torch.bool
        ).to(self.encoder.device)
        # Beams with a stop token predicted in the current step can be finished.
        finished_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        ends_stop_token = tokens[:, step] == self.stop_token
        finished_beams[ends_stop_token] = True
        # Beams with a dummy token predicted in the current step can be
        # discarded.
        discarded_beams = torch.zeros(tokens.shape[0], dtype=torch.bool).to(
            self.encoder.device
        )
        discarded_beams[tokens[:, step] == 0] = True
        # Discard beams with invalid modification combinations (i.e. N-terminal
        # modifications occur multiple times or in internal positions).
        if step > 1:  # Only relevant for longer predictions.
            dim0 = torch.arange(tokens.shape[0])
            final_pos = torch.full((ends_stop_token.shape[0],), step)
            final_pos[ends_stop_token] = step - 1
            # Multiple N-terminal modifications.
            multiple_mods = torch.isin(
                tokens[dim0, final_pos], n_term
            ) & torch.isin(tokens[dim0, final_pos - 1], n_term)
            # N-terminal modifications occur at an internal position.
            # Broadcasting trick to create a two-dimensional mask.
            mask = (final_pos - 1)[:, None] >= torch.arange(tokens.shape[1])
            internal_mods = torch.isin(
                torch.where(mask.to(self.encoder.device), tokens, 0), n_term
            ).any(dim=1)
            discarded_beams[multiple_mods | internal_mods] = True

        # Check which beams should be terminated or discarded based on the
        # predicted peptide.
        for i in range(len(finished_beams)):
            # Skip already discarded beams.
            if discarded_beams[i]:
                continue
            pred_tokens = tokens[i][: step + 1]
            peptide_len = len(pred_tokens)
            # whh：detokenize解码张量，获得具体的肽段序列
            # 然后进行后续操作
            peptide = self.decoder.detokenize(pred_tokens)
            # Omit stop token.
            if self.decoder.reverse and peptide[0] == "$":
                peptide = peptide[1:]
                peptide_len -= 1
            elif not self.decoder.reverse and peptide[-1] == "$":
                peptide = peptide[:-1]
                peptide_len -= 1
            # Discard beams that were predicted to end but don't fit the minimum
            # peptide length.
            if finished_beams[i] and peptide_len < self.min_peptide_len:
                discarded_beams[i] = True
                continue
            # Terminate the beam if it has not been finished by the model but
            # the peptide mass exceeds the precursor m/z to an extent that it
            # cannot be corrected anymore by a subsequently predicted AA with
            # negative mass.
            precursor_charge = precursors[i, 1]
            precursor_mz = precursors[i, 2]
            matches_precursor_mz = exceeds_precursor_mz = False
            # whh_counter = 0
            for aa in [None] if finished_beams[i] else aa_neg_mass:
                if aa is None:
                    calc_peptide = peptide
                else:
                    calc_peptide = peptide.copy()
                    calc_peptide.append(aa)
                try:
                    calc_mz = self.peptide_mass_calculator.mass(
                        seq=calc_peptide, charge=precursor_charge
                    )
                    delta_mass_ppm = [
                        _calc_mass_error(
                            calc_mz,
                            precursor_mz,
                            precursor_charge,
                            isotope,
                        )
                        for isotope in range(
                            self.isotope_error_range[0],
                            self.isotope_error_range[1] + 1,
                        )
                    ]
                    # Terminate the beam if the calculated m/z for the predicted
                    # peptide (without potential additional AAs with negative
                    # mass) is within the precursor m/z tolerance.
                    matches_precursor_mz = aa is None and any(
                        abs(d) < self.precursor_mass_tol
                        for d in delta_mass_ppm
                    )
                    # Terminate the beam if the calculated m/z exceeds the
                    # precursor m/z + tolerance and hasn't been corrected by a
                    # subsequently predicted AA with negative mass.
                    if matches_precursor_mz:
                        exceeds_precursor_mz = False
                    else:
                        exceeds_precursor_mz = all(
                            d > self.precursor_mass_tol for d in delta_mass_ppm
                        )
                        exceeds_precursor_mz = (
                            finished_beams[i] or aa is not None
                        ) and exceeds_precursor_mz
                    if matches_precursor_mz or exceeds_precursor_mz:
                        break
                except KeyError:
                    matches_precursor_mz = exceeds_precursor_mz = False
            # Finish beams that fit or exceed the precursor m/z.
            # Don't finish beams that don't include a stop token if they don't
            # exceed the precursor m/z tolerance yet.

            # ***************************************************whh:被注释的基因组评分***************************************************
            # # print(finished_beams[i],exceeds_precursor_mz)
            # # whh：如果该束已经打上了终止标记，且并没有被mz filter:
            # # 结果存放在whh_res里，例如：$AKEEEDEEDVDGK（前面的$是开始符号）
            # # 过滤了filter是115个结果，没过滤是127个结果(但原本就只有127个峰谱啊！！)
            # # 这里要确保确实有基因组数据传进来
            # if finished_beams[i] and len(self.gene_sequence) > 0:
            #     whh_need_peptide = self.decoder.detokenize(pred_tokens)
            #     whh_res = ''.join(whh_need_peptide)
            #     whh_res = whh_res.lstrip('$')
            #     # whh_counter = whh_counter + 1
            #     # 对whh_res评分，进入评分函数（传入self和whh_res）
            #     # match_score = 1  # 一开始要扣掉1分
            #     # 在循环中，随着每次匹配，match_score会因为匹配的相关度提高而越来越小
            #     # 重写Smith_Waterman的匹配
            #     gene_score[i] = 1 - calculate_smith_waterman_score(self.gene_sequence, whh_res)
            #
            #     # for each_gene in self.gene_sequence:
            #     #     # 若完全匹配，则这个temp_score也应该是1
            #     #     temp_score = calculate_match_score(whh_res, each_gene)
            #     #     if temp_score < match_score:
            #     #         match_score = temp_score
            #     # gene_score[i] = match_score
            #     # print("res!"+whh_res)

            # whh：***********我们需要的是终止了的且，没有超过m/z值的预测束************
            # if该束已经被标上终止标记，则记录此时该束与m/z值的状态
            if finished_beams[i]:
                beam_fits_precursor[i] = matches_precursor_mz
            # else没被终止，检查是否超过m/z值
            # if超过m/z值，手动标记终止，同时记录此时该束与m/z值的状态
            # 可见能走进这个elif的都是没被终止的
            elif exceeds_precursor_mz:
                finished_beams[i] = True
                beam_fits_precursor[i] = matches_precursor_mz
        # ***************************************************whh:被注释的基因组评分***************************************************
        # return finished_beams, beam_fits_precursor, discarded_beams, gene_score
        return finished_beams, beam_fits_precursor, discarded_beams

    def _cache_finished_beams(
        self,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        step: int,
        beams_to_cache: torch.Tensor,
        beam_fits_precursor: torch.Tensor,
        #     whh：这里的pred_cache是字典，质谱索引->列表（预测的结果）
        #     列表的每一个预测结果都是元组（
        #       peptide_score：肽段分数、
        #       随机数用来打破平局、
        #       aa_scores：氨基酸level的分数、
        #       pred_tokens：预测的氨基酸标记）
        pred_cache: Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ],
        # ***************************************************whh:被注释的基因组评分***************************************************
        # gene_score: torch.Tensor,
    ):
        """
        Cache terminated beams.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        step : int
            Index of the current decoding step.
        beams_to_cache : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        beam_fits_precursor: torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the beams are within the
            precursor m/z tolerance.
        pred_cache : Dict[
                int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the (negated)
            peptide score, a random tie-breaking float, the amino acid-level
            scores, and the predicted tokens is stored.
        """
        for i in range(len(beams_to_cache)):
            if not beams_to_cache[i]:
                continue
            # Find the starting index of the spectrum.
            spec_idx = i // self.n_beams
            # FIXME: The next 3 lines are very similar as what's done in
            #  _finish_beams. Avoid code duplication?
            pred_tokens = tokens[i][: step + 1]
            # Omit the stop token from the peptide sequence (if predicted).
            has_stop_token = pred_tokens[-1] == self.stop_token
            pred_peptide = pred_tokens[:-1] if has_stop_token else pred_tokens
            # Don't cache this peptide if it was already predicted previously.
            if any(
                torch.equal(pred_cached[-1], pred_peptide)
                for pred_cached in pred_cache[spec_idx]
            ):
                # TODO: Add duplicate predictions with their highest score.
                continue
            smx = self.softmax(scores[i : i + 1, : step + 1, :])
            aa_scores = smx[0, range(len(pred_tokens)), pred_tokens].tolist()
            # Add an explicit score 0 for the missing stop token in case this
            # was not predicted (i.e. early stopping).
            if not has_stop_token:
                aa_scores.append(0)
            aa_scores = np.asarray(aa_scores)

            # whh：m/z filter后处理（以及gene的后处理）
            # Calculate the updated amino acid-level and the peptide scores.
            aa_scores, peptide_score = _aa_pep_score(
                aa_scores, beam_fits_precursor[i]
                # ***************************************************whh:被注释的基因组评分***************************************************
                # aa_scores, beam_fits_precursor[i], gene_score[i]
            )
            # # gene基因组的后处理
            # aa_scores, peptide_score = _aa_pep_gene_score(
            #     aa_scores, gene_score[i]
            # )
            # Omit the stop token from the amino acid-level scores.
            aa_scores = aa_scores[:-1]
            # Add the prediction to the cache (minimum priority queue, maximum
            # the number of beams elements).
            if len(pred_cache[spec_idx]) < self.n_beams:
                heapadd = heapq.heappush
            else:
                heapadd = heapq.heappushpop
            heapadd(
                pred_cache[spec_idx],
                (
                    peptide_score,
                    np.random.random_sample(),
                    aa_scores,
                    torch.clone(pred_peptide),
                ),
            )

    def _get_topk_beams(
        self,
        tokens: torch.tensor,
        scores: torch.tensor,
        finished_beams: torch.tensor,
        batch: int,
        step: int,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Find the top-k beams with the highest scores and continue decoding
        those.

        Stop decoding for beams that have been finished.

        Parameters
        ----------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        finished_beams : torch.Tensor of shape (n_spectra * n_beams)
            Boolean tensor indicating whether the current beams are ready for
            caching.
        batch: int
            Number of spectra in the batch.
        step : int
            Index of the next decoding step.

        Returns
        -------
        tokens : torch.Tensor of shape (n_spectra * n_beams, max_length)
            Predicted amino acid tokens for all beams and all spectra.
         scores : torch.Tensor of shape
         (n_spectra *  n_beams, max_length, n_amino_acids)
            Scores for the predicted amino acid tokens for all beams and all
            spectra.
        """
        beam = self.n_beams  # S
        vocab = self.decoder.vocab_size + 1  # V

        # Reshape to group by spectrum (B for "batch").
        tokens = einops.rearrange(tokens, "(B S) L -> B L S", S=beam)
        scores = einops.rearrange(scores, "(B S) L V -> B L V S", S=beam)

        # Get the previous tokens and scores.
        prev_tokens = einops.repeat(
            tokens[:, :step, :], "B L S -> B L V S", V=vocab
        )
        prev_scores = torch.gather(
            scores[:, :step, :, :], dim=2, index=prev_tokens
        )
        prev_scores = einops.repeat(
            prev_scores[:, :, 0, :], "B L S -> B L (V S)", V=vocab
        )

        # Get the scores for all possible beams at this step.
        step_scores = torch.zeros(batch, step + 1, beam * vocab).type_as(
            scores
        )
        step_scores[:, :step, :] = prev_scores
        step_scores[:, step, :] = einops.rearrange(
            scores[:, step, :, :], "B V S -> B (V S)"
        )

        # Find all still active beams by masking out terminated beams.
        active_mask = (
            ~finished_beams.reshape(batch, beam).repeat(1, vocab)
        ).float()
        # Mask out the index '0', i.e. padding token, by default.
        # FIXME: Set this to a very small, yet non-zero value, to only
        # get padding after stop token.
        active_mask[:, :beam] = 1e-8

        # Figure out the top K decodings.
        _, top_idx = torch.topk(step_scores.nanmean(dim=1) * active_mask, beam)
        v_idx, s_idx = np.unravel_index(top_idx.cpu(), (vocab, beam))
        s_idx = einops.rearrange(s_idx, "B S -> (B S)")
        b_idx = einops.repeat(torch.arange(batch), "B -> (B S)", S=beam)

        # Record the top K decodings.
        tokens[:, :step, :] = einops.rearrange(
            prev_tokens[b_idx, :, 0, s_idx], "(B S) L -> B L S", S=beam
        )
        tokens[:, step, :] = torch.tensor(v_idx)
        scores[:, : step + 1, :, :] = einops.rearrange(
            scores[b_idx, : step + 1, :, s_idx], "(B S) L V -> B L V S", S=beam
        )
        scores = einops.rearrange(scores, "B L V S -> (B S) L V")
        tokens = einops.rearrange(tokens, "B L S -> (B S) L")
        return tokens, scores

    def _get_top_peptide(
        self,
        pred_cache: Dict[
            int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ],
    ) -> Iterable[List[Tuple[float, np.ndarray, str]]]:
        """
        Return the peptide with the highest confidence score for each spectrum.

        Parameters
        ----------
        pred_cache : Dict[
                int, List[Tuple[float, float, np.ndarray, torch.Tensor]]
        ]
            Priority queue with finished beams for each spectrum, ordered by
            peptide score. For each finished beam, a tuple with the peptide
            score, a random tie-breaking float, the amino acid-level scores,
            and the predicted tokens is stored.

        Returns
        -------
        pred_peptides : Iterable[List[Tuple[float, np.ndarray, str]]]
            For each spectrum, a list with the top peptide prediction(s). A
            peptide predictions consists of a tuple with the peptide score,
            the amino acid scores, and the predicted peptide sequence.
        """
        for peptides in pred_cache.values():
            if len(peptides) > 0:
                yield [
                    (
                        pep_score,
                        aa_scores,
                        # whh：
                        self.modify_peptide("".join(self.decoder.detokenize(pred_tokens))),
                        # "".join(self.decoder.detokenize(pred_tokens)),
                    )
                    for pep_score, _, aa_scores, pred_tokens in heapq.nlargest(
                        self.top_match, peptides
                    )
                ]
            else:
                yield []

    def _forward_step(
        self,
        spectra: torch.Tensor,
        precursors: torch.Tensor,
        sequences: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward learning step.

        Parameters
        ----------
        spectra : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra for which to predict peptide sequences.
            Axis 0 represents an MS/MS spectrum, axis 1 contains the peaks in
            the MS/MS spectrum, and axis 2 is essentially a 2-tuple specifying
            the m/z-intensity pair for each peak. These should be zero-padded,
            such that all the spectra in the batch are the same length.
        precursors : torch.Tensor of size (n_spectra, 3)
            The measured precursor mass (axis 0), precursor charge (axis 1), and
            precursor m/z (axis 2) of each MS/MS spectrum.
        sequences : List[str] of length n_spectra
            The partial peptide sequences to predict.

        Returns
        -------
        scores : torch.Tensor of shape (n_spectra, length, n_amino_acids)
            The individual amino acid scores for each prediction.
        tokens : torch.Tensor of shape (n_spectra, length)
            The predicted tokens for each spectrum.
        """
        return self.decoder(sequences, precursors, *self.encoder(spectra))

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, List[str]],
        *args,
        mode: str = "train",
    ) -> torch.Tensor:
        """
        A single training step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences as torch Tensors.
        mode : str
            Logging key to describe the current stage.

        Returns
        -------
        torch.Tensor
            The loss of the training step.
        """
        pred, truth = self._forward_step(*batch)
        pred = pred[:, :-1, :].reshape(-1, self.decoder.vocab_size + 1)
        if mode == "train":
            loss = self.celoss(pred, truth.flatten())
        else:
            loss = self.val_celoss(pred, truth.flatten())
        self.log(
            f"{mode}_CELoss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, List[str]], *args
    ) -> torch.Tensor:
        """
        A single validation step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, List[str]]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            peptide sequences.

        Returns
        -------
        torch.Tensor
            The loss of the validation step.
        """
        # Record the loss.
        loss = self.training_step(batch, mode="valid")
        if not self.calculate_precision:
            return loss

        # Calculate and log amino acid and peptide match evaluation metrics from
        # the predicted peptides.
        peptides_pred, peptides_true = [], batch[2]
        for spectrum_preds in self.forward(batch[0], batch[1]):
            for _, _, pred in spectrum_preds:
                peptides_pred.append(pred)

        aa_precision, _, pep_precision = evaluate.aa_match_metrics(
            *evaluate.aa_match_batch(
                peptides_true,
                peptides_pred,
                self.decoder._peptide_mass.masses,
            )
        )
        log_args = dict(on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "Peptide precision at coverage=1",
            pep_precision,
            **log_args,
        )
        self.log(
            "AA precision at coverage=1",
            aa_precision,
            **log_args,
        )
        return loss

    def predict_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], *args
    ) -> List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]:
        """
        A single prediction step.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A batch of (i) MS/MS spectra, (ii) precursor information, (iii)
            spectrum identifiers as torch Tensors.

        Returns
        -------
        predictions: List[Tuple[np.ndarray, float, float, str, float, np.ndarray]]
            Model predictions for the given batch of spectra containing spectrum
            ids, precursor information, peptide sequences as well as peptide
            and amino acid-level confidence scores.
        """
        predictions = []
        for (
            precursor_charge,
            precursor_mz,
            spectrum_i,
            spectrum_preds,
        ) in zip(
            batch[1][:, 1].cpu().detach().numpy(),
            batch[1][:, 2].cpu().detach().numpy(),
            batch[2],
            self.forward(batch[0], batch[1]),
        ):
            for peptide_score, aa_scores, peptide in spectrum_preds:
                predictions.append(
                    (
                        spectrum_i,
                        precursor_charge,
                        precursor_mz,
                        peptide,
                        peptide_score,
                        aa_scores,
                    )
                )

        # whh：这里返回的是对于预测质谱文件的每一张质谱图，得到的对应预测结果（只有一个）
        # 这个predictions是通过forward函数（self.forward）来预测的
        return predictions

    def on_train_epoch_end(self) -> None:
        """
        Log the training loss at the end of each epoch.
        """
        train_loss = self.trainer.callback_metrics["train_CELoss"].detach()
        metrics = {
            "step": self.trainer.global_step,
            "train": train_loss.item(),
        }
        self._history.append(metrics)
        self._log_history()

    def on_validation_epoch_end(self) -> None:
        """
        Log the validation metrics at the end of each epoch.
        """
        callback_metrics = self.trainer.callback_metrics
        metrics = {
            "step": self.trainer.global_step,
            "valid": callback_metrics["valid_CELoss"].detach().item(),
        }

        if self.calculate_precision:
            metrics["valid_aa_precision"] = (
                callback_metrics["AA precision at coverage=1"].detach().item()
            )
            metrics["valid_pep_precision"] = (
                callback_metrics["Peptide precision at coverage=1"]
                .detach()
                .item()
            )
        self._history.append(metrics)
        self._log_history()

    def on_predict_batch_end(
        self,
        outputs: List[Tuple[np.ndarray, List[str], torch.Tensor]],
        *args,
    ) -> None:
        """
        Write the predicted peptide sequences and amino acid scores to the
        output file.
        """
        if self.out_writer is None:
            return
        # Triply nested lists: results -> batch -> step -> spectrum.
        for (
            spectrum_i,
            charge,
            precursor_mz,
            peptide,
            peptide_score,
            aa_scores,
        ) in outputs:
            if len(peptide) == 0:
                continue
            self.out_writer.psms.append(
                (
                    peptide,
                    tuple(spectrum_i),
                    peptide_score,
                    charge,
                    precursor_mz,
                    self.peptide_mass_calculator.mass(peptide, charge),
                    ",".join(list(map("{:.5f}".format, aa_scores))),
                ),
            )

    def _log_history(self) -> None:
        """
        Write log to console, if requested.
        """
        # Log only if all output for the current epoch is recorded.
        if len(self._history) == 0:
            return
        if len(self._history) == 1:
            header = "Step\tTrain loss\tValid loss\t"
            if self.calculate_precision:
                header += "Peptide precision\tAA precision"

            logger.info(header)
        metrics = self._history[-1]
        if metrics["step"] % self.n_log == 0:
            msg = "%i\t%.6f\t%.6f"
            vals = [
                metrics["step"],
                metrics.get("train", np.nan),
                metrics.get("valid", np.nan),
            ]

            if self.calculate_precision:
                msg += "\t%.6f\t%.6f"
                vals += [
                    metrics.get("valid_pep_precision", np.nan),
                    metrics.get("valid_aa_precision", np.nan),
                ]

            logger.info(msg, *vals)
            if self.tb_summarywriter is not None:
                for descr, key in [
                    ("loss/train_crossentropy_loss", "train"),
                    ("loss/val_crossentropy_loss", "valid"),
                    ("eval/val_pep_precision", "valid_pep_precision"),
                    ("eval/val_aa_precision", "valid_aa_precision"),
                ]:
                    metric_value = metrics.get(key, np.nan)
                    if not np.isnan(metric_value):
                        self.tb_summarywriter.add_scalar(
                            descr, metric_value, metrics["step"]
                        )

    def configure_optimizers(
        self,
    ) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
        """
        Initialize the optimizer.

        This is used by pytorch-lightning when preparing the model for training.

        Returns
        -------
        Tuple[torch.optim.Optimizer, Dict[str, Any]]
            The initialized Adam optimizer and its learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), **self.opt_kwargs)
        # Apply learning rate scheduler per step.
        lr_scheduler = CosineWarmupScheduler(
            optimizer, self.warmup_iters, self.cosine_schedule_period_iters
        )
        return [optimizer], {"scheduler": lr_scheduler, "interval": "step"}

    def modify_peptide(self, seq):
        # 确保有传基因组数据
        if len(self.gene_sequence) <= 0:
            return seq
        return self.Smith_Waterman(seq)

    def Smith_Waterman(self, origin_query):
        if not origin_query:
            return origin_query

        # 对序列的修饰清洗(记得在此处添加美元符号)
        query, modifications = self.extract_modifications('$' + origin_query)
        # 去除开头的$符号
        query = self.remove_dollar(query)

        # 计算总质量(此处没有考虑脱水缩合)
        total_mass = self.calculate_total_mass(query)
        # 搜索最佳匹配
        best_alignment = None
        best_score = 0
        best_index = -1  # 用于记录最佳比对的序列索引，从0开始

        # 定义评分标准
        match_score = 2  # 匹配正确的得分
        mismatch_penalty = -1  # 错配扣分
        gap_open_penalty = -5  # 插入扣分
        gap_extend_penalty = -5  # 增添扣分

        # 计算最大可能得分（假设两个序列完全匹配）
        max_score = len(query) * match_score

        for index, sequence in enumerate(self.gene_sequence):
            alignments = pairwise2.align.localms(query, sequence, match_score, mismatch_penalty, gap_open_penalty,
                                                 gap_extend_penalty)
            for alignment in alignments:
                score = alignment[2]
                # # 低分直接跳过
                # if score < max_score * 0.5:
                #     continue
                if score > best_score:
                    # 说明完全匹配，不用再修改了
                    if score == max_score:
                        return origin_query
                    best_alignment = alignment
                    best_score = score
                    best_index = index


        # 打印最佳匹配结果
        if best_alignment:
            alignment_res = format_alignment(*best_alignment)
            # 对最佳匹配结果alignment_res鉴定，看看是否有空格（间隙）
            gap_num = self.find_gaps_from_alignment(alignment_res)
            # 找query匹配的开头位置
            match = re.search(r'\d+', alignment_res.strip().split('\n')[0].strip())
            startA = 0
            if match:
                startA = int(match.group(0)) - 1 - gap_num  # 注意这里-1才是起始位置
            correction_A = query
            if self.get_sequence_segment(best_index, best_alignment.start, startA, len(query)):
                correction_A = self.gene_sequence[best_index][best_alignment.start - startA:best_alignment.start - startA + len(query)]

            # 计算质量是否超过误差
            total_mass_after_revise = self.calculate_total_mass(correction_A)
            if abs(total_mass - total_mass_after_revise) > 1:
                return origin_query
            # 对纠正后的结果添加回修饰
            corrected_seq_with_mods = self.reapply_modifications('$' + correction_A, modifications)
            # 最后记得把$符号去掉
            return self.remove_dollar(corrected_seq_with_mods)
        else:
            return origin_query

    @staticmethod
    def extract_modifications(seq):
        pattern = re.compile(r'([\w$])([+-]\d+\.\d+)')
        modifications = []
        # amino_acid_counts = {}  # 用于跟踪每种氨基酸出现的次数

        for match in pattern.finditer(seq):
            # 氨基酸出现的次数默认为1
            aa_times = 0
            # 第一个捕获组的文本
            amino_acid = match.group(1)
            # 第二个捕获组的文本
            modification = match.group(2)
            # 更新氨基酸出现的次数
            for i in range(match.start(1), -1, -1):
                if amino_acid == seq[i]:
                    aa_times += 1
            modifications.append((amino_acid, modification, aa_times))

        # 定位结束后，移除修饰信息
        pattern_modify = re.compile(r'[+-]\d+\.\d+')
        clean_seq = re.sub(pattern_modify, '', seq)

        return clean_seq, modifications

    @staticmethod
    def remove_dollar(seq):
        return seq.replace('$', '', 1)

    @staticmethod
    def find_gaps_from_alignment(alignment):
        # 分行,其中第二行就是中间的匹配信息行
        match_line = alignment.split('\n')[1].strip()
        # 去除开头的空格
        gaps_counter = 0
        for char in match_line:
            if char == ' ':
                gaps_counter += 1

        return gaps_counter

    def get_sequence_segment(self, index, begin, query_start, query_length):
        # 检查索引是否在列表范围内
        if index < 0 or index >= len(self.gene_sequence):
            return False
        # 获取指定索引的序列
        sequence = self.gene_sequence[index]
        # 检查起始位置和长度是否有效
        if begin < 0 or begin >= len(sequence):
            return False
        if query_length < 0 or begin + query_length > len(sequence):
            return False

        # 返回指定的序列段
        return True

    @staticmethod
    def reapply_modifications(seq, modifications):
        modified_seq = seq  # 初始时，矫正后的序列就是最终序列
        for amino_acid, modification, aa_times in modifications:
            if amino_acid == '$':  # 特殊处理序列开头的修饰
                modified_seq = modified_seq[0] + modification + modified_seq[1:]
            else:
                # 找到第 aa_times 次出现的氨基酸的位置
                count = 0
                for i, char in enumerate(modified_seq):
                    if char == amino_acid:
                        count += 1
                        if count == aa_times:
                            # 在该氨基酸前面添加修饰
                            modified_seq = modified_seq[:i + 1] + modification + modified_seq[i + 1:]
                            break
        return modified_seq

    def calculate_total_mass(self, query):
        self.peptide_mass_calculator.masses['C'] = 103.009184505
        return sum(self.peptide_mass_calculator.masses[aa] for aa in query)


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler with linear warm-up followed by cosine shaped decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer object.
    warmup_iters : int
        The number of iterations for the linear warm-up of the learning rate.
    cosine_schedule_period_iters : int
        The number of iterations for the cosine half period of the learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_iters: int,
        cosine_schedule_period_iters: int,
    ):
        self.warmup_iters = warmup_iters
        self.cosine_schedule_period_iters = cosine_schedule_period_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (
            1 + np.cos(np.pi * epoch / self.cosine_schedule_period_iters)
        )
        if epoch <= self.warmup_iters:
            lr_factor *= epoch / self.warmup_iters
        return lr_factor


def _calc_mass_error(
    calc_mz: float, obs_mz: float, charge: int, isotope: int = 0
) -> float:
    """
    Calculate the mass error in ppm between the theoretical m/z and the observed
    m/z, optionally accounting for an isotopologue mismatch.

    Parameters
    ----------
    calc_mz : float
        The theoretical m/z.
    obs_mz : float
        The observed m/z.
    charge : int
        The charge.
    isotope : int
        Correct for the given number of C13 isotopes (default: 0).

    Returns
    -------
    float
        The mass error in ppm.
    """
    return (calc_mz - (obs_mz - isotope * 1.00335 / charge)) / obs_mz * 10**6


def _aa_pep_score(
    aa_scores: np.ndarray, fits_precursor_mz: bool
    # ***************************************************whh:被注释的基因组评分***************************************************
    # aa_scores: np.ndarray, fits_precursor_mz: bool, gene_score: float
) -> Tuple[np.ndarray, float]:
    """
    Calculate amino acid and peptide-level confidence score from the raw amino
    acid scores.

    The peptide score is the mean of the raw amino acid scores. The amino acid
    scores are the mean of the raw amino acid scores and the peptide score.

    Parameters
    ----------
    aa_scores : np.ndarray
        Amino acid level confidence scores.
    fits_precursor_mz : bool
        Flag indicating whether the prediction fits the precursor m/z filter.

    Returns
    -------
    aa_scores : np.ndarray
        The amino acid scores.
    peptide_score : float
        The peptide score.
    """
    peptide_score = np.mean(aa_scores)
    aa_scores = (aa_scores + peptide_score) / 2
    if not fits_precursor_mz:
        peptide_score -= 1
    # ***************************************************whh:被注释的基因组评分***************************************************
    # peptide_score -= gene_score
    return aa_scores, peptide_score

# def _aa_pep_gene_score(
#     aa_scores: np.ndarray, gene_score: float
# ):


# whh：基因组评分
def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # 删除
                                   dp[i][j - 1],  # 插入
                                   dp[i - 1][j - 1])  # 替换
    return dp[m][n]


def calculate_match_score(predicted_peptide, genome_sequence):
    max_length = len(predicted_peptide)
    best_score = 0
    best_distance = float('inf')

    # 遍历基因组序列，寻找最佳匹配
    for i in range(len(genome_sequence) - len(predicted_peptide) + 1):
        subseq = genome_sequence[i:i + len(predicted_peptide)]
        distance = edit_distance(predicted_peptide, subseq)
        current_score = (1 - distance / max_length) * 100
        best_score = max(best_score, current_score)
        best_distance = min(best_distance, distance)

    return 1 - best_score / 100


# 这里的sequences是整个基因组序列
def calculate_smith_waterman_score(sequences, query):
    if not query:
        # print(f"Normalized Score: {0:.5f}")
        return 0
    # 搜索最佳匹配
    best_alignment = None
    best_score = 0
    best_normalized_score = 0

    # 定义评分标准
    match_score = 2  # 匹配正确的得分
    mismatch_penalty = -1  # 错配扣分
    gap_open_penalty = -5  # 插入扣分
    gap_extend_penalty = -5  # 增添扣分

    # 计算最大可能得分（假设两个序列完全匹配）
    max_score = len(query) * match_score

    # 计算最小可能得分（假设两个序列完全不同）
    # min_score = len(query_sequence) * (
    #             -mismatch_penalty + gap_open_penalty + (len(query_sequence) - 1) * gap_extend_penalty)
    min_score = 0

    for sequence in sequences:
        alignments = pairwise2.align.localms(query, sequence, match_score, mismatch_penalty, gap_open_penalty, gap_extend_penalty)
        for alignment in alignments:
            score = alignment[2]
            # 计算归一化得分
            normalized_score = (score - min_score) / (max_score - min_score)
            if score > best_score:
                best_alignment = alignment
                best_score = score
                best_normalized_score = normalized_score

    # 打印最佳匹配结果
    if best_alignment:
        # print(format_alignment(*best_alignment))
        # print(f"Score: {best_score}")
        # print(f"Normalized Score: {best_normalized_score:.5f}")
        return best_normalized_score
    else:
        # print("No significant alignment found.")
        return 0
