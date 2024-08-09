# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    RobertaForSequenceClassification,
)


@dataclass
class ScoPECriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")
    target_domain: str = field(
        default="",
        metadata={"help": "target domain"},
    )
    sent_disc: bool = field(
        default=False,
        metadata={
            "help": "if true, sentiment discriminator logits are added to scores"
        },
    )
    skip_rep_score: bool = field(
        default=False,
        metadata={"help": "if true, rep score is not added to scores"},
    )
    skip_mlm_score: bool = field(
        default=False,
        metadata={"help": "if true, mlm score is not added to scores"},
    )
    edit_iter: int = field(
        default=2,
        metadata={"help": "iteration steps for edit during training"},
    )
    rep_scale: float = field(
        default=10.0,
        metadata={"help": "repetition score scale"},
    )
    disc_scale: float = field(
        default=1.0,
        metadata={"help": "discriminator score scale"},
    )


@register_criterion("scope_loss", dataclass=ScoPECriterionConfig)
class ScoPECriterion(FairseqCriterion):
    def __init__(self, cfg: ScoPECriterionConfig, task):
        super().__init__(task)
        self.sentence_avg = cfg.sentence_avg
        self.target_domain = cfg.target_domain
        self.sent_disc = cfg.sent_disc
        self.skip_rep_score = cfg.skip_rep_score
        self.skip_mlm_score = cfg.skip_mlm_score
        self.edit_iter = cfg.edit_iter
        self.rep_scale = cfg.rep_scale
        self.disc_scale = cfg.disc_scale
        self.ebm = None
        self.plm = None
        self.disc = None
        self.roberta_to_gpt_emb = None

        self.gpu = None
        self.fp16 = None
        self.tokenizer_roberta = None
        self.tokenizer_gpt = None

    def load_models(self, gpu, fp16):
        self.gpu = gpu
        self.fp16 = fp16
        path_ebm = "./roberta_ft_" + self.target_domain + "_hf/"

        if fp16:
            self.ebm = AutoModelForMaskedLM.from_pretrained(
                path_ebm, torch_dtype=torch.float16
            )
            if gpu:
                self.ebm = self.ebm.to("cuda")
        else:
            self.ebm = AutoModelForMaskedLM.from_pretrained(path_ebm)
            if gpu:
                self.ebm = self.ebm.to("cuda")
        self.ebm = self.ebm.eval()

        if self.sent_disc:
            path_disc = "siebert/sentiment-roberta-large-english"

            if fp16:
                self.disc = RobertaForSequenceClassification.from_pretrained(
                    path_disc, torch_dtype=torch.float16
                )
                if gpu:
                    self.disc = self.disc.to("cuda")
            else:
                self.disc = RobertaForSequenceClassification.from_pretrained(path_disc)
                if gpu:
                    self.disc = self.disc.to("cuda")
            self.disc = self.disc.eval()

        return

    def build_mapping_emb(self):
        self.tokenizer_roberta = AutoTokenizer.from_pretrained("roberta-base")
        self.tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")

        dict_roberta = self.tokenizer_roberta.vocab
        dict_gpt = self.tokenizer_gpt.vocab

        keys_gpt = []
        for k, v in dict_gpt.items():
            keys_gpt.append(k)

        keys_roberta = []
        for k, v in dict_roberta.items():
            keys_roberta.append(k)

        mapping_weight_roberta = torch.ones([len(dict_roberta), 1]) * 50256
        for k in keys_gpt:
            mapping_weight_roberta[dict_roberta[k]] = dict_gpt[k]
        roberta_to_gpt_emb = torch.nn.Embedding(
            len(dict_roberta), 1, _weight=mapping_weight_roberta
        )

        if self.gpu:
            self.roberta_to_gpt_emb = roberta_to_gpt_emb.cuda()
        else:
            self.roberta_to_gpt_emb = roberta_to_gpt_emb

        return

    def raw_logits_per_block_pos(self, tokens, cal_mask):
        if self.gpu:
            device = "cuda"
        else:
            device = "cpu"

        input_ids = tokens.detach()
        attn_mask = (input_ids != self.tokenizer_roberta.pad_token_id).type(torch.int64)

        b_size = cal_mask.sum(dim=-1).type(torch.int64)
        cal_idx_row = cal_mask.nonzero(as_tuple=True)[0]
        cal_idx_col = cal_mask.nonzero(as_tuple=True)[1]

        input_ids_rep = input_ids.repeat_interleave(b_size, dim=0)
        input_ids_rep[torch.arange(input_ids_rep.size(0)), cal_idx_col] = (
            self.tokenizer_roberta.mask_token_id
        )
        attn_mask_rep = attn_mask.repeat_interleave(b_size, dim=0)

        with torch.no_grad():
            logits = self.ebm.eval()(input_ids_rep, attn_mask_rep).logits.float()

        phi_outputs = torch.zeros([input_ids.size(0), input_ids.size(1)], device=device)
        phi_outputs[cal_idx_row, cal_idx_col] = logits[
            torch.arange(input_ids_rep.size(0)),
            cal_idx_col,
            input_ids[cal_idx_row, cal_idx_col],
        ]

        return phi_outputs[:, :]

    def cal_rep_score(self, tokens, cal_mask):
        """
        'tokens' : (B, T) token batch tensor
        'cal_mask' : (B, T) float mask for position to calculate rep score
        'cal_idx' : (k, 1) indices of each sample corresponding to tokens for score calculation
        'cal_per_sample' : (B,) number of tokens for score calculation in each sample

        return 'rep_score', (B, T) tensor that contains repetition score for each tokens for calculation(else 0)
        """
        if self.gpu:
            device = "cuda"
        else:
            device = "cpu"

        scale = self.rep_scale

        cal_per_sample = cal_mask.sum(dim=-1).type(torch.int64)
        cal_idx = cal_mask.nonzero(as_tuple=True)[1].unsqueeze(1)

        row_idx = cal_mask.nonzero(as_tuple=True)[0]

        cal_tokens = tokens[row_idx, cal_idx[:, 0]].unsqueeze(1)

        mask_same = (
            cal_tokens == tokens.repeat_interleave(cal_per_sample, dim=0)
        ).float()
        mask_same[torch.arange(mask_same.size(0)), cal_idx[:, 0]] = 0

        idx_dist = (
            cal_idx
            - torch.arange(tokens.size(1), device=device).repeat(cal_tokens.size(0), 1)
        ).abs()
        idx_dist[torch.arange(idx_dist.size(0)), cal_idx[:, 0]] = -1
        idx_dist = scale / idx_dist
        rep_score_1d = (idx_dist * mask_same).sum(dim=-1)

        rep_score = torch.zeros([tokens.size(0), tokens.size(1)], device=device)
        rep_score[row_idx, cal_idx[:, 0]] = rep_score_1d

        return rep_score

    def cal_disc_score(self, tokens):

        if self.gpu:
            device = "cuda"
        else:
            device = "cpu"

        scale = self.disc_scale

        attn_mask = (tokens != self.tokenizer_roberta.pad_token_id).type(torch.long)

        with torch.no_grad():
            logits = self.disc.eval()(tokens, attn_mask)[0]

        if self.target_domain == "pos":
            disc_score = logits[:, 1]
            disc_score = disc_score.unsqueeze(1).repeat(1, tokens.size(1))
        elif self.target_domain == "camera":
            disc_score = logits[:, 1]
            disc_score = disc_score.unsqueeze(1).repeat(1, tokens.size(1))
        elif self.target_domain == "neg":
            disc_score = logits[:, 0]
            disc_score = disc_score.unsqueeze(1).repeat(1, tokens.size(1))
        else:
            disc_score = torch.zeros([tokens.size(0), tokens.size(1)], device=device)

        disc_score *= scale

        return disc_score

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # net_output = model(**sample["net_input"])
        (
            loss,
            rewards,
            b_size,
            equalities,
            ener_diffs,
            rep_diffs,
            disc_diffs,
            grad_cum_stack,
            iter_step,
        ) = self.compute_loss(model, sample, update_num, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "b_size": b_size,
            "reward": rewards[0],
            "equality": equalities[0],
            "energy_diff": ener_diffs[0],
            "rep_diff": rep_diffs[0],
            "disc_diff": disc_diffs[0],
            "grad_cum_stack": grad_cum_stack,
        }
        log_multi = {
            "reward": rewards,
            "equality": equalities,
            "energy_diff": ener_diffs,
            "rep_diff": rep_diffs,
            "disc_diff": disc_diffs,
        }

        if iter_step > 1:
            for i in range(1, iter_step):
                for key in log_multi.keys():
                    key2 = key + str(i + 1)
                    logging_output[key2] = log_multi[key][i]

        return loss, sample_size, logging_output

    def compute_loss(self, model, sample, update_num=0, reduce=True):
        if self.gpu:
            device = "cuda"
        else:
            device = "cpu"

        grad_cum_stack = 1
        iter_step = self.edit_iter

        losses = []
        reward_logs = []
        ener_diffs = []
        rep_diffs = []
        disc_diffs = []
        equalities = []

        inputs = (
            sample["net_input"]["src_tokens"] * (sample["target"] == 1).int()
            + sample["target"] * (sample["target"] != 1).int()
        )

        bsz, seq_len = inputs.size()

        inp_toks = torch.concat([inputs[:, :-2], inputs[:, -1].unsqueeze(1)], dim=-1)
        b_sizes = (
            self.roberta_to_gpt_emb(inputs[:, -2])
            .detach()
            .type(torch.int64)
            .squeeze(dim=-1)
        )
        b_sizes_cumsum = b_sizes.cumsum(dim=0)
        b_sizes_cumsum_rep = b_sizes_cumsum.repeat_interleave(b_sizes, dim=0)

        block_idx_row = torch.arange(bsz, device=device).repeat_interleave(
            b_sizes, dim=0
        )
        block_idx_col = (
            -b_sizes_cumsum_rep + torch.arange(b_sizes_cumsum[-1], device=device) - 1
        )
        origin_toks = inp_toks[block_idx_row, block_idx_col].clone()

        for i in range(iter_step):
            logits = model(inp_toks)[0]
            logits_ener = logits
            logits_ener[:, :, :3] = -1e2
            probs_ener = F.softmax(logits_ener.float(), dim=-1).detach()
            sampled_argmax = torch.argmax(probs_ener, dim=-1).detach()
            sampled_argmax[:, -2] = inp_toks[:, -2].clone()
            sampled_argmax = sampled_argmax[block_idx_row, block_idx_col]
            sampler = torch.distributions.categorical.Categorical(logits=logits_ener)
            sampled = sampler.sample()[block_idx_row, block_idx_col]
            probs_ener = probs_ener[block_idx_row, block_idx_col, sampled]

            equality = ((sampled_argmax == origin_toks).float().sum()) / b_sizes_cumsum[
                -1
            ]

            mask_for_changed = (sampled != origin_toks).float()
            sampled_toks = inp_toks.clone()
            sampled_toks[block_idx_row, block_idx_col] = sampled
            sampled_toks[:, -2] = inp_toks[:, -2].clone()
            sampled = sampled_toks[block_idx_row, block_idx_col]
            mask_for_changed_2d = (sampled_toks != inp_toks).float()

            mask_for_all_block = torch.zeros(
                [inp_toks.size(0), inp_toks.size(1)], device=device
            )
            mask_for_all_block[block_idx_row, block_idx_col] = 1.0

            # mask_for_changed_2d = mask_for_all_block  # 230825 analysis for all position training

            if model.training:
                phi_per_pos_origin = self.raw_logits_per_block_pos(
                    inp_toks, mask_for_changed_2d
                )
                phi_per_pos_sampled = self.raw_logits_per_block_pos(
                    sampled_toks, mask_for_changed_2d
                )

            else:
                phi_per_pos_origin = self.raw_logits_per_block_pos(
                    inp_toks, mask_for_all_block
                )
                phi_per_pos_sampled = self.raw_logits_per_block_pos(
                    sampled_toks, mask_for_all_block
                )

            if self.skip_rep_score:
                rep_score_origin = torch.zeros(
                    [inp_toks.size(0), inp_toks.size(1)], device=device
                )
                rep_score_sampled = torch.zeros(
                    [sampled_toks.size(0), sampled_toks.size(1)], device=device
                )

            else:
                rep_score_origin = self.cal_rep_score(inp_toks, mask_for_all_block)
                rep_score_sampled = self.cal_rep_score(sampled_toks, mask_for_all_block)

            if self.sent_disc:
                disc_score_origin = self.cal_disc_score(inp_toks)
                disc_score_sampled = self.cal_disc_score(sampled_toks)
            else:
                disc_score_origin = torch.zeros(
                    [inp_toks.size(0), inp_toks.size(1)], device=device
                )
                disc_score_sampled = torch.zeros(
                    [sampled_toks.size(0), sampled_toks.size(1)], device=device
                )

            rewards = phi_per_pos_sampled - phi_per_pos_origin
            rewards_rep = -(rep_score_sampled - rep_score_origin)
            rewards_disc = disc_score_sampled - disc_score_origin
            energy_diff = rewards.sum(dim=-1).mean()
            rep_diff = rewards_rep.sum(dim=-1).mean()
            disc_diff = rewards_disc[:, 0].mean()
            rewards += rewards_rep + rewards_disc
            rewards = rewards[block_idx_row, block_idx_col] * mask_for_changed
            rewards_2d = torch.zeros(
                [inp_toks.size(0), inp_toks.size(1)], device=device
            )
            rewards_2d[block_idx_row, block_idx_col] = rewards
            rewards_2d_clipped = rewards_2d.clip(min=-10.0, max=10.0)
            rewards = rewards_2d_clipped[block_idx_row, block_idx_col]
            rewards = rewards.detach()

            rewards_for_log = rewards[(rewards != 0.0)].clone()
            if len(rewards_for_log) == 0:
                rewards_for_log = torch.tensor(0, device=device)
            else:
                rewards_for_log = rewards_for_log.mean()

            lprobs_ener_total = F.log_softmax(
                logits_ener[block_idx_row, block_idx_col].float(), dim=-1
            )
            lprobs_ener = lprobs_ener_total[
                torch.arange(lprobs_ener_total.size(0)), sampled
            ]

            loss = rewards * probs_ener * lprobs_ener
            loss = -loss.sum()

            inp_toks = sampled_toks.detach()
            origin_toks = sampled.detach()

            equalities.append(equality)
            ener_diffs.append(energy_diff.detach())
            rep_diffs.append(rep_diff.detach())
            disc_diffs.append(disc_diff.detach())
            reward_logs.append(rewards_for_log)
            losses.append(loss)

        loss = sum(losses) / iter_step

        return (
            loss,
            reward_logs,
            b_sizes_cumsum[-1],
            equalities,
            ener_diffs,
            rep_diffs,
            disc_diffs,
            grad_cum_stack,
            int(iter_step),
        )

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        metrics_dict = {}
        keys_filter = ["reward", "equality", "energy_diff", "rep_diff", "disc_diff"]
        for key in logging_outputs[0].keys():
            metrics_dict[key] = sum(log.get(key, 0) for log in logging_outputs)

        b_size = metrics_dict["b_size"]
        sample_size = metrics_dict["sample_size"]
        ntokens = metrics_dict["ntokens"]

        for key in metrics_dict.keys():
            norm_scale = metrics_dict["grad_cum_stack"]
            if key == "loss":
                norm_scale = b_size
                metrics.log_scalar(
                    key,
                    metrics_dict[key] / norm_scale / math.log(2),
                    norm_scale,
                    round=3,
                )

            rec = False
            for kf in keys_filter:
                rec += key[:-1] in kf

            if rec > 0:
                metrics.log_scalar(
                    key, metrics_dict[key] / norm_scale, norm_scale, round=3
                )

        if sample_size != ntokens:
            # we divide by log(2) to convert the loss from base e to base 2
            metrics.log_scalar(
                "nll_loss",
                metrics_dict["loss"] / ntokens / math.log(2),
                ntokens,
                round=3,
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
