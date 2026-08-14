#!/usr/bin/env python3
"""Compare a reproducible scratch initialization with a bounded checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


EVALUATIONS = {
    (512, 128, 770011): {
        "checkpoint": Path("artifacts/x2_scratch_512_pilot_seed770011"),
        "checkpoint_sha256": "a6c3062b8e8e5adec31e8b7366723ad42c36e73a520944d7f23413ca30d9ac4b",
        "latent_seed": 770012,
        "report": None,
    },
    (512, 400, 770023): {
        "checkpoint": Path("artifacts/x2_scratch_512_half_million_seed770023"),
        "checkpoint_sha256": "a4fc463cd303c754a84dc4b75df3a897a964cfd0228ab8b783425d3a0a28c811",
        "latent_seed": 770024,
        "report": Path("reports/x2_scratch_512_half_million_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_512_half_million_expert_latent_screen.json"
        ),
    },
    (512, 400, 770031): {
        "checkpoint": Path(
            "artifacts/x2_scratch_512_half_million_state71_seed770031"
        ),
        "checkpoint_sha256": "130c4da4f7483ee1a0b514377a603b3108709f66e7a8ddf514aabddb2b6a9d2f",
        "latent_seed": 770032,
        "report": Path("reports/x2_scratch_512_half_million_state71_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_512_half_million_state71_expert_latent_screen.json"
        ),
    },
    (512, 400, 770041): {
        "checkpoint": Path(
            "artifacts/x2_scratch_512_half_million_expert_rollout_seed770041"
        ),
        "checkpoint_sha256": "8ab06bf148bf01ef8618f386f8b51193b239188661c861c4c219b940af4054eb",
        "latent_seed": 770042,
        "report": Path("reports/x2_scratch_512_half_million_expert_rollout_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_512_half_million_expert_rollout_latent_screen.json"
        ),
    },
    (512, 400, 770051): {
        "checkpoint": Path("artifacts/x2_scratch_state71_cont1_seed770051"),
        "checkpoint_sha256": "81e85ee3f2a52a2ea572874effbb64e5d67ef709aee448d019e3b392b868f6da",
        "latent_seed": 770052,
        "report": Path("reports/x2_scratch_state71_cont1_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_state71_cont1_expert_latent_screen.json"
        ),
    },
    (512, 400, 770061): {
        "checkpoint": Path("artifacts/x2_scratch_terminal_aware_seed770061"),
        "checkpoint_sha256": "ef918e67b59d098a7bafbb94de8a4c0e19fd96e3e7303784b0b13093aa0520d5",
        "latent_seed": 770062,
        "report": Path("reports/x2_scratch_terminal_aware_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_terminal_aware_expert_latent_screen.json"
        ),
    },
    (512, 400, 770071): {
        "checkpoint": Path("artifacts/x2_scratch_state71_safety_ft_seed770071"),
        "checkpoint_sha256": "ea85be660ca33f5e7a72d6d148640c21b18530ea35963cd7a03f40f00af11ebc",
        "latent_seed": 770072,
        "report": Path("reports/x2_scratch_state71_safety_ft_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_state71_safety_ft_survival_audit.json"
        ),
    },
    (512, 400, 770081): {
        "checkpoint": Path(
            "artifacts/x2_scratch_state71_taskreward_ft_seed770081"
        ),
        "checkpoint_sha256": "5a56b0b7963ac7e636cff64b242b166aa8a39f9eef79b150d2028bf711b79800",
        "latent_seed": 770082,
        "report": Path("reports/x2_scratch_state71_taskreward_ft_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_state71_taskreward_ft_survival_audit.json"
        ),
    },
    (512, 400, 770091): {
        "checkpoint": Path("artifacts/x2_scratch_closed_loop_expert_seed770091"),
        "checkpoint_sha256": "9585582c2e497f9b9128f08a8c3f7bf22d54b2152da405d05443349f844bd963",
        "latent_seed": 770092,
        "report": Path("reports/x2_scratch_closed_loop_expert_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_closed_loop_expert_survival_audit.json"
        ),
    },
    (512, 400, 770101): {
        "checkpoint": Path("artifacts/x2_scratch_closed_loop_bc_seed770101"),
        "checkpoint_sha256": "2928190a78d54c14b59d5fe408da9c348315646fa60a6f3c98d5356d6af9699d",
        "latent_seed": 770102,
        "report": Path("reports/x2_scratch_closed_loop_bc_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_closed_loop_bc_survival_audit.json"
        ),
    },
    (512, 400, 770111): {
        "checkpoint": Path(
            "artifacts/x2_scratch_closed_loop_bc_fixed_z_seed770111"
        ),
        "checkpoint_sha256": "bd185935d181823b31cfaa5f98553b4e1a8f352a75c4f434d1b2f4aaacd72f28",
        "latent_seed": 770112,
        "report": Path("reports/x2_scratch_closed_loop_bc_fixed_z_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_closed_loop_bc_fixed_z_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_fixed_z.json"
        ),
        "canonical_latent_report_sha256": "00247c7d84b6654b47945bed90638ae16fc52ad5fe164692b0f0bb16a5b6e097",
    },
    (512, 400, 770121): {
        "checkpoint": Path("artifacts/x2_scratch_dagger1_seed770121"),
        "checkpoint_sha256": "977333c44f70d6c2d79f3ca67e93c9dd7d108aaa7bf68eda83281b3711983388",
        "latent_seed": 770122,
        "report": Path("reports/x2_scratch_dagger1_eval.json"),
        "expert_report": Path("reports/x2_scratch_dagger1_survival_audit.json"),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_fixed_z.json"
        ),
        "canonical_latent_report_sha256": "00247c7d84b6654b47945bed90638ae16fc52ad5fe164692b0f0bb16a5b6e097",
    },
    (512, 400, 770131): {
        "checkpoint": Path("artifacts/x2_scratch_dagger2_seed770131"),
        "checkpoint_sha256": "e7fb10bf5077b54326a6ab52f49ae90b8b73c70c96b5464a1cc3186897bdc326",
        "latent_seed": 770132,
        "report": Path("reports/x2_scratch_dagger2_eval.json"),
        "expert_report": Path("reports/x2_scratch_dagger2_survival_audit.json"),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_fixed_z.json"
        ),
        "canonical_latent_report_sha256": "00247c7d84b6654b47945bed90638ae16fc52ad5fe164692b0f0bb16a5b6e097",
    },
    (512, 400, 770151): {
        "checkpoint": Path(
            "artifacts/x2_scratch_closed_loop_bc_physical_v2_seed770151"
        ),
        "checkpoint_sha256": "6225b09a76efac4799d0cd38ecd7d366b12e6461b65cd1bff3ce96be2d93e413",
        "latent_seed": 770152,
        "report": Path("reports/x2_scratch_closed_loop_bc_physical_v2_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_closed_loop_bc_physical_v2_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_physical_v2.json"
        ),
        "canonical_latent_report_sha256": "3f17f9016d934595eb794d602df33b0b68d68369c768717b3ac7262c71d8cd2c",
        "source_domain_contract": True,
    },
    (512, 400, 770181): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_observation_v3_seed770181"
        ),
        "checkpoint_sha256": "b7191f3c3438c697e4d9db215dba00e57b6d62000448ea8f3ecc473d7fc25fb1",
        "latent_seed": 770182,
        "report": Path("reports/x2_scratch_dagger_observation_v3_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_observation_v3_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_observation_v3.json"
        ),
        "canonical_latent_report_sha256": "dd1a4d123ac0ad62861adb4372b29cae0cdaae4ea0782530bd315766583726a1",
        "source_domain_contract": True,
    },
    (512, 400, 770191): {
        "checkpoint": Path(
            "artifacts/x2_scratch_closed_loop_bc_joint_v4_seed770191"
        ),
        "checkpoint_sha256": "8c16bf2c240162d1b7d571370011f7ea93312b235b035b6cc64945c0ebfedc02",
        "latent_seed": 770192,
        "report": Path("reports/x2_scratch_closed_loop_bc_joint_v4_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "source_domain_contract": True,
    },
    (512, 400, 770201): {
        "checkpoint": Path("artifacts/x2_scratch_dagger_joint_v4_seed770201"),
        "checkpoint_sha256": "c3d9ff0ad08022eadc03a04cc5df3b8a4db3a7dafaddd8b1f3fd2342b129120a",
        "latent_seed": 770202,
        "report": Path("reports/x2_scratch_dagger_joint_v4_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_joint_v4_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "source_domain_contract": True,
    },
    (512, 400, 770211): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_joint_v4_round2_seed770211"
        ),
        "checkpoint_sha256": "a0e91b7dc7d706099fa306fe1253077f50243c5e934b79f35b7f126afdcb9bb1",
        "latent_seed": 770212,
        "report": Path("reports/x2_scratch_dagger_joint_v4_round2_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_joint_v4_round2_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "source_domain_contract": True,
    },
    (512, 1000, 770221): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_joint_v4_round2_seed770211"
        ),
        "checkpoint_sha256": "a0e91b7dc7d706099fa306fe1253077f50243c5e934b79f35b7f126afdcb9bb1",
        "latent_seed": 770222,
        "report": Path("reports/x2_scratch_dagger_joint_v4_round2_long_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_joint_v4_round2_long_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "source_domain_contract": True,
    },
    (512, 400, 770231): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_joint_v4_round2_seed770211"
        ),
        "checkpoint_sha256": "a0e91b7dc7d706099fa306fe1253077f50243c5e934b79f35b7f126afdcb9bb1",
        "latent_seed": 770232,
        "report": Path("reports/x2_scratch_dagger_joint_v4_round2_response_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_joint_v4_round2_response_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "response_domain_contract": True,
    },
    (512, 400, 770241): {
        "checkpoint": Path("artifacts/x2_scratch_dagger_response_v5_seed770241"),
        "checkpoint_sha256": "1843dd53c80ede2d82be95c307be0c7a21709d14e3bd12444e0f3d693fd2c006",
        "latent_seed": 770242,
        "report": Path("reports/x2_scratch_dagger_response_v5_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_response_v5_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "response_domain_contract": True,
    },
    (512, 400, 770251): {
        "checkpoint": Path("artifacts/x2_scratch_dagger_response_v5_seed770241"),
        "checkpoint_sha256": "1843dd53c80ede2d82be95c307be0c7a21709d14e3bd12444e0f3d693fd2c006",
        "latent_seed": 770252,
        "report": Path("reports/x2_scratch_dagger_response_v5_ideal_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_response_v5_ideal_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "source_domain_contract": True,
    },
    (512, 400, 770261): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_response_v5_round2_seed770261"
        ),
        "checkpoint_sha256": "8e0c558007770787974e5d0a37e6ddf25c25875f082a653bcab24f513b1f71e3",
        "latent_seed": 770262,
        "report": Path("reports/x2_scratch_dagger_response_v5_round2_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_response_v5_round2_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "response_domain_contract": True,
    },
    (512, 400, 770271): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_response_v5_round2_seed770261"
        ),
        "checkpoint_sha256": "8e0c558007770787974e5d0a37e6ddf25c25875f082a653bcab24f513b1f71e3",
        "latent_seed": 770272,
        "report": Path("reports/x2_scratch_dagger_response_v5_round2_ideal_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_response_v5_round2_ideal_survival_audit.json"
        ),
        "canonical_latent_report": Path(
            "reports/x2_scratch_closed_loop_bc_joint_v4.json"
        ),
        "canonical_latent_report_sha256": "de9a49ea27d44e55a27bf1fb0f33be63d880ef242cdf0c166b4cb5c5bef300bc",
        "source_domain_contract": True,
    },
    (512, 400, 770301): {
        "checkpoint": Path("artifacts/x2_scratch_phase_bc_v6_seed770301"),
        "checkpoint_sha256": "0ef2c74dedbda264fb885a434c8732aa268b3221b3a7a62d10d409ee61c191c7",
        "latent_seed": 770302,
        "report": Path("reports/x2_scratch_phase_bc_v6_response_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_phase_bc_v6_response_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "response_domain_contract": True,
        "command_phase_contract": True,
    },
    (512, 400, 770311): {
        "checkpoint": Path("artifacts/x2_scratch_phase_bc_v6_seed770301"),
        "checkpoint_sha256": "0ef2c74dedbda264fb885a434c8732aa268b3221b3a7a62d10d409ee61c191c7",
        "latent_seed": 770312,
        "report": Path("reports/x2_scratch_phase_bc_v6_ideal_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_phase_bc_v6_ideal_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "source_domain_contract": True,
        "command_phase_contract": True,
    },
    (512, 400, 770321): {
        "checkpoint": Path("artifacts/x2_scratch_dagger_phase_v6_seed770321"),
        "checkpoint_sha256": "4a48e4940e704c6dbcd1df336481ab64ba3f5613ed1cc90b3b30130fc27e4703",
        "latent_seed": 770322,
        "report": Path("reports/x2_scratch_dagger_phase_v6_ideal_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_phase_v6_ideal_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "source_domain_contract": True,
        "command_phase_contract": True,
    },
    (512, 400, 770331): {
        "checkpoint": Path("artifacts/x2_scratch_dagger_phase_v6_seed770321"),
        "checkpoint_sha256": "4a48e4940e704c6dbcd1df336481ab64ba3f5613ed1cc90b3b30130fc27e4703",
        "latent_seed": 770332,
        "report": Path("reports/x2_scratch_dagger_phase_v6_response_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_dagger_phase_v6_response_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "response_domain_contract": True,
        "command_phase_contract": True,
    },
    (512, 400, 770341): {
        "checkpoint": Path(
            "artifacts/x2_scratch_dagger_phase_response_v6_seed770341"
        ),
        "checkpoint_sha256": "5252ea30061124430e25b566089ed1aaf2f1c7816b46a2e389585c942d2162b9",
        "latent_seed": 770342,
        "report": Path(
            "reports/x2_scratch_dagger_phase_response_v6_eval.json"
        ),
        "expert_report": Path(
            "reports/x2_scratch_dagger_phase_response_v6_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "response_domain_contract": True,
        "command_phase_contract": True,
    },
    (512, 400, 770361): {
        "checkpoint": Path(
            "artifacts/x2_scratch_response_teacher_v7_seed770361"
        ),
        "checkpoint_sha256": "8a41f61c9896c2bb9e062c24b69924513b0d458cd11b4315cf9071758a99fded",
        "latent_seed": 770362,
        "report": Path("reports/x2_scratch_response_teacher_v7_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_response_teacher_v7_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "response_domain_contract": True,
        "command_phase_contract": True,
    },
    (512, 400, 770371): {
        "checkpoint": Path(
            "artifacts/x2_scratch_response_teacher_v7_seed770361"
        ),
        "checkpoint_sha256": "8a41f61c9896c2bb9e062c24b69924513b0d458cd11b4315cf9071758a99fded",
        "latent_seed": 770372,
        "report": Path("reports/x2_scratch_response_teacher_v7_ideal_eval.json"),
        "expert_report": Path(
            "reports/x2_scratch_response_teacher_v7_ideal_survival_audit.json"
        ),
        "canonical_latent_report": Path("reports/x2_scratch_phase_bc_v6.json"),
        "canonical_latent_report_sha256": "9771b6b92c0c5c66faa64d23e0786f58fcba98fa4daedcf2aae1cc851c85d1fe",
        "source_domain_contract": True,
        "command_phase_contract": True,
    },
}


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode())
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--seed", type=int, default=770011)
parser.add_argument("--role-swap", action="store_true")
parser.add_argument("--expert-latent-screen", action="store_true")
parser.add_argument("--survival-audit", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
evaluation_key = (args.num_envs, args.steps, args.seed)
if evaluation_key not in EVALUATIONS:
    raise ValueError("the paired scratch evaluations are fixed to registered tuples")
evaluation = EVALUATIONS[evaluation_key]
checkpoint = evaluation["checkpoint"]
checkpoint_sha256 = evaluation["checkpoint_sha256"]
report_path = evaluation["report"]
if args.role_swap:
    if evaluation_key != (512, 400, 770023):
        raise ValueError("role swap is registered only for the half-million evaluation")
    report_path = Path("reports/x2_scratch_512_half_million_eval_role_swap.json")
if args.expert_latent_screen:
    if "expert_report" not in evaluation or args.role_swap:
        raise ValueError("expert latent screen is registered only for the half-million evaluation")
    report_path = evaluation["expert_report"]
if args.survival_audit:
    if evaluation_key not in {
        (512, 400, 770051),
        (512, 400, 770071),
        (512, 400, 770081),
        (512, 400, 770091),
        (512, 400, 770101),
        (512, 400, 770111),
        (512, 400, 770121),
        (512, 400, 770131),
        (512, 400, 770151),
        (512, 400, 770181),
        (512, 400, 770191),
        (512, 400, 770201),
        (512, 400, 770211),
        (512, 1000, 770221),
        (512, 400, 770231),
        (512, 400, 770241),
        (512, 400, 770251),
        (512, 400, 770261),
        (512, 400, 770271),
        (512, 400, 770301),
        (512, 400, 770311),
        (512, 400, 770321),
        (512, 400, 770331),
        (512, 400, 770341),
        (512, 400, 770361),
        (512, 400, 770371),
    } or not args.expert_latent_screen:
        raise ValueError("survival audit is registered only for the selected continuation")
    report_path = evaluation["expert_report"]
if not checkpoint.is_dir() or tree_hash(checkpoint) != checkpoint_sha256:
    raise RuntimeError("the bounded scratch checkpoint is absent or has drifted")
canonical_latent_report = evaluation.get("canonical_latent_report")
if canonical_latent_report is not None and (
    not canonical_latent_report.is_file()
    or hashlib.sha256(canonical_latent_report.read_bytes()).hexdigest()
    != evaluation["canonical_latent_report_sha256"]
):
    raise RuntimeError("the canonical latent report is absent or has drifted")
if report_path is not None:
    report_temporary = report_path.with_name(f".{report_path.name}.tmp")
    report_sidecar = report_path.with_name(f"{report_path.name}.sha256")
    report_sidecar_temporary = report_sidecar.with_name(f".{report_sidecar.name}.tmp")
    if any(
        path.exists()
        for path in (
            report_path,
            report_temporary,
            report_sidecar,
            report_sidecar_temporary,
        )
    ):
        raise FileExistsError(f"refusing to overwrite evaluation report: {report_path}")

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from gear_sonic.envs.x2_velocity import X2LowerVelocityFlatEnvCfg_PLAY  # noqa: E402
from gear_sonic.envs.manager_env.modular_tracking_env_cfg import (  # noqa: E402
    _apply_x2_actuator_response,
)
from gear_sonic.envs.manager_env.robots.x2 import (  # noqa: E402
    X2_URDF_BY_COLLISION_PROFILE,
)

from humanoidverse.agents.envs.x2_isaaclab import X2IsaacLabVectorEnv  # noqa: E402
from humanoidverse.x2_scratch import (  # noqa: E402
    ACTION_DIM,
    X2_ALL_AUX_REWARD_NAMES,
    X2_LOWER_JOINTS_15,
    X2_SCRATCH_ACTION_SCALE_15,
    X2ScratchDataConfig,
    build_x2_scratch_replay_buffers,
    convert_stage219_critic_rollout,
    load_x2_motion_files,
)
from humanoidverse.x2_scratch_model import (  # noqa: E402
    x2_scratch_agent_config,
    x2_scratch_observation_space,
)


MOTION_ROOT = Path(
    "/home/yu/x2_teleop_final/x2_sonic/motion_lib_x2/"
    "stage72_official_true_forward4_v1"
)


def tensor_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(f"{array.dtype}:{array.shape}".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_cfg():
    cfg = X2LowerVelocityFlatEnvCfg_PLAY()
    cfg.seed = args.seed
    cfg.sim.device = args.device
    cfg.scene.num_envs = args.num_envs
    if args.steps > 400:
        cfg.episode_length_s = 30.0
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.35, 0.35)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_standing_envs = 0.0
    cfg.actions.joint_pos.scale = {
        name: float(scale)
        for name, scale in zip(X2_LOWER_JOINTS_15, X2_SCRATCH_ACTION_SCALE_15)
    }
    if evaluation.get("source_domain_contract", False) or evaluation.get(
        "response_domain_contract", False
    ):
        cfg.scene.robot.spawn.asset_path = X2_URDF_BY_COLLISION_PROFILE["sole12"]
        cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        _apply_x2_actuator_response(
            cfg.scene.robot,
            {
                "enabled": True,
                "profile": "session03_session04_group",
                "randomize": False,
                "strength": 1.0,
                "filter_strength": 1.0,
                "delay_strength": 1.0,
                "include_ideal_endpoint": False,
                "ideal_env_fraction": (
                    0.0 if evaluation.get("response_domain_contract", False) else 1.0
                ),
                "filter_only_env_fraction": 0.0,
            },
            physics_dt_sec=cfg.sim.dt,
        )
    return cfg


def without_time(observation: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in observation.items() if key != "time"}


def empty_summary() -> dict[str, float]:
    return {
        "attempted": 0.0,
        "alive_samples": 0.0,
        "terminated": 0.0,
        "truncated": 0.0,
        "forward_velocity_sum": 0.0,
        "speed_error_sum": 0.0,
        "root_height_sum": 0.0,
        "root_height_min": math.inf,
        "tilt_sum": 0.0,
        "tilt_max": 0.0,
        "action_abs_sum": 0.0,
        "action_values": 0.0,
        "action_clip_count": 0.0,
        "action_rate_sum": 0.0,
        "action_rate_values": 0.0,
        "initial_lanes": 0.0,
        "first_episode_terminated_lanes": 0.0,
        "first_episode_steps_sum": 0.0,
        "first_episode_alive_samples": 0.0,
        "first_episode_forward_velocity_sum": 0.0,
        "first_episode_speed_error_sum": 0.0,
        "first_episode_root_height_sum": 0.0,
        "first_episode_tilt_sum": 0.0,
        **{f"aux_{name}_sum": 0.0 for name in X2_ALL_AUX_REWARD_NAMES},
    }


def finalize_summary(raw: dict[str, float]) -> dict[str, float]:
    alive = max(raw["alive_samples"], 1.0)
    action_values = max(raw["action_values"], 1.0)
    action_rate_values = max(raw["action_rate_values"], 1.0)
    return {
        "attempted_transitions": int(raw["attempted"]),
        "alive_metric_samples": int(raw["alive_samples"]),
        "terminated": int(raw["terminated"]),
        "truncated": int(raw["truncated"]),
        "termination_fraction": raw["terminated"] / max(raw["attempted"], 1.0),
        "forward_velocity_mean_mps": raw["forward_velocity_sum"] / alive,
        "speed_error_mean_mps": raw["speed_error_sum"] / alive,
        "root_height_mean_m": raw["root_height_sum"] / alive,
        "root_height_min_m": raw["root_height_min"],
        "tilt_mean_rad": raw["tilt_sum"] / alive,
        "tilt_max_rad": raw["tilt_max"],
        "action_abs_mean": raw["action_abs_sum"] / action_values,
        "action_clip_fraction": raw["action_clip_count"] / action_values,
        "action_rate_mean_abs": raw["action_rate_sum"] / action_rate_values,
        "first_episode_survival_fraction": (
            (raw["initial_lanes"] - raw["first_episode_terminated_lanes"])
            / max(raw["initial_lanes"], 1.0)
        ),
        "first_episode_terminated_lanes": int(
            raw["first_episode_terminated_lanes"]
        ),
        "first_episode_survival_steps_mean": (
            (
                raw["first_episode_steps_sum"]
                + (raw["initial_lanes"] - raw["first_episode_terminated_lanes"])
                * args.steps
            )
            / max(raw["initial_lanes"], 1.0)
        ),
        "first_episode_forward_velocity_mean_mps": (
            raw["first_episode_forward_velocity_sum"]
            / max(raw["first_episode_alive_samples"], 1.0)
        ),
        "first_episode_speed_error_mean_mps": (
            raw["first_episode_speed_error_sum"]
            / max(raw["first_episode_alive_samples"], 1.0)
        ),
        "first_episode_root_height_mean_m": (
            raw["first_episode_root_height_sum"]
            / max(raw["first_episode_alive_samples"], 1.0)
        ),
        "first_episode_tilt_mean_rad": (
            raw["first_episode_tilt_sum"]
            / max(raw["first_episode_alive_samples"], 1.0)
        ),
        "auxiliary_raw_means": {
            name: raw[f"aux_{name}_sum"] / alive for name in X2_ALL_AUX_REWARD_NAMES
        },
    }


def expert_skill_latents(candidate, episodes) -> tuple[torch.Tensor, list[dict]]:
    """Encode four fixed eight-frame windows from each expert gait clip."""

    latents = []
    metadata = []
    for motion_id, episode in enumerate(episodes):
        frame_count = int(episode["action"].shape[0])
        starts = np.linspace(8, frame_count - 9, num=4).round().astype(np.int64)
        if len(set(starts.tolist())) != 4:
            raise RuntimeError("expert clip is too short for four distinct latent windows")
        for window_index, start in enumerate(starts.tolist()):
            next_observation = {
                key: torch.as_tensor(
                    value[start + 1 : start + 9], dtype=torch.float32, device="cuda"
                )
                for key, value in episode["observation"].items()
            }
            with torch.inference_mode():
                backward = candidate._model.backward_map(next_observation)
                latent = candidate._model.project_z(backward.mean(dim=0, keepdim=True))
            if latent.shape != (1, 64) or not torch.isfinite(latent).all():
                raise RuntimeError("expert latent encoding is invalid")
            latents.append(latent)
            metadata.append(
                {
                    "group": f"expert_{len(metadata):02d}",
                    "motion_id": motion_id,
                    "window_index": window_index,
                    "start_frame": start,
                    "end_frame_exclusive": start + 8,
                }
            )
    encoded = torch.cat(latents, dim=0)
    if encoded.shape != (16, 64):
        raise RuntimeError("expert latent bank shape differs from 16x64")
    return encoded, metadata


def main() -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    motions = sorted(
        path for path in MOTION_ROOT.glob("*.pkl") if path.name != "metadata.pkl"
    )
    episodes, dataset_audit = load_x2_motion_files(
        motions, config=X2ScratchDataConfig(history_length=4)
    )
    expected_expert_frames = 962
    stage219_rollout_data_used = evaluation_key in {
        (512, 400, 770091),
        (512, 400, 770101),
        (512, 400, 770111),
        (512, 400, 770121),
        (512, 400, 770131),
        (512, 400, 770151),
        (512, 400, 770181),
        (512, 400, 770191),
        (512, 400, 770201),
        (512, 400, 770211),
        (512, 1000, 770221),
        (512, 400, 770231),
        (512, 400, 770241),
        (512, 400, 770251),
        (512, 400, 770261),
        (512, 400, 770271),
        (512, 400, 770301),
        (512, 400, 770311),
        (512, 400, 770321),
        (512, 400, 770331),
        (512, 400, 770341),
        (512, 400, 770361),
        (512, 400, 770371),
    }
    if stage219_rollout_data_used:
        bundle = Path(
            "/home/yu/projects/ZHY/CWI_CrossEmbodiment_Sim/artifacts/retarget/"
            "x2_phase70_long_lookahead/rollout_evidence_rerun1.pt"
        )
        if (
            not bundle.is_file()
            or hashlib.sha256(bundle.read_bytes()).hexdigest()
            != "0b7f5f4026eb62ea474a2204b11b67e7c500024e1f3b814add0dffc482a07d81"
        ):
            raise RuntimeError("closed-loop expert bundle is absent or has drifted")
        payload = torch.load(bundle, map_location="cpu", weights_only=False)
        episodes, closed_loop_audit = convert_stage219_critic_rollout(
            payload["critic_observation"].numpy(),
            include_command_phase=evaluation.get("command_phase_contract", False),
            config=X2ScratchDataConfig(history_length=4),
        )
        if not closed_loop_audit["all_finite"]:
            raise RuntimeError("closed-loop expert data are non-finite")
        expected_expert_frames = 25_600
    latent_episodes = episodes[:4]
    expert_replay = build_x2_scratch_replay_buffers(
        episodes, seq_length=8, z_dim=64, seed=args.seed, device="cpu"
    )

    env = ManagerBasedRLEnv(cfg=build_cfg())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    adapter = X2IsaacLabVectorEnv(
        env,
        wrapped,
        history_length=4,
        to_numpy=False,
        include_command_phase=evaluation.get("command_phase_contract", False),
    )
    observation, _ = adapter.reset(seed=args.seed)

    config = x2_scratch_agent_config(
        device="cuda",
        hidden_dim=256,
        hidden_layers=3,
        z_dim=64,
        history_length=4,
        batch_size=32,
        use_x2_aux_rewards=True,
        include_command_phase=evaluation.get("command_phase_contract", False),
    )
    source = config.build(
        x2_scratch_observation_space(
            include_command_phase=evaluation.get("command_phase_contract", False)
        ),
        ACTION_DIM,
    )
    candidate = source.__class__.load(checkpoint, device="cuda")
    source_hash_before = tensor_hash(source._model)
    candidate_hash_before = tensor_hash(candidate._model)

    env_ids = torch.arange(args.num_envs, device="cuda")
    expert_latent_metadata = []
    if args.expert_latent_screen:
        latent_bank, expert_latent_metadata = expert_skill_latents(
            candidate, latent_episodes
        )
        group_ids = torch.arange(16, device="cuda").repeat_interleave(32)
        if canonical_latent_report is not None:
            canonical_payload = json.loads(canonical_latent_report.read_text())
            canonical = torch.as_tensor(
                canonical_payload["canonical_latent"],
                dtype=torch.float32,
                device="cuda",
            ).reshape(1, -1)
            if canonical.shape != (1, 64) or not torch.isfinite(canonical).all():
                raise RuntimeError("canonical latent is invalid")
            z = canonical.expand(args.num_envs, -1).clone()
        else:
            z = latent_bank.index_select(0, group_ids)
        roles = {
            f"expert_{index:02d}": group_ids == index for index in range(16)
        }
        candidate_mask = torch.ones(args.num_envs, dtype=torch.bool, device="cuda")
    else:
        pair_ids = torch.div(env_ids, 2, rounding_mode="floor")
        candidate_mask = env_ids.remainder(2) == pair_ids.remainder(2)
        if args.role_swap:
            candidate_mask = ~candidate_mask
        if int(candidate_mask.sum()) != args.num_envs // 2:
            raise RuntimeError("source/candidate lane allocation is imbalanced")
        torch.manual_seed(evaluation["latent_seed"])
        pair_z = source._model.sample_z(args.num_envs // 2, device="cuda")
        z = pair_z.repeat_interleave(2, dim=0)
        if not torch.equal(z[0::2], z[1::2]):
            raise RuntimeError("paired lanes do not share an exact skill latent")
        roles = {"source": ~candidate_mask, "candidate": candidate_mask}
    summaries = {name: empty_summary() for name in roles}
    for role_name, role_mask in roles.items():
        summaries[role_name]["initial_lanes"] = int(role_mask.sum())
    ever_done = torch.zeros(args.num_envs, dtype=torch.bool, device="cuda")
    previous_action = torch.zeros(args.num_envs, ACTION_DIM, device="cuda")
    for step_index in range(1, args.steps + 1):
        current = without_time(observation)
        with torch.inference_mode():
            candidate_action = candidate.act(current, z, mean=True).clamp(-1.0, 1.0)
            if args.expert_latent_screen:
                action = candidate_action
            else:
                source_action = source.act(current, z, mean=True).clamp(-1.0, 1.0)
                action = torch.where(
                    candidate_mask[:, None], candidate_action, source_action
                )
        next_observation, reward, terminated, truncated, info = adapter.step(action)
        done = terminated | truncated
        data = env.scene["robot"].data
        velocity_x = data.root_lin_vel_b[:, 0]
        root_height = data.root_pos_w[:, 2]
        tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        if not all(
            torch.isfinite(value).all()
            for value in (action, torch.as_tensor(reward), velocity_x, root_height, tilt)
        ):
            raise RuntimeError("closed-loop evaluation produced a non-finite value")

        for role_name, role_mask in roles.items():
            raw = summaries[role_name]
            alive = role_mask & ~done
            first_alive = role_mask & ~ever_done & ~done
            first_done = role_mask & ~ever_done & done
            raw["attempted"] += int(role_mask.sum())
            raw["terminated"] += int(terminated[role_mask].sum())
            raw["truncated"] += int(truncated[role_mask].sum())
            raw["action_abs_sum"] += float(action[role_mask].abs().sum())
            raw["action_values"] += int(role_mask.sum()) * ACTION_DIM
            raw["action_clip_count"] += int((action[role_mask].abs() >= 0.999).sum())
            raw["action_rate_sum"] += float(
                (action[role_mask] - previous_action[role_mask]).abs().sum()
            )
            raw["action_rate_values"] += int(role_mask.sum()) * ACTION_DIM
            raw["first_episode_terminated_lanes"] += int(first_done.sum())
            raw["first_episode_steps_sum"] += int(first_done.sum()) * step_index
            if alive.any():
                count = int(alive.sum())
                raw["alive_samples"] += count
                raw["forward_velocity_sum"] += float(velocity_x[alive].sum())
                raw["speed_error_sum"] += float((velocity_x[alive] - 0.35).abs().sum())
                raw["root_height_sum"] += float(root_height[alive].sum())
                raw["root_height_min"] = min(
                    raw["root_height_min"], float(root_height[alive].min())
                )
                raw["tilt_sum"] += float(tilt[alive].sum())
                raw["tilt_max"] = max(raw["tilt_max"], float(tilt[alive].max()))
                for name, value in info["aux_rewards"].items():
                    raw[f"aux_{name}_sum"] += float(value[alive].sum())
            if first_alive.any():
                count = int(first_alive.sum())
                raw["first_episode_alive_samples"] += count
                raw["first_episode_forward_velocity_sum"] += float(
                    velocity_x[first_alive].sum()
                )
                raw["first_episode_speed_error_sum"] += float(
                    (velocity_x[first_alive] - 0.35).abs().sum()
                )
                raw["first_episode_root_height_sum"] += float(
                    root_height[first_alive].sum()
                )
                raw["first_episode_tilt_sum"] += float(tilt[first_alive].sum())
        ever_done |= done
        previous_action.copy_(action)
        previous_action[done] = 0.0
        observation = next_observation

    source_hash_after = tensor_hash(source._model)
    candidate_hash_after = tensor_hash(candidate._model)
    finalized = {name: finalize_summary(value) for name, value in summaries.items()}
    valid = (
        dataset_audit["all_finite"]
        and expert_replay["audit"]["expert_frames"] == expected_expert_frames
        and source_hash_before == source_hash_after
        and candidate_hash_before == candidate_hash_after
        and all(item["truncated"] == 0 for item in finalized.values())
        and all(
            math.isfinite(value)
            for item in finalized.values()
            for key, value in item.items()
            if isinstance(value, float)
        )
    )
    return {
        "schema": "x2_bfm_zero_scratch_bounded_closed_loop_eval_v2",
        "decision": "EVAL_FINITE" if valid else "FAIL_EVAL_INVALID",
        "scratch_from_zero": True,
        "stage219_weights_loaded": False,
        "stage219_rollout_data_used": stage219_rollout_data_used,
        "expert_frames": expert_replay["audit"]["expert_frames"],
        "checkpoint_tree_sha256": checkpoint_sha256,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "role_lanes": {name: int(mask.sum()) for name, mask in roles.items()},
        "role_assignment": (
            "adjacent pairs with inverted alternating candidate parity"
            if args.role_swap
            else "adjacent pairs with alternating candidate parity"
        ),
        "role_swap": args.role_swap,
        "paired_skill_latent_exact": not args.expert_latent_screen,
        "latent_source": (
            "single canonical latent frozen by the BC report"
            if canonical_latent_report is not None
            else "candidate backward-map mean over fixed expert eight-frame windows"
            if args.expert_latent_screen
            else "fixed random paired latent"
        ),
        "expert_latent_metadata": expert_latent_metadata,
        "deterministic_actor_mean": True,
        "models_unchanged": {
            "source": source_hash_before == source_hash_after,
            "candidate": candidate_hash_before == candidate_hash_after,
        },
        "groups": finalized,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "performance_claim": False,
        "long_training_unlocked": False,
    }


try:
    report = main()
    serialized = json.dumps(report, sort_keys=True, allow_nan=False)
    if report["decision"] != "EVAL_FINITE":
        raise RuntimeError(serialized)
except BaseException as error:
    print(
        json.dumps(
            {
                "schema": "x2_bfm_zero_scratch_update8_eval_failure_v1",
                "error_type": type(error).__name__,
                "error": str(error),
                "optimizer_steps": 0,
                "checkpoint_writes": 0,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
else:
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        report_temporary.write_bytes(payload)
        os.replace(report_temporary, report_path)
        sidecar_payload = (
            f"{hashlib.sha256(payload).hexdigest()}  {report_path.name}\n"
        )
        report_sidecar_temporary.write_text(sidecar_payload)
        os.replace(report_sidecar_temporary, report_sidecar)
    print(serialized, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
