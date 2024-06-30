#!/usr/bin/env python3
"""
Computes embeddings on a set of tasks
"""
import argparse
import json
import os
import shutil
from collections import namedtuple
from pathlib import Path
from typing import Optional
from time import sleep

import submitit
from slugify import slugify
from tqdm import tqdm

from heareval.embeddings.task_embeddings import Embedding, task_embeddings
from heareval.utils import run_utils

Submission = namedtuple("Submission", ["task_path", "embed_task_dir", "done_embeddings", "jobs"])


def pop_finished_submission(submissions: list[Submission]) -> Optional[Submission]:
    for i, s in enumerate(submissions):
        if count_done_jobs(s.jobs) == len(s.jobs):
            return submissions.pop(i)

    return None


def count_done_jobs(jobs: list[submitit.Job]):
    return len([j.done() for j in jobs])

def runner(
    module: str,
    model: str,
    tasks_dir: str,
    task: str,
    embeddings_dir: str,
    model_options: str,
    slurm_args: argparse.Namespace,
) -> None:
    model_options_dict = json.loads(model_options)
    if isinstance(model_options_dict, dict):
        if model_options_dict:
            options_str = "-" + "-".join(
                [
                    "%s=%s" % (slugify(k), slugify(str(v)))
                    for k, v in model_options_dict.items()
                ]
            )
        else:
            options_str = ""
    else:
        raise ValueError("model_options should be a JSON dict")

    # Check for directory containing the tasks
    tasks_dir_path = Path(tasks_dir)
    embeddings_dir_path = Path(embeddings_dir)
    print(embeddings_dir_path)
    if not tasks_dir_path.is_dir():
        raise ValueError(
            "Cannot locate directory containing tasks. "
            f"Ensure that directory named {tasks_dir_path} exists or specify a folder "
            f"containing HEAR tasks using the argument --tasks-dir"
        )

    # Load the embedding model
    embedding = Embedding(module, model, model_options_dict)

    submissions: list[Submission] = []
    all_jobs: list[submitit.Job] = []

    if task == "all":
        tasks = list(tasks_dir_path.iterdir())
    else:
        tasks = [tasks_dir_path.joinpath(task)]
        assert os.path.exists(tasks[0]), f"{tasks[0]} does not exist"
    for task_path in tqdm(tasks):
        # TODO: Would be good to include the version here
        # https://github.com/hearbenchmark/hear2021-eval-kit/issues/37
        embed_dir = embeddings_dir_path.joinpath(embedding.name + options_str)

        task_name = task_path.name
        embed_task_dir = embed_dir.joinpath(task_name)

        done_embeddings = embed_task_dir.joinpath(".done.embeddings")
        # if os.path.exists(done_embeddings):
        #     continue

        if os.path.exists(embed_task_dir):
            shutil.rmtree(embed_task_dir)

        jobs = task_embeddings(embedding, task_path, embed_task_dir, slurm_args)

        all_jobs.extend(jobs)
        submissions.append(Submission(task_path, embed_task_dir, done_embeddings, jobs))

    with tqdm(total=len(all_jobs)) as pbar:
        sleep(1)
        pbar.moveto(count_done_jobs(all_jobs))

        submission = pop_finished_submission(submissions)
        if submission is not None:
            print(f"...computed embeddings for {submission.task_path.name} using {module} {model_options}")

    run_utils.wait(all_jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_utils.add_slurm_args(parser, cpus_per_task=10, gpus_per_node=1)
    parser.add_argument("module", type=str)
    parser.add_argument("--model", type=Path, help="Location of model weights file")
    parser.add_argument("--tasks-dir", default="tasks", type=str, help="Location of tasks to compute embeddings on")
    parser.add_argument("--task", type=str, help="Task to run", default="all")
    parser.add_argument("--embeddings-dir", default="embeddings", type=str, help="Location to save task embeddings")
    parser.add_argument("--model-options", default="{}", type=str)
    args = parser.parse_args()

    runner(args.module, args.model, args.tasks_dir, args.task, args.embeddings_dir, args.model_options, slurm_args=args)


