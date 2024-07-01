import argparse
from typing import List, Optional, Any, Callable
import submitit

# Type alias for a run function that takes a configuration dictionary, an argparse object, a rank,
# and any number of additional arguments. It can return any type.
RunFn = Callable[[...], Optional[Any]]


def add_slurm_args(parser: argparse.ArgumentParser, cpus_per_task: int = 1, gpus_per_node: int = 0,
                   mem: str = '16GB', partition: str = 'gpu_p13', timeout: int = 120, account: Optional[str] = None,
                   qos: str = 'qos_gpu-dev', log_dir: str = 'logs', constraint: Optional[str] = None,
                   exclude: Optional[str] = None) -> None:
    """
    Adds submit arguments to the provided argparse.ArgumentParser object.

    Parameters:
    parser (argparse.ArgumentParser): The parser to which arguments will be added.
    cpus_per_task (int): The number of CPUs per task.
    gpus_per_node (int): The number of GPUs per node.
    partition (str): The partition for the resource allocation.
    timeout (int): The timeout in minutes.
    memory (str): Minimum memory per node. Number followed by unit prefx.
    account (str | None): The account to use.
    qos (str): The quality of service.
    constraint (str | None): The feature constraints for the jobs.
    exclude (str | None): The nodes to exclude.
    """
    parser.add_argument('-c', '--cpus-per-task', default=cpus_per_task, help='Number of CPUs per task', type=int)
    parser.add_argument('-gpus', '--gpus-per-node', default=gpus_per_node, help='Number of GPUs per task', type=int)
    parser.add_argument('-p', '--partition', default=partition, help='Partition for the resource allocation', type=str)
    parser.add_argument('-t', '--timeout', default=timeout, help='Timeout in minutes', type=int)
    parser.add_argument('-a', '--account', default=account, help='Account to use.', type=str)
    parser.add_argument('-q', '--qos', default=qos, help='Quality of service', type=str)
    parser.add_argument('-C', '--constraint', default=constraint, help='Feature constraints for the jobs', type=str)
    parser.add_argument('-l', '--log-dir', default=log_dir, help='Log directory', type=str)
    parser.add_argument('--mem', default=mem, help='Minimum memory per node. Number followed by unit prefix.',
                        type=str)
    parser.add_argument('--exclude', default=exclude, help='Nodes to exclude', type=str)


def execute(args: argparse.Namespace, fn: RunFn, *other_args) -> submitit.Job:
    """
    Executes a function on a set of arguments and a configuration.

    Parameters:
    args (argparse.Namespace): The arguments to use.
    cfg (Any): The configuration to use.
    fn (RunFn): The function to execute.

    Returns:
    List[Any]: The list of submitted jobs.
    """
    log_folder = f"{args.log_dir}/%j"
    executor = submitit.AutoExecutor(folder=log_folder)
    executor.update_parameters(timeout_min=args.timeout, slurm_constraint=args.constraint,
                               slurm_cpus_per_task=args.cpus_per_task, slurm_gpus_per_node=args.gpus_per_node,
                               slurm_partition=args.partition, slurm_exclude=args.exclude, slurm_mem=args.mem,
                               slurm_additional_parameters={'account': args.account,
                                                            'qos': args.qos, } if args.account is not None else None)

    # submit the jobs
    return executor.submit(fn, *other_args)


def wait_for_results(jobs: List[submitit.Job]) -> List[Any]:
    """
    Waits for all jobs to finish and returns the results.

    Parameters:
    jobs (List[submitit.helpers.DelayedSubmission]): The list of jobs to wait for.

    Returns:
    List[Any]: The list of results.
    """
    return [job.result() for job in jobs]


def wait(jobs: List[submitit.Job]) -> None:
    """
    Waits for all jobs to finish.

    Parameters:
    jobs (List[submitit.helpers.DelayedSubmission]): The list of jobs to wait for.
    """
    wait_for_results(jobs)
