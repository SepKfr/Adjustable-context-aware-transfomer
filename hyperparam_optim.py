from clearml import Task
from clearml.automation.optuna.optuna import OptimizerOptuna
from clearml.automation.optimization import HyperParameterOptimizer
from clearml.automation.parameters import DiscreteParameterRange
import argparse

search_strategy = OptimizerOptuna


def job_complete_callback(
        job_id,                 #type: str
        objective_value,        #type: float
        objective_iteration,    #type: int
        job_parameters,         #type: dict
        top_performance_job_id  #type: str
):
    print('Job completed!', job_id, objective_value, objective_iteration, job_parameters)
    if job_id == top_performance_job_id:
        print('WOOT WOOT we broke the record! Objective reached {}'.format(objective_value))


def run_hyperparam_optim(project_name, task_name, task_id):
    # Connecting ClearML
    task = Task.init(project_name='watershed Hyper-param optimization',
                     task_name='Automatic Hyper-param optimization',
                     task_type=Task.TaskTypes.optimizer,
                     reuse_last_task_id=False)

    # experiment template task experiment that we want to optimize in the hyper-parameter optimization
    args = {
        'template_task_id': task_id,
        'run_as_service': False,
    }
    args = task.connect(args)

    # Get the template task experiment that we want to optimize
    if not args['template_task_id']:
        args['template_task_id'] = Task.get_task(
            project_name=project_name, task_name=task_name).id

    an_optimizer = HyperParameterOptimizer(
        base_task_id=args['template_task_id'],
        hyper_parameters=[
            DiscreteParameterRange('args/n_heads', values=[1, 4]),
            DiscreteParameterRange('args/n_layers', values=[1, 3]),
            DiscreteParameterRange('args/lr', values=[0.0001, 0.001, 0.01]),
            DiscreteParameterRange('args/dr', values=[0.1, 0.5])
        ],
        objective_metric_title='evaluate',
        objective_metric_series='loss',
        objective_metric_sign='max',
        max_number_of_concurrent_tasks=2,
        optimizer_class=search_strategy,
        execution_queue='default',
        time_limit_per_job=60.,
        total_max_jobs=10,
        min_iteration_per_job=100,
        max_iteration_per_job=10000
    )

    an_optimizer.set_report_period(1.0)

    an_optimizer.start(job_complete_callback=job_complete_callback)

    an_optimizer.set_time_limit(in_minutes=1440.0)

    an_optimizer.wait()

    top_exp = an_optimizer.get_top_experiments(top_k=5)
    print([t.id for t in top_exp])

    an_optimizer.stop()

    print('We are done, good bye')


def main():
    parser = argparse.ArgumentParser(description='Run hyper-parameter optimization for watershed')
    parser.add_argument('--project_name', default='watershed')
    parser.add_argument('--task_name', default='watershed training')
    parser.add_argument('-ti', '--task_id')
    args = parser.parse_args()
    run_hyperparam_optim(args.project_name, args.task_name, args.task_id)


if __name__ == '__main__':
    main()