"""Smoke test: every task in task_table.tsv makes a task file and runs one short block."""
import inspect
import types
import pandas as pd
import pytest
import MultiTaskBattery.task_file as tf
import MultiTaskBattery.utils as ut
import MultiTaskBattery.experiment_block as exp_block

# shortest block (s) each task can fill; tasks not listed run 6 s
DUR = {'finger_rhythmic': 70, 'time_perception': 10, 'demand_grid': 10,
       'action_observation': 15, 'passage_listening': 15, 'semantic_prediction': 15,
       'theory_of_mind': 15,
       'movie': 30, 'auditory_narrative': 30, 'spatial_navigation': 30}


@pytest.fixture(scope='session')
def const(tmp_path_factory):
    d = tmp_path_factory.mktemp('exp')
    return types.SimpleNamespace(
        exp_name='smoke', exp_dir=d, task_dir=d / 'task_files', run_dir=d / 'run_files',
        data_dir=d / 'data', stim_dir=ut._PACKAGE_STIM_DIR, default_run_filename='run_01.tsv',
        response_keys=['a', 's', 'd', 'f'], eye_tracker=False, debug=True, show_scoreboard=False,
        screen={'size': [800, 600], 'fullscr': False, 'number': 0, 'width': 53, 'distance': 57})


@pytest.fixture(scope='session')
def screen(const):
    return exp_block.Experiment(const, subj_id='setup').screen  # one window for all tests


@pytest.mark.parametrize('name', ut.get_task_table().name)
def test_task(name, const, screen):
    dur = DUR.get(name, 6)
    (const.task_dir / name).mkdir(parents=True, exist_ok=True)
    const.run_dir.mkdir(exist_ok=True)

    # task file
    task_file = ut.get_task_file_class(const, tf.get_task_class(name))(const)
    args = {'task_dur': dur, 'file_name': f'{name}.tsv'}
    if 'run_number' in inspect.signature(task_file.make_task_file).parameters:
        args['run_number'] = 1
    task_file.make_task_file(**args)

    # one-block run
    tf.make_run_file([name], [f'{name}.tsv'], instruction_dur=1, task_dur=dur,
                     shuffle=False).to_csv(const.run_dir / 'run_01.tsv', sep='\t', index=False)
    e = exp_block.Experiment(const, subj_id=name, screen=screen)
    e.confirm_run_info()
    e.subj_id, e.wait_ttl = name, False
    e.init_run()
    e.run()

    # data saved, block starts on time and does not run past its end
    block = pd.read_csv(const.data_dir / name / f'{name}.tsv', sep='\t').iloc[-1]
    assert list((const.data_dir / name).glob(f'{name}_task-*.tsv')), 'no trial data saved'
    assert abs(block.real_start_time - block.start_time) < 0.1
    assert block.real_end_time < block.end_time + 0.5, 'block ran past its end'
