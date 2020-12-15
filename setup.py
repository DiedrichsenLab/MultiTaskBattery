from setuptools import find_packages, setup

setup(
    name='experiment_code',
    packages=find_packages(),
    version='0.1.0',
    description='This project uses a reduced battery of cognitive and motor tasks to map the functional sub-regions of the human cerebellum',
    entry_points={
        'console_scripts': [
            'run-behavioral=experiment_code.scripts.run_behavioral:run',
            'run-fmri=experiment_code.scripts.run_fmri:run',
            'makefiles-behavioral=experiment_code.scripts.makefiles_behavioral:make_files',
            'makefiles-fmri=experiment_code.scripts.makefiles_fmri:make_files',
        ]
    },
    author='Maedbh King',
    license='MIT',
)
