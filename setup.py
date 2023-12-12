from setuptools import setup

setup(
    name='conditional_explainer',
    version='0.0.1',
    license='MIT',
    description='Provides explainers based on conditional imputers',
    packages=['conditional_explainer'],  # same as name
    keywords=['XAI', 'PredDiff', 'Shapley values', 'Interactions', 'Model-agnostic attributions'],
    author='Stefan Bluecher',
    author_email='bluecher@tu-berlin.de',
    # external packages as dependencies
    install_requires=['numpy'],
)
