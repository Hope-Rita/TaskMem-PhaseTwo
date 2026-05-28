from setuptools import setup

setup(
    name='vllm-modeling-ext',
    version='1.0',
    packages=['modeling'],
    entry_points={
        'vllm.general_plugins': [
            "qwen2vllm_ada = modeling.qwen2vllm_ada:register",
            "qwen3vllm_ada = modeling.qwen3vllm_ada:register",
        ]
    }
)