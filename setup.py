from setuptools import setup, find_packages

setup(
    name='eagle-llm',
    version='1.2.1',
    description='Accelerating LLMs by 3x with No Quality Loss',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='yuhui.li@stu.pku.edu.cn',
    url='https://github.com/SafeAILab/EAGLE',
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "accelerate",
        "fschat == 0.2.31",
        "gradio == 3.50.2",
        "openai == 0.28.0",
        "anthropic == 0.5.0",
        "sentencepiece == 0.1.99",
        "protobuf",
        "wandb"
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)