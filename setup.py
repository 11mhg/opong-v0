from setuptools import setup

setup(name='gym_opong',
        version='0.0.1',
        description='Private openai gym implementation of pong with custom state space encoding',
        author='Mohammed Gasmallah',
        author_email='11mhg@queensu.ca',
        licens='unlicense',
        install_requires=['gym','pygame','numpy'],
        zip_safe=False
)
