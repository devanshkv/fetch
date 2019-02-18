from setuptools import setup

setup(
    name='fetch',
    version='0.1.5',
    packages=['fetch'],
    scripts=['bin/predict.py', 'bin/train.py', 'bin/candmaker.py'],
    url='https://github.com/devanshkv/fetch',
    license='GNU General Public License v3.0',
    author=['Devansh Agarwal', 'Kshitij Aggarwal'],
    author_email=['devansh.kv@gmail.com', 'ka0064@mix.wvu.edu'],
    description='FETCH (Fast Extragalactic Transient Candidate Hunter)'
)
