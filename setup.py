import setuptools

setuptools.setup(
    packages = ['tectosaur'],
    install_requires = ['numpy', 'sympy', 'scipy', 'dill'],
    zip_safe = False,

    name = 'tectosaur',
    version = '0.0.1',
    description = 'Observe the tectonosaurus and the elastosaurus pleasant romp in the fields of crustal rock.',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any']
    # Add classifiers
)
