import setuptools

try:
   import pypandoc
   description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   description = open('README.md').read()

setuptools.setup(
    packages = ['tectosaur'],
    install_requires = ['numpy', 'sympy', 'scipy', 'mako', 'cppimport', 'joblib'],
    zip_safe = False,

    name = 'tectosaur',
    version = '0.0.1',
    author = 'T. Ben Thompson',
    author_email = 't.ben.thompson@gmail.com',
    license = 'MIT',
    platforms = ['any']
    # Add classifiers
)
