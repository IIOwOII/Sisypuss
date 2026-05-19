import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = 'sisypuss',
    version = '0.1',
    author = 'IIOwOII',
    author_email = 'sasd9750o@naver.com',
    description = 'The puss named Sisyphus',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url = '',
    packages = setuptools.find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires = '>=3.9'
)
