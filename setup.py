import setuptools

setuptools.setup(
    name = 'sisypuss', 
    version = '1.0', 
    author = 'IIOwOII', 
    author_email = 'sasd9750o@naver.com', 
    description = 'Adorable puss will go through countless trials and errors to get better results for you.', 
    url = '프로젝트 깃허브 주소',
    install_requires = [
    "matplotlib",
    "numpy",
    "pytorch"
    ],
    packages = setuptools.find_packages(),
    python_requires = '>=3.9.13',
)