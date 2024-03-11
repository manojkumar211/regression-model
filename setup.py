from setuptools import find_packages, setup


setup(
    name="regression_model",
      version="0.0.1",
    author="manojkumar21",
    author_email="mk.chinthakindi@gmail.com",
    install_requiress=['numpy','pandas','seaborn','matplotlib','scikit-learn','statsmodels'],
    packages=find_packages()

)