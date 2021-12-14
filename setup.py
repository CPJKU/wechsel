from setuptools import setup, find_packages

setup(
    name="wechsel",
    version="0.0.3",
    author="Benjamin Minixhofer",
    packages=find_packages(),
    author_email="bminixhofer@gmail.com",
    url="https://github.com/cpjku/wechsel",
    description="Code for WECHSEL: Effective initialization of subword embeddings for cross-lingual transfer of monolingual language models.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6.0",
    install_requires=[
        "requests>=2.26.0",
        "gensim>=4.1.2",
        "fasttext>=0.9.2",
        "datasets>=1.16.1",
        "tqdm>=4.62.3",
        "scipy>=1.7.3",
        "scikit-learn>=1.0.1",
        "nltk>=3.6.5",
    ],
    license="MIT",
)
