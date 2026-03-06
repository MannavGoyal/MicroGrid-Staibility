"""Setup configuration for microgrid backend package."""
from setuptools import setup, find_packages

setup(
    name='microgrid-backend',
    version='0.1.0',
    description='Backend API for microgrid stability prediction and analysis',
    author='Microgrid Research Team',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'Flask>=3.0.0',
        'Flask-CORS>=4.0.0',
        'Flask-RESTful>=0.3.10',
        'pandas>=2.1.4',
        'numpy>=1.26.2',
        'torch>=2.1.2',
        'scikit-learn>=1.3.2',
        'statsmodels>=0.14.1',
        'pydantic>=2.5.3',
        'scipy>=1.11.4',
        'python-dotenv>=1.0.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'hypothesis>=6.92.2',
            'ruff>=0.1.9',
            'black>=23.12.1',
            'mypy>=1.8.0',
        ],
    },
)
