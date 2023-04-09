from setuptools import setup

with open("requirements.txt") as f:
    required = f.read().splitlines()


setup(
    name="llm_optimize",
    version="0.0.0",
    description="",
    url="https://github.com/sshh12/llm_optimize",
    author="Shrivu Shankar",
    license="MIT",
    packages=["llm_optimize"],
    include_package_data=True,
    install_requires=required,
)
