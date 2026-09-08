from setuptools import find_packages, setup


setup(
    name="simtoolreal-animrl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["numpy", "tensorboard", "torch"],
    extras_require={
        "video": ["imageio", "imageio-ffmpeg"],
        "sim2sim": ["matplotlib>=3.5", "mujoco>=3.2"],
    },
)
