# A PHYSICIST'S INTRODUCTION TO MACHINE LEARNING
Machine Learning PhD Course held at IFT (Madrid)

You can find the schedule in: http://asabiovera.wixsite.com/doctorado/advanced-1

## Setup
We will teach Python using the [Jupyter](https://jupyter.org/) notebook, a programming environment that runs in a web browser. 
However, you can use any Python interpreter compatible with Jupyter notebooks.

### Anaconda (Local)
1. Download miniconda ***Python 3.7*** (light version of Anaconda) from https://docs.conda.io/en/latest/miniconda.html
2. Installation instructions in https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
3. Setup the conda env from the `env.yml` file with
    `conda env create -f env.yml`
4. Activate the environment with
    `conda activate ML_Course`
5. Open Jupyter with
    `jupyter notebook`


### Google Colab (Cloud)
Google has released its own flavour of Jupyter called Colab, which has free GPUs available.

Here's how you can use it:
1. Open https://colab.research.google.com, click **Sign in** in the upper right corner, use your Google credentials to sign in.
2. Click **GITHUB** tab, paste https://github.com/davidggphy/2019_03_Curso_IFT and press Enter
3. Choose the notebook you want to open *.ipynb
4. Click **File -> Save a copy in Drive...** to save your progress in Google Drive
5. Click **Runtime -> Change runtime type** and select **GPU** in Hardware accelerator box

## Warm-up
You can find extra information about how to install Anaconda and a tutorial on [python basics](https://adgdt.github.io/2018-11-28-cftmat-python-novice-inflammation) in
https://adgdt.github.io/2018-11-28-cftmat/.


## Contributing
Contributions and issues are welcome. Thanks!
