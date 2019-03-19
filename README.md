# A PHYSICIST'S INTRODUCTION TO MACHINE LEARNING
Machine Learning PhD Course held at IFT (Madrid)

You can find the schedule in: http://asabiovera.wixsite.com/doctorado/advanced-1

## Setup
We will teach Python using the [Jupyter](https://jupyter.org/) notebook, a programming environment that runs in a web browser. 
However, you can use any Python interpreter compatible with Jupyter notebooks.

### Anaconda (Local, Load predefined env)
1. Download miniconda ***Python 3.7*** (light version of Anaconda) from https://docs.conda.io/en/latest/miniconda.html
2. Installation instructions in https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
3. Setup the conda env from the `env.yml` file with: 
    `conda create -f env.yml`
4. Activate the environment with: 
    `conda activate ML_Course` or `source activate ML_Course` 
5. Open Jupyter with: 
    `jupyter notebook`
    
### Anaconda (Local, Install your own packages)
1. Download miniconda ***Python 3.7*** (light version of Anaconda) from https://docs.conda.io/en/latest/miniconda.html
2. Installation instructions in https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
3. Create a new environment with
    `conda create -n ML_Course python=3.6`
4. Activate the environment with
    `conda activate ML_Course` or `source activate ML_Course` 
5. Install the following packages with conda: 
    `conda install pip graphviz`
6. Install the required packages with pip: 
    `pip install -r requirements.txt`
7. Open Jupyter with: 
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

## Issues

### Graphviz executables not found. (Thanks to Eduardo García-Valdecasas)
1. Download and install graphviz-2.38.msi from
https://graphviz.gitlab.io/_pages/Download/Download_windows.html
2. Set the path variable
    *  Control Panel > System and Security > System > Advanced System Settings > Environment Variables > Path > Edit
    *  add `C:\Program Files (x86)\Graphviz2.38\bin`
