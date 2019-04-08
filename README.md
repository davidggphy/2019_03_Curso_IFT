# A Physicist's Introduction to Machine Learning
Machine Learning PhD Course held at IFT (Madrid)

You can find the schedule in: https://moseranette.wixsite.com/doctorado/advanced-1

Contact email: david.gg.phy@gmail.com

Theory sessions delivered by [Bryan Zaldivar](https://github.com/bzaldivarm). Email: bryan.zaldivarm@uam.es

## Outline of the hands-on sessions

* Setup and basics (Days 11/03/2019 and 13/03/2019). [Here](/notebooks/00_intro_to_python)
    * Installation party
    * [Python basics](/notebooks/00_intro_to_python/00_python.ipynb)
    * [Numpy basics](/notebooks/00_intro_to_python/01_numpy.ipynb)
* Classification (Day 20/03/2019). [Here](/notebooks/01_classification)
    * [Decision tree from scratch](/notebooks/01_classification/00_decision_tree.ipynb) 
    * [Classification algorithms with sklearn](/notebooks/01_classification/01_classification_algorithms_sklearn.ipynb)
* Neural Networks (Day 27/03/2019). [Here](/notebooks/02_neural_networks)
    * [Feed forward NN from scratch](/notebooks/02_neural_networks/00_neural_network_from_scratch.ipynb) 
* Unsupervised methods (Day 08/04/2019). [Here](/notebooks/03_unsupervised/)
    * [Gaussian Mixture Models and EM algorithm](/notebooks/03_unsupervised/00_EM_GMM.ipynb)
    * [Principal Component Analysis](/notebooks/03_unsupervised/01_PCA.ipynb)
    
## Setup
We will teach Python using the [Jupyter](https://jupyter.org/) notebook, a programming environment that runs in a web browser. 
However, you can use any Python interpreter compatible with Jupyter notebooks.

### Anaconda (Local, Load predefined env)
1. Download miniconda ***Python 3.7*** (light version of Anaconda) from https://docs.conda.io/en/latest/miniconda.html
2. Installation instructions in https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
3. Update all the packages in conda, and itself, with: 
    `conda update --all`
4. Setup the conda env from the `env.yml` file with: 
    `conda env create --file env.yml`
    
    Depending on your conda version the syntax can be instead:  `conda create --file env.yml`
5. Activate the environment with: 
    - Windows: `activate ML_Course`
    - Linux, macOS: `conda activate ML_Course` or `source activate ML_Course` 
6. Open Jupyter with: 
    `jupyter notebook`
    
### Anaconda (Local, Install your own packages)
1. Download miniconda ***Python 3.7*** (light version of Anaconda) from https://docs.conda.io/en/latest/miniconda.html
2. Installation instructions in https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
3. Update all the packages in conda, and itself, with: 
    `conda update --all`
4. Create a new environment with
    `conda create -n ML_Course python=3.6`
    
    Depending on your conda version the syntax can be instead:  `conda env create -n ML_Course python=3.6`
5. Activate the environment with: 
    - Windows: `activate ML_Course`
    - Linux, macOS: `conda activate ML_Course` or `source activate ML_Course` 
6. Install the following packages with conda: 
    `conda install pip graphviz`
7. Install the required packages with pip: 
    `pip install -r requirements.txt`
8. Open Jupyter with: 
    `jupyter notebook`


### Google Colab (Cloud)
Google has released its own flavour of Jupyter called Colab, which has free GPUs available.

Here's how you can use it:
1. Open https://colab.research.google.com, click **Sign in** in the upper right corner, use your Google credentials to sign in.
2. Click **GITHUB** tab, paste https://github.com/davidggphy/2019_03_Curso_IFT and press Enter
3. Choose the notebook you want to open *.ipynb
4. Click **File -> Save a copy in Drive...** to save your progress in Google Drive
5. Optional: Click **Runtime -> Change runtime type** and select **GPU** in Hardware accelerator box

## Warm-up
You can find extra information about how to install Anaconda and a tutorial on [python basics](https://adgdt.github.io/2018-11-28-cftmat-python-novice-inflammation) in
https://adgdt.github.io/2018-11-28-cftmat/.


## Contributing
Contributions and issues are welcome. Thanks!

### Issues

#### Graphviz executables not found. Windows. (Thanks to Eduardo García-Valdecasas)
1. Download and install graphviz-2.38.msi from
https://graphviz.gitlab.io/_pages/Download/Download_windows.html
2. Set the path variable
    *  Control Panel > System and Security > System > Advanced System Settings > Environment Variables > Path > Edit
    *  add `C:\Program Files (x86)\Graphviz2.38\bin`
