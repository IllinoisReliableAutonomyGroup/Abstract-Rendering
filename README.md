# Abstract-Rendering

This repository contains the Abstract-Rendering tool for neural rendering and robust analysis.  
Follow the steps below to install and set up the environment.

---

## Installation Instructions

### 1. Clone the Abstract-Rendering repository

Download the repository from GitHub:

```bash
cd ~
git clone https://github.com/IllinoisReliableAutonomyGroup/Abstract-Rendering.git
```

### 2. Install auto_LiRPA
Follow the installation instructions provided in the *auto_LiRPA* repository to complete the setup. And clone the *auto_LiRPA* repository from the specified branch:
```bash
cd ~
git clone -b van_verify_fix_six https://github.com/Verified-Intelligence/auto_LiRPA.git
```
This step requires repo access accepted by Prof Huan Zhang.

### 3. Create a symbolic link to auto_LiRPA
From the root folder of *Abstract-Rendering*, create a soft link to the auto_LiRPA folder:

```bash
cd ~/Abstract-Rendering
ln -s ~/Verifier_Development/auto_LiRPA auto_LiRPA
```
This allows the project to access *auto_LiRPA* directly.

### 4. Download Nerfstudio data
Download the Nerfstudio dataset of *airplane_grey* case from: *https://drive.google.com/drive/folders/1hmvUQKnrVkC5pbjkJPBLuzCAUa3K97c-?usp=sharing*

From the root folder of *Abstract-Rendering*, create a soft link to the nerfstudio folder:

```bash
cd ~/Abstract-Rendering
ln -s /path/to/downloaded/nerfstudio nerfstudio
```

### 5. Run Example
Test rendering performance of example *airplane_grey*:
```bash
cd ./scripts
python3 render_gsplat.py
```

Test abstract rendering performance of example *airplane_grey*:
```bash
python3 abstract_gsplat.py
```

### 6. View Outputs
Rendered Images can be viewed in 
```bash
~/Abstract-Rendering/Outputs/RenderedImages/**example_name**/**perturbation_type**/
```

Abstract Rendered Images can be viewed in 
```bash
~/Abstract-Rendering/Outputs/AbstractImages/**example_name**/**perturbation_type**/