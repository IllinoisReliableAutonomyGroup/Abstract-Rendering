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
Download the Nerfstudio dataset from: *https://drive.google.com/drive/folders/1tUD5NPt9iT4OKgfWAXkyUW0u_2F-1qFb?usp=drive_link*

Replace the folder *nerfstudio* under the root folder of *Abstract-Rendering* with the downloaded dataset:

```bash
mv /path/to/downloaded/nerfstudio ./nerfstudio
```

### 5. Run Example
Test rendering performance of example *airplane_grey*:
```bash
cd ./examples/airplane_grey
python3 render_gsplat.py
```

Test abstract rendering performance of example *airplane_grey*:
```bash
cd ./examples/airplane_grey
python3 abstract_gsplat.py
```

### 6. View Ourputs
Rendered Images can be viewed in 
```bash
./Outputs/RenderedImages/**example_name**/**perturbation_type**/
```

Abstract Rendered Images can be viewed in 
```bash
./Outputs/AbstractImages/**example_name**/**perturbation_type**/