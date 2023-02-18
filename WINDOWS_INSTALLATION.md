# SageMaker Local Mode on Windows

SageMaker Local will not work on Windows unless you install WSL 2 and then, a Linux distro (Ubuntu is the default).

If you try to run the examples in this repo, you'll eventually get `TypeError: object of type 'int' has no len()` error after completing the training job.
![Error training in Windows - exception](img/windows_error_01.png)

The problem is because of failures to output the model on temporary folders created for the training job by SageMaker Local, and are related to Windows directory structure/permissions. 

![Error training in Windows - directory structure](img/windows_error_02.png)