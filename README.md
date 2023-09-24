# Predictor Explainer

AutoML and ExplainableAI for JMP (+Python) for Windows and Mac
[Download latest version](https://github.com/industrial-data/predictor-explainer/raw/main/Latest_PredictorExplainer.jmpaddin)

Predictor explainer automates the screening of process variables using feature engineering and machine learning (known as AutoML). Parallel coordinate plots and trends will be automatically shown to interpret the results. If [PyJMP](https://github.com/industrial-data/pyJMP/) is installed (**optional**), SHAP plots and UMAP will be automatically calculated as well.

For further details and applications of ML applied to industrial processes, you can have a look at our open-access review:

* Review - [Industrial data science – a review of machine learning applications for chemical and process industries](https://pubs.rsc.org/en/content/articlelanding/2022/re/d1re00541c)
* Book chapter - [Industrial Data Science for Batch Manufacturing Processes](https://arxiv.org/abs/2209.09660)

* Batch demo - [Predictor Explainer at JMP Discovery Summit 2023](https://community.jmp.com/t5/Discovery-Summit-Europe-2023/Industrial-Data-Science-for-Batch-Manufacturing-Processes-2023/ta-p/572636?trMode=source)

![](/media/image1.png)



---

## Installation steps

## 1 – Download and installation

Download and click the . **jmpaddin** file to install Predictor Explainer:

[Download latest version](https://github.com/industrial-data/predictor-explainer/raw/main/Latest_PredictorExplainer.jmpaddin)

If you have a previous version installed, it will be automatically removed.

Optional: [PyJMP (Python for JMP (Windows only))](https://github.com/industrial-data/pyjmp) can also be installed (SHAP plots will be generated in JMP).
If you have Mac or another Python installation, you can still run the notebook included in the folder by clicking on the button "code|examples".
Temporary CSVs will be generated inside the 'temp' folder.


## 2 – Example: Distillation column (continuous process)

In this example will be illustrate how to screen tags (sensor data) to quickly identify correlated factors to the yield in a distillation column.

![](/media/image2.png)

Go to the Add-ins menu and open Predictor Explainer.

![](/media/image3.png)

Predictor explainer contains example files. To open the folder, click in the button on the bottom left corner (see A in image below). The following steps are using the file '_distillation\_column\_na.jmp'_.

1. Select a target (Y)
2. Add all the input variables or predictors (X)
3. You can pass extra information such as date, supplementary information or weights. Grouping variables, will be used to create summary statistics before the analysis is done (automatic feature engineering).
4. You can specify several options, such as the creating of differences for all the input variables, the number of trees in the bootstrap forest, and the threshold for the signal to noise ratio.
5. Predictor Explainer will create a subset of the table and show the analysis results

(A) The add-in folder contains other example files and a python code

(B) If pyJMP is installed, additional options for SHAP and UMAP will be shown.

![](/media/image4.png)

The main results are:

1. Predictor screening (bootstrap forest) of the variables introduced or pre-calculated ones using grouping or differences.
2. Uniform and random noise columns will be added and used as threshold to select the strongest predictors. The signal-to-noise option parameter determines how much difference in contribution will be taken
3. A parallel coordinate plot will be shown with the top #3 variables. If signal is not strong enough compared to noise, this plot will not be shown.

![](/media/image5.png)

A hidden and temporal table with all the pre-selected predictors, target and time variables can be accessed via JMP home. The original analysis table won't be modified.

![](/media/image6.png)

If PyJMP is installed and the option to show SHAP plots is activated, an interactive violin plot will appear after the analysis.

![](/media/image7.png)

Additional hidden tables containing SHAP, UMAP and clustering results will be accessible via the home menu.

## 3 – Example: Batch data analysis

Predictor Explainer can also be used to screen sensors measuring batch processes. The file named '_Fermentation\_Batch\_Data.jmp'_ illustrates the challenge of combining unique values coming from a lab analysis with process data.

![](/media/image8.png)

Predictor explainer will first create a table with summary statistics (also called fingerprints or landmarks).

![](/media/image9.png)

If there is information about product (grade) or phase (stage) of the batch, these will be also used to generate more granular summary statistics. Using the noise contribution as a filter will eliminate all calculated features below it. When PhaseID is introduced as numeric and ordinal column (1, 2, 3…), an automatic aligning of the batch will be performed and shown.

In the example, Predictor Explainer identified the strongest sensor (tag) in terms of correlation (supervised learning). ![](/media/image10.png)

If no output is given (no Y, only X's), Predictor screening identifies the sensor with highest variability and creates a global anomaly score (unsupervised learning). 

![](/media/image13.png)

## Creating your own version of the add-in

If you want to distribute or modify a new version of the JMP addin, there are two important things to consider.

1. Changing the Jupyter notebook will not require changing the JMP add-in itself, as JMP calls the Python code independently via PyJMP. JMP will generate temporary files to communicate with Python and then read the results. The Jupyter Notebooks contains all the steps, to access it open the folder clicking the button on bottom left corner. There is one executable called open\_notebook.bat which will allow you to modify the jupyter that is executed.

1. When modifying the JML add-in source code and saving it (exporting application), make sure to keep the same Unique ID:

![](/media/image11.png)

As well as add the Notebook and other files you want to distribute with the add-in.

![](/media/image12.png)

In case you have suggestions, errors, or success stories (!) contact us

[https://github.com/Industrial-data/predictor-explainer](https://github.com/Industrial-data/predictor-explainer)
