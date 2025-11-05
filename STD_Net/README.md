#  **STD-Net: Spatially Topology-aware and Dynamically Fused Network for Cell-type Deconvolution in Spatial Transcriptomics**

**STD-Net: Spatially Topology-aware Deconvolution Network** is a graph-based deep learning framework designed for high-resolution cell-type deconvolution of spatial transcriptomics data. The model integrates single-cell RNA sequencing data with spatial topology information to accurately resolve cellular heterogeneity within tissue microenvironments. By incorporating a spatially topology-enhanced data generation module and a dynamic graph attention network, STD-Net achieves superior performance in identifying rare cell types and preserving spatial domain integrity. The implementation demonstrates robust capability across multiple platforms including seqFISH, seqFISH+, and MERFISH datasets.

![图片1](C:\Users\冷寻非\Desktop\图片1.png)

## **Requirements**  
numpy==1.26.4
pandas==2.2.2
scipy==1.13.0
matplotlib==3.8.3
seaborn==0.13.2

scanpy==1.10.1
anndata==0.10.6
scanorama==1.7.3

--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.0+cu121
torch_geometric==2.5.3
tensorflow==2.16.1

networkx==3.2.1
shap==0.45.1
lime==0.2.0.1

umap-learn==0.5.5
scikit-learn==1.4.2

tqdm==4.66.2
multiprocess==0.70.16

nvidia-cudnn-cu12==8.9.6.50
nvidia-cublas-cu12==12.4.0.0
nvidia-cuda-nvrtc-cu12==12.1.105

--extra-index-url https://data.pyg.org/whl/torch-2.3.0+cu121.html
torch-scatter==2.1.2+cu121
torch-sparse==0.6.18+cu121
torch-cluster==1.6.3+cu121

## **Run STD_Net**  
STD-Net is designed for straightforward execution. The system offers a one-click operation mode. Simply execute the following command to start the complete workflow, including data preprocessing, model training, evaluation, and visualization: "python Run_STD.py". If you are using an IDE like PyCharm or a code editor, you can directly open and run the `Run_STD.py`script from within your development environment.

## **The dataset files (example)**  

#### ../final_data/

To facilitate readers in quickly setting up and debugging the project, we provide a subset of the seqFISH dataset within the `finaldata`directory. Due to the large size of the complete original dataset, this example includes only three selected slices for demonstration purposes (The seqFISH dataset slice names followed the rule "embryo[number]_z[slice number]" (such as embryo1_z2) (Lohoff et al., 2021), where the number represents the embryonic individual and the z value indicates the position of the slice along the head-tail axis of the embryo. ), which are:

•	embryo1_z2
•	embryo1_z5
•	embryo2_z2 
For access to the full dataset and a detailed description of the complete data processing pipeline, please refer to our associated research paper.

## **Output files**  

**After the training is completed, corresponding folders will be generated under the "system_result" directory based on the research OP taxon and training parameters, such as "Lepidoptera_10.0_SVM_5".**

#### ../results/ 

•	marker_genes.tsv: Contains the list of cell type-specific marker genes used for model training.
•	Loss_function.jpg: Image file displaying the loss function curves for the training, validation, and test datasets, used for monitoring the training process and model convergence.
•	metrics_<slice_id>.txt: Text file recording the detailed quantitative evaluation results (e.g., JSD, RMSE, Spearman's correlation coefficient) for the specified slice (e.g., embryo1_z2).
•	metrics_chart_<slice_id>.png: This image file presents the performance evaluation indicators of the specified slice in a bar chart format in a clear and intuitive manner.
•	model_parameters.pth: Directory for storing the trained parameter weights of the STD-Net model, which can be loaded for subsequent prediction or continued training.
•	predict_result.csv: Core output file containing the predicted cell type proportions for each spot in the spatial transcriptomics data.
•	umap_<slice_id>: Directory storing the UMAP dimensionality reduction visualization results for the specified slice, used for intuitively comparing the spatial structure consistency between the predicted and true cell type distributions. 
