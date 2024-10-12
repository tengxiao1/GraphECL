## GraphECL: Towards Efficient Contrastive Learning for Graphs (ICML 2024)


### Required Packages

- CUDA Version: 12.2
- dgl==1.1.2+cu117
- matplotlib==3.7.3
- networkx==3.1
- numpy==1.24.3
- seaborn==0.13.0
- torch==1.13.0
- torch_geometric==2.3.1
- tqdm==4.66.1
- ogb==1.3.2



### Run Experiments
-  homophilous graphs 
```
sh homophilous/run.sh
sh homophilous/run_ind.sh
```


- heterophilous graphs 
```
sh heterophilous/run.sh
sh heterophilous/run_ind.sh
```

##  Reference

If you find this repo to be useful, please cite our paper:
```bibtex
@inproceedings{xiaoefficient,
  title={Efficient Contrastive Learning for Fast and Accurate Inference on Graphs},
  author={Xiao, Teng and Zhu, Huaisheng and Zhang, Zhiwei and Guo, Zhimeng and Aggarwal, Charu C and Wang, Suhang and Honavar, Vasant G},
  booktitle={Forty-first International Conference on Machine Learning (ICML)},
  year={2024}
}
