# SurgBox
![Image](https://github.com/franciszchen/SurgBox/blob/main/work%20flow.png)

## 1. Environment setting
```
step 1.
pip install --upgrade metagpt

step 2.
metagpt --init-config

step 3. 
modify your own config.yaml as follow with your openai_key

llm:
  api_type: "openai"
  api_version: "2024-02-15-preview"
  base_url: "your_url"
  api_key: 'your_key'
  model: "openai"
```

## 2. Inference
```
step 1. cd ./MetaGPT-main/examples/clinic

step 2. python SurgBox.py
```

## Citation

If you use our code or paper in your work, please cite our paper.

```bibtex
@article{wu2024surgbox,
  title={SurgBox: Agent-Driven Operating Room Sandbox with Surgery Copilot},
  author={Wu, Jinlin and Liang, Xusheng and Bai, Xuexue and Chen, Zhen},
  journal={arXiv preprint arXiv:2412.05187},
  year={2024}
}

```

