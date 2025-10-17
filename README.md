 # Anonymous-RAL

Anonymous repository for double-blind review.  

This repository contains code and/or data used in our paper submission.  
All author-identifying information has been removed for the purpose of anonymous review.  

This is the Docker environment to reproduce the minimal validation results of our submission paper.  
Download checkpoints from:  
👉 https://huggingface.co/anonymous50531/RAL-review-model/tree/main  
Put the folders under `./checkpoints`.  

---

### 🐳 Docker

Please check the GPU setting and path.  
Test is done only on a single A6000 GPU.  

**Run:**
```bash
docker-compose build
docker-compose up -d tester_task1  # will generate task1.log
docker-compose up -d tester_task2  # will generate task2.log

docker-compose run -it it-debug  #You can also do to interactively run the scripts.


Conda:
Please refer to requirements.txt to setup your conda environment.
Then refer to docker-compose.yaml to run the corresponding file under your conda environment.




