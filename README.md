# Anonymous-RAL
Anonymous repository for double-blind review.

This repository contains code and/or data used in our paper submission.  
All author-identifying information has been removed for the purpose of anonymous review.

This is the docker to reproduce the minimal validation results of our submission paper.
Download checkpoints from  https://huggingface.co/anonymous50531/RAL-review-model/tree/main
Put the folders under ./checkpoints.

Docker:
Please checke the GPU setting and path. Test is done only on single A6000.
Run
docker-compose build
docker-compose up -d tester_task1 -> will generate task1.log
docker-compose up -d tester_task2 -> will generate task2.log

You can also do
docker-compose run -it it-debug 
to interactively run the scripts.

Conda:
Please refer to requirements.txt to setup your conda environment.
Then refer to docker-compose.yaml to run the corresponding file under your conda environment.




