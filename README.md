# KnowledgeDistillation

## Training Teacher Network
    python main.py 
## Training Student Network without Knowledge Distillation
    python main.py --student
 You can change hyperparameter settings & model save directory
 
    python main.py --student --save-dir './checkpoints' --alpha 0.9 --temp 3
## Results
![Accuracy_teacher](./.png)
