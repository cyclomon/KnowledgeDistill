# KnowledgeDistillation

## Training Teacher Network
    python main.py 
## Training Student Network without Knowledge Distillation
    python main.py --student
 You can change hyperparameter settings & model save directory
 
    python main.py --student --save-dir './checkpoints' --alpha 0.9 --temp 3
    
## Results
### Teacher Network
![teacheracc](./img/accuracy_fig.png)

 Teacher Network Accuracy Graph
 Highest Accuracy : 93.08%
 
![teacherloss](./img/loss_fig.png)

 Teacher Network Loss Graph
 
### Student Network without KD VS with KD 
![KDvsNOKDacc](./img/accuracy_stu_vs_KD.png)
 accuracy comparison
 Student Network without KD accuracy : 89.07%
 Student Network with KD accuracy : 88.9%

![KDvsNOKDacc](./img/loss_stu_vs_KD.png)
 loss comparison
