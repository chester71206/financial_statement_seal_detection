import re
import matplotlib.pyplot as plt

log_file="./log/log_2025-08-08_17-13-54.txt"
loss_values=[]
steps=[]
with open(log_file,"r",encoding="utf-8") as f:
    for line in f:
        match=re.search(r"Train step_(\d+): loss : ([\d\.]+)", line)
        if match:
            step=int(match.group(1))
            loss=float(match.group(2))
            steps.append(step)
            loss_values.append(loss)
            
plt.figure(figsize=(10,6))
plt.plot(steps,loss_values,marker="o",linewidth=1,markersize=1)
plt.xlabel("training step")
plt.ylabel("training loss")
plt.title("total loss over training steps")
plt.grid(True)
plt.tight_layout()
plt.show