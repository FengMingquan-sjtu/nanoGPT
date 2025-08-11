# a two layer MLP with a residual connection
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)  
        self.fc3 = nn.Linear(hidden_dim, output_dim, bias=False)


    def forward(self, x, do_coll=False):
        x1 = self.fc1(x) #input layer
        x2 = self.fc2(x1)**2 #non-linear transformation
        c = 0 #collision flag
        if not do_coll:
            x3 = x2 + x1 #residual connection
        else: #kinet
            y1 = x1[:,0]
            y2 = x1[:,1]
            v1 = x2[:,0]
            v2 = x2[:,1]
            
            if torch.sign(y2- y1) != torch.sign(y1+v1-(y2+v2)):
                t = (y1-y2) / (v2-v1)
                if torch.rand(1).item() > t:
                    y1_mid = y1 + v1 * t
                    y2_mid = y2 + v2 * t
                    y1_new = y1_mid + v2 * (1-t)
                    y2_new = y2_mid + v1 * (1-t)
                    x3 = torch.stack([y1_new, y2_new], dim=1)
                    c = 1
                else:
                    x3 = x2 + x1
            else:
                x3 = x2 + x1


        x4 = self.fc3(x3)  # final output layer
        return x1, x2, x3, x4, c




def main(seed=42, do_coll=False):
    torch.manual_seed(seed)
    print(f"Using seed: {seed}")
    print(f"Collision enabled: {do_coll}")
    # Example usage
    input_dim = 1
    hidden_dim = 2
    output_dim = 1
    train_step = 5000
    batch_size = 1
    lr = 5e-3

    model = MLP(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_step, eta_min=lr*0.1)
    coll_cnt = 0

    for i in range(train_step):
        
        # Simulate a batch of data
        x = torch.randn(batch_size, input_dim) * 0.1 + 0.1  # small noise + shift
        output1, output2, output3, output4, coll = model(x, do_coll=do_coll)
        y = x**2 + x
        loss = F.mse_loss(output4, y, do_coll) / y**2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        coll_cnt += coll

        if i==0 or (i + 1) % 100 == 0:
            with torch.no_grad():
                test_losses = []
                for _ in range(100):
                    x = torch.randn(1, input_dim) * 0.1  # small noise
                    output1, output2, output3, output4, c = model(x, do_coll=False)
                    y = x**2 + x
                    loss = F.mse_loss(output4, y) / y**2
                    test_losses.append(loss.item())
                avg_test_loss = sum(test_losses) / len(test_losses)
                print(f"Test loss at step {i+1}: {avg_test_loss:.4f}")
                print(f"---Step {i+1}/{train_step}---")
                print(f"Input : {x.item():.4f}")
                print(f"True y: {y.item():.4f}")
                print(f"Out x4: {output4.detach().squeeze():.4f}")
                print(f"Loss  : {loss.item():.4f}")
                w1 = model.fc1.weight.detach()
                print(f"fc1 weight: {w1.squeeze()}")
                print(f"x1: {output1.detach().squeeze()}")
                w2 = model.fc2.weight.detach()
                print(f"fc2 weight: {w2.squeeze()}")
                #print(f"x2: {output2.detach().squeeze():.4f}")
                #w1.shape = (2, 1), w2.shape = (1, 2)
                #print(f"coef: {torch.matmul(w2, w1**2).squeeze():.4f}")

                print(f"collision count: {coll_cnt}")
            

if __name__ == "__main__":
    main(7, do_coll=True)