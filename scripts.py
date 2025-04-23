import torch
from datetime import datetime
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_loss(model, u, t, model_name="fno"): 
    x = model(u, t)
    if model_name=="lno":
        dt=(t[1]-t[0]).item()
    elif model_name=="fno":
        dt=(t[0,1]-t[0, 0]).item()
    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt  # crude finite diff

    if model_name=="lno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1, :]
    elif model_name=="fno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1].unsqueeze(-1)

    physics_loss = torch.mean(residual ** 2)
    initial_loss = torch.mean((x[:, 0, :] - 1.0)**2)
    return physics_loss, initial_loss


def train_fno(model, dataloader, optimizer, scheduler, epochs, t_grid, alpha = 1, beta = 1, folder = "trained_models/fno", logging = True):

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for u, _, _, _ in dataloader:

            u = u.to(device)
            t = t_grid[:u.shape[0], :].to(device)
            optimizer.zero_grad()
            
            physics_loss, initial_loss = compute_loss(model, u, t, model_name="fno")
            loss = alpha*physics_loss + beta*initial_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if logging:
                print(f'Epoch [{epoch+1}/{epochs}], Physics loss: {physics_loss}, Initial loss: {initial_loss}, Time: {datetime.now().time()}')

        
        epoch_loss /= n_batches
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Time: {datetime.now().time()}') 
              
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        os.makedirs(folder, exist_ok=True)
        torch.save(model.state_dict(), folder+f"/epoch-[{epoch+1}]_model_{timestamp}_loss{epoch_loss:.4f}.pth")
        scheduler.step()

    return model

def train_lno(model, dataloader, optimizer, scheduler, epochs, alpha = 1, beta = 1, folder = "trained_models/lno", logging = True):

    for epoch in range(epochs):
        
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for u, t, _, _ in dataloader:

            u = u.to(device).unsqueeze(-1)
            optimizer.zero_grad()

            physics_loss, initial_loss = compute_loss(model, u, t, model_name="lno")
            loss = alpha * physics_loss + beta * initial_loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if logging:
                print(f"Epoch: {epoch}, Physics loss: {physics_loss}, Initial loss: {initial_loss}")

        epoch_loss /= n_batches
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}, Time: {datetime.now().time()}') 

        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            os.makedirs(folder, exist_ok=True)
            model_filename = f'epochs_[{epoch+1}]_model_time_[{timestamp}]_loss_[{epoch_loss:.4f}].pth'
            torch.save(model.state_dict(), folder+f"/{model_filename}")

        scheduler.step()

def objective_function(args, model_name):

    x = args['x']
    u = args['u']
    t = args['t']
    w = args['w']

    if model_name=="lno":
        dt = (t[1] - t[0]).item()
    elif model_name=="fno":
        dt = (t[0, 1] - t[0, 0]).item()


    dx_dt = (x[:, 1:, :] - x[:, :-1, :]) / dt  # crude finite diff

    if model_name=="lno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1, :]
    elif model_name=="fno":
        residual = dx_dt + x[:, :-1, :] - u[:, :-1].unsqueeze(-1)

    physics_loss = torch.mean(residual ** 2)
    initial_loss = torch.mean((x[:, 0] - torch.ones(10, device = device))**2)
    control_cost = torch.mean(torch.trapz((x**2 + u**2).squeeze(), t.squeeze()))

    J = w[0] * control_cost + w[1] * physics_loss + w[2] * initial_loss 
    return J

def optimize_neural_operator(model, objective_function, m, end_time, num_epochs, learning_rate, w=[1,1,1], model_name="lno"):  

    if model_name=="lno":
        u = (0.2 * torch.randn(1, m, device=device)).unsqueeze(-1)  # shape = (1, m, 1)
        t = torch.linspace(0, end_time, m, device= device)
    elif model_name=="fno":
        u = (0.2 * torch.randn(1, m, device=device))  # shape: [1, m]
        t = torch.linspace(0, end_time, m, device=device).unsqueeze(0).unsqueeze(-1)  # shape: [1, m, 1]

    u.requires_grad_(True)
    optimizer = torch.optim.Adam([u], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)

    for epoch in range(num_epochs):

        optimizer.zero_grad()
        x = model(u, t)

        args = {'u': u, 't': t, 'x':x, 'w':w}
        loss = objective_function(args, model_name)
        loss.backward()
        optimizer.step()


        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, time: {datetime.now().time()}')

        scheduler.step()

    return u, x, t
