import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---- Parameters ----
grid_size = 100
num_random = 10000
U_inlet = 1.0

# ---- Block and Domain Definition ----
x_min, x_max = -5.0, 5.0
y_min, y_max = 0.0, 10.0
block_x_min, block_x_max = -0.5, 0.5
block_y_min, block_y_max = 0.0, 1.0

# ---- Collocation Points (biased toward ground) ----
x_random = (x_max - x_min) * torch.rand((num_random, 1), device=device) + x_min
y_random = y_max * (1.0 - torch.rand((num_random, 1), device=device) ** 0.3)

in_block_x = (x_random >= block_x_min) & (x_random <= block_x_max)
in_block_y = (y_random >= block_y_min) & (y_random <= block_y_max)
mask_block = ~(in_block_x & in_block_y)

x_collocation = x_random[mask_block]
y_collocation = y_random[mask_block]


# Convert collocation tensors to CPU NumPy arrays for plotting
x_col_np = x_collocation.cpu().numpy()
y_col_np = y_collocation.cpu().numpy()

# Collocation points (random)

# ---- Boundary Points ----
x_wall = torch.linspace(x_min, x_max, grid_size, device=device).view(-1, 1)
y_wall = torch.linspace(y_min, y_max, grid_size, device=device).view(-1, 1)

x_left = x_min * torch.ones_like(y_wall)
y_left = y_wall

x_right = x_max * torch.ones_like(y_wall)
y_right = y_wall

x_top = x_wall
y_top = y_max * torch.ones_like(x_top)

x_bottom = x_wall
y_bottom = y_min * torch.ones_like(x_bottom)

# Block walls
x_block_vert = torch.linspace(block_y_min, block_y_max, grid_size, device=device).view(-1, 1)
y_block_left = block_x_min * torch.ones_like(x_block_vert)
y_block_right = block_x_max * torch.ones_like(x_block_vert)

x_block_top = torch.linspace(block_x_min, block_x_max, grid_size, device=device).view(-1, 1)
y_block_top = block_y_max * torch.ones_like(x_block_top)



# ---- Define PINN Model ----
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 3)
        )

    def forward(self, x, y):
        x = x.view(-1, 1)
        y = y.view(-1, 1)
        inputs = torch.cat([x, y], dim=1)
        uvp = self.hidden(inputs)
        return uvp[:, 0:1], uvp[:, 1:2], uvp[:, 2:3]

# ---- PDE Residual ----
def pde_residual(x, y, model, nu=0.1):
    x.requires_grad_(True)
    y.requires_grad_(True)
    u, v, p = model(x, y)

    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    momentum_x = u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
    momentum_y = u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
    continuity = u_x + v_y

    return momentum_x, momentum_y, continuity

# ---- Training ----
model = PINN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5000

#loss_history = []  # this line to start recording loss per epoch
loss_history_total = []
loss_history_pde = []
loss_history_bc = [] 
loss_history_pref = []

for epoch in range(num_epochs):
    model.train()

    u_left, v_left, _ = model(x_left, y_left)
    loss_left = torch.mean((u_left - U_inlet) ** 2 + v_left ** 2)

    x_right.requires_grad_(True)
    u_right, v_right, p_right = model(x_right, y_right)
    u_x_right = torch.autograd.grad(u_right, x_right, torch.ones_like(u_right), create_graph=True)[0]
    v_x_right = torch.autograd.grad(v_right, x_right, torch.ones_like(v_right), create_graph=True)[0]
    p_x_right = torch.autograd.grad(p_right, x_right, torch.ones_like(p_right), create_graph=True)[0]
    loss_right = torch.mean(u_x_right ** 2 + v_x_right ** 2 + p_x_right ** 2) 

    y_top.requires_grad_(True)  # ✅ Must be BEFORE using y_top in model
    u_top, v_top, _ = model(x_top, y_top)

    # Compute ∂u/∂y to enforce zero tangential gradient (slip condition)
    u_y_top = torch.autograd.grad(u_top, y_top, torch.ones_like(u_top), create_graph=True)[0]

    # Slip condition: v = 0, ∂u/∂y = 0
    loss_top = torch.mean(v_top ** 2 + u_y_top ** 2)

    u_bottom, v_bottom, _ = model(x_bottom, y_bottom)
    loss_bottom = torch.mean(u_bottom ** 2 + v_bottom ** 2)

    u_bL, v_bL, _ = model(y_block_left, x_block_vert)
    u_bR, v_bR, _ = model(y_block_right, x_block_vert)
    u_bT, v_bT, _ = model(x_block_top, y_block_top)
    loss_block = torch.mean(u_bL ** 2 + v_bL ** 2 + u_bR ** 2 + v_bR ** 2 + u_bT ** 2 + v_bT ** 2)

    loss_bc = loss_left + loss_right + loss_top + loss_bottom + loss_block

    momentum_x, momentum_y, continuity = pde_residual(x_collocation, y_collocation, model)
    loss_pde = torch.mean(momentum_x ** 2) + torch.mean(momentum_y ** 2) + torch.mean(continuity ** 2)

    x_ref = torch.tensor([[4.5]], device=device)
    y_ref = torch.tensor([[5.0]], device=device)
    _, _, p_ref = model(x_ref, y_ref)
    loss_p_ref = torch.mean(p_ref ** 2)

    loss = 0.1 * loss_bc + 50 * loss_pde + 1.0 * loss_p_ref

    # Record each loss component
    loss_history_total.append(loss.item())
    loss_history_pde.append(loss_pde.item())
    loss_history_bc.append(loss_bc.item())
    loss_history_pref.append(loss_p_ref.item())


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, PDE: {loss_pde.item():.6f}, BC: {loss_bc.item():.6f}, Pref: {loss_p_ref.item():.6f}")

# ---- Plot Individual Loss Components ----
plt.figure(figsize=(10, 6))
epochs = np.arange(len(loss_history_total))
plt.plot(epochs, loss_history_total, label='Total Loss')
plt.plot(epochs, loss_history_pde, label='PDE Loss')
plt.plot(epochs, loss_history_bc, label='BC Loss')
plt.plot(epochs, loss_history_pref, label='Pref Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.title('Loss Components over Training Epochs', fontweight='bold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save trained model
torch.save(model.state_dict(), "pinn_model_weights.pth")
print("Model weights saved.")

# ---- Evaluation and Plotting ----
x_test = torch.linspace(x_min, x_max, grid_size)
y_test = torch.linspace(y_min, y_max, grid_size)
xg, yg = torch.meshgrid(x_test, y_test, indexing='xy')
x_full = xg.numpy()
y_full = yg.numpy()
x_flat = xg.reshape(-1, 1).to(device)
y_flat = yg.reshape(-1, 1).to(device)

model.eval()
with torch.no_grad():
    u_pred, v_pred, p_pred = model(x_flat, y_flat)
    u_pred = u_pred.cpu().numpy().reshape(grid_size, grid_size)
    v_pred = v_pred.cpu().numpy().reshape(grid_size, grid_size)
    p_pred = p_pred.cpu().numpy().reshape(grid_size, grid_size)
    velocity_mag = np.sqrt(u_pred**2 + v_pred**2)

# Block mask
mask = ~((x_full >= block_x_min) & (x_full <= block_x_max) & (y_full >= block_y_min) & (y_full <= block_y_max))
u_pred[~mask] = np.nan
v_pred[~mask] = np.nan
velocity_mag[~mask] = np.nan
p_pred[~mask] = np.nan

# Plot 1: Velocity Magnitude
plt.figure(figsize=(8, 6))
contour = plt.contourf(x_full, y_full, velocity_mag, levels=50, cmap='jet')
plt.colorbar(contour, label='Velocity Magnitude')
plt.title('Contour Plot of Velocity Magnitude', fontweight='bold')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Streamlines
plt.figure(figsize=(8, 6))
plt.streamplot(x_full, y_full, u_pred, v_pred, color=velocity_mag, cmap='jet', density=1.5)
plt.title('Streamlines of Velocity Field', fontweight='bold')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Quiver
plt.figure(figsize=(8, 6))
valid = ~np.isnan(u_pred)
plt.quiver(x_full[valid], y_full[valid], u_pred[valid], v_pred[valid], color='blue', angles='xy', scale_units='xy', scale=1)
plt.title('Velocity Field (u, v) - Quiver', fontweight='bold')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: Pressure
plt.figure(figsize=(6, 5))
plt.contourf(x_full, y_full, p_pred, levels=50, cmap='coolwarm')
plt.colorbar(label='Pressure')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Pressure Field - Block Obstacle Flow', fontweight='bold')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()
