model = BipartiteGNN(d_u, d_p, hidden_dim=128, out_dim=ℓ, dropout=0.3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.BCEWithLogitsLoss()

model.train()
for epoch in range(1, 201):
    optimizer.zero_grad()
    logits = model(x_u, x_p, edge_index)      # [m × ℓ]
    loss   = criterion(logits[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
