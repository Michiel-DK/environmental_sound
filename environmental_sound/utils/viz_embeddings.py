import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Extract embeddings
frozen_encoder.eval()
embeddings = []
labels = []

for x, y in val_loader:
    with torch.no_grad():
        embedding = frozen_encoder(x).cpu().numpy()
        embeddings.append(embedding)
        labels.append(y.cpu().numpy())

embeddings = np.vstack(embeddings)
labels = np.hstack(labels)

# t-SNE visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
plt.colorbar()
plt.title("t-SNE Visualization of Embeddings")
plt.show()
