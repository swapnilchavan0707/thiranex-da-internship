import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ========== 0. GLOBAL FONT SETUP ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
sns.set_style("whitegrid", {'font.family':'serif', 'font.serif':['Times New Roman']})

# ========== 1. LOAD ==========
URL = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Mall_Customers.csv"
df = pd.read_csv(URL)
print(f"Dataset loaded: {df.shape}")
df = df.rename(columns={
    'Annual Income (k$)': 'Annual_Income_k',
    'Spending Score (1-100)': 'Spending_Score'
})
print(df.head())

# ========== 2. EDA ==========
g = sns.pairplot(df[['Age','Annual_Income_k','Spending_Score']],
                 diag_kind='kde',
                 height=2.5,
                 aspect=1.1,
                 plot_kws={'alpha':0.7, 's':45, 'edgecolor':'white', 'linewidth':0.5},
                 diag_kws={'shade':True, 'linewidth':1.5, 'color':'steelblue'})

g.fig.set_size_inches(11, 9)

# Force Times New Roman on every label inside pairplot
for ax in g.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontname='Times New Roman', fontsize=11, labelpad=8)
    ax.set_ylabel(ax.get_ylabel(), fontname='Times New Roman', fontsize=11, labelpad=8)
    for label in ax.get_xticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)
    for label in ax.get_yticklabels():
        label.set_fontname('Times New Roman')
        label.set_fontsize(10)

g.fig.suptitle('Exploratory Data Analysis - Mall Customers',
               y=0.98, fontname='Times New Roman', fontsize=15, fontweight='bold')
g.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.97, hspace=0.35, wspace=0.35)
g.fig.savefig('eda_pairplot.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
plt.show()

# ========== 3. FEATURES & SCALING ==========
features = ['Age', 'Annual_Income_k', 'Spending_Score']
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 4. ELBOW METHOD ==========
inertia = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax.plot(K, inertia, 'o-', color='#2C3E50', linewidth=2.5, markersize=9, markeredgecolor='white', markeredgewidth=1)
ax.set_xlabel('Number of Clusters (k)', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontname='Times New Roman', labelpad=12)
ax.set_title('Elbow Method for Optimal k\n(Mall Customers Dataset)', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xticks(K)
# Force tick font
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
ax.grid(True, alpha=0.3, linestyle='--')
fig.savefig('elbow_method.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 5. FINAL CLUSTERING ==========
k_optimal = 5
kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
df['Segment'] = kmeans.fit_predict(X_scaled)
print(f"Silhouette Score for k={k_optimal}: {silhouette_score(X_scaled, df['Segment']):.3f}")
print(df['Segment'].value_counts().sort_index())

# ========== 6. VISUALIZE - INCOME VS SPENDING ==========
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
sns.scatterplot(data=df, x='Annual_Income_k', y='Spending_Score', hue='Segment', palette='tab10', s=150, edgecolor='black', linewidth=0.7, alpha=0.9, ax=ax)

# Correct centroids (inverse transform)
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids_original[:,1], centroids_original[:,2], s=400, c='red', marker='X', edgecolors='black', linewidths=1.5, label='Centroids', zorder=5)

ax.set_title('Customer Segments (k=5)\nAnnual Income vs Spending Score', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Annual Income (k$)', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Spending Score (1-100)', fontname='Times New Roman', labelpad=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
ax.legend(title='Segment', title_fontsize=11, frameon=True, loc='best')
for text in ax.get_legend().get_texts():
    text.set_fontname('Times New Roman')
ax.get_legend().get_title().set_fontname('Times New Roman')
ax.grid(True, alpha=0.3, linestyle='--')
fig.savefig('segments_income_spending.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 7. VISUALIZE - AGE VS SPENDING ==========
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
sns.scatterplot(data=df, x='Age', y='Spending_Score', hue='Segment', palette='tab10', s=150, edgecolor='black', linewidth=0.7, alpha=0.9, ax=ax)
ax.set_title('Customer Segments\nAge vs Spending Score', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Age (Years)', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Spending Score (1-100)', fontname='Times New Roman', labelpad=12)
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Times New Roman')
ax.legend(title='Segment', frameon=True, loc='best')
for text in ax.get_legend().get_texts():
    text.set_fontname('Times New Roman')
ax.get_legend().get_title().set_fontname('Times New Roman')
ax.grid(True, alpha=0.3, linestyle='--')
fig.savefig('segments_age_spending.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 8. PROFILE & HEATMAP ==========
profile = df.groupby('Segment').agg({
    'Age':'mean',
    'Annual_Income_k':'mean',
    'Spending_Score':'mean',
    'Gender':'count'
}).rename(columns={'Gender':'Count'}).round(1)
print("\n=== SEGMENT PROFILE ===")
print(profile)

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
sns.heatmap(profile[['Age','Annual_Income_k','Spending_Score']],
            annot=True, cmap='Blues', fmt='.1f', linewidths=0.8, linecolor='white',
            annot_kws={"fontname":"Times New Roman", "fontsize":11, "fontweight":"bold"},
            cbar_kws={'label': 'Mean Value'}, ax=ax)
ax.set_title('Segment Characteristics - Mean Values', fontname='Times New Roman', fontsize=13, fontweight='bold', pad=20)
ax.set_xlabel('Features', fontname='Times New Roman', labelpad=12)
ax.set_ylabel('Segment ID', fontname='Times New Roman', labelpad=12)
# Fix heatmap labels font
for label in ax.get_xticklabels():
    label.set_fontname('Times New Roman')
    label.set_rotation(0)
for label in ax.get_yticklabels():
    label.set_fontname('Times New Roman')
    label.set_rotation(0)
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Mean Value', fontname='Times New Roman')
for label in cbar.ax.get_yticklabels():
    label.set_fontname('Times New Roman')
fig.savefig('segment_heatmap.png', dpi=300, bbox_inches='tight', pad_inches=0.4)
plt.show()

# ========== 9. SAVE ==========
df.to_csv('segmented_Mall_Customers.csv', index=False)
print("\n Output Images: eda_pairplot.png, elbow_method.png, segments_income_spending.png, segment_heatmap.png")