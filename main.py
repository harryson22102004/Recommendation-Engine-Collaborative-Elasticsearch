import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
 
class CollaborativeFilter:
    def __init__(self, n_factors=20, lr=0.01, reg=0.01, n_iter=100):
        self.k=n_factors; self.lr=lr; self.reg=reg; self.n_iter=n_iter
    def fit(self, R):
        m,n=R.shape; self.P=np.random.randn(m,self.k)*0.1; self.Q=np.random.randn(n,self.k)*0.1
        mask=(R>0)
        for _ in range(self.n_iter):
            E=mask*(R-self.P@self.Q.T)
            self.P+=self.lr*(E@self.Q-self.reg*self.P)
            self.Q+=self.lr*(E.T@self.P-self.reg*self.Q)
        self.R_hat=np.clip(self.P@self.Q.T,1,5)
    def recommend(self, user_id, top_k=5, seen=None):
        scores=self.R_hat[user_id].copy()
        if seen: scores[list(seen)]-=1e9
        return np.argsort(scores)[::-1][:top_k].tolist()
 
class ContentBasedFilter:
    def fit(self, item_features): self.feat=item_features; self.sim=cosine_similarity(item_features)
    def recommend(self, liked_items, top_k=5):
        scores=self.sim[liked_items].mean(axis=0)
        scores[liked_items]-=1e9
        return np.argsort(scores)[::-1][:top_k].tolist()
 
class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        self.cf=CollaborativeFilter(); self.cb=ContentBasedFilter()
        self.w_cf=cf_weight; self.w_cb=cb_weight
    def fit(self, R, item_features): self.cf.fit(R); self.cb.fit(item_features)
    def recommend(self, user_id, liked_items, top_k=5):
        cf_scores=self.cf.R_hat[user_id]
        cb_scores=self.cb.sim[liked_items].mean(axis=0)
        hybrid=(self.w_cf*cf_scores/cf_scores.max() + self.w_cb*cb_scores/(cb_scores.max()+1e-8))
      hybrid[liked_items]-=1e9
        return np.argsort(hybrid)[::-1][:top_k].tolist()
 
np.random.seed(42)
n_users,n_items=100,200
R=np.random.choice([0,1,2,3,4,5],(n_users,n_items),p=[0.7,.06,.06,.06,.06,.06]).astype(float)
item_features=np.random.randn(n_items,20)
rec=HybridRecommender(); rec.fit(R,item_features)
liked=[5,12,30,45]
recs=rec.recommend(0,liked,top_k=5)
print(f"Liked items: {liked}")
print(f"Hybrid recommendations: {recs}")
cf_recs=rec.cf.recommend(0,top_k=5,seen=set(liked))
print(f"CF-only recommendations: {cf_recs}")
