### 调用方法
from AEDPC import AEDPC   
aedpc = AEDPC(sample_ratio=1, k_min=n, k_max=m) # n,m可自行设置   
aedpc.fit(x)   
aedpc_labels = np.array(aedpc.labels_) # 算法返回标签   ### Method Call
from AEDPC import AEDPC   
aedpc = AEDPC(sample_ratio=1, k_min=n, k_max=m) # n,m can be set as needed   
aedpc.fit(x)
aedpc_labels = np.array(aedpc.labels_) # Labels returned by the algorithm
