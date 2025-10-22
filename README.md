### 调用方法：

```python
from AEDPC import AEDPC

aedpc = AEDPC(sample_ratio=1, k_min=n, k_max=m)  # n,m可自行设置
aedpc.fit(x)
aedpc_labels = np.array(aedpc.labels_)  # 算法返回标签
