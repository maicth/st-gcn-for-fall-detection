# Introduction
Skeleton-based fall detection using Computer Vision develops a model using ST-GCN combines with transfer learning technique and attention mechanism to detect falls more accurately. 

# Data
- Use NTU RGB+D to pre-train ST-GCN model
- Train and test in 2 datasets: TST v2 and FallFree

# Implementation
- The pre-trained ST-GCN model is implemented in processor/recognition.py
- Temporal attention mechanism is implemented in net/utils/sfd-gcn.py and it is used in net/st-gcn.py, after 9 layers of st-gcn
- For other configurations, see OLD_README.md

# Report
https://docs.google.com/document/d/1mnoCdjwXcPp2IVADUCDGD_NM1B_ndFZ7-qx6ip-2fgg
