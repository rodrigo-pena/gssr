# README

Source: [http://snap.stanford.edu/data/email-Eu-core.html](http://snap.stanford.edu/data/email-Eu-core.html)

### Dataset information

The network was generated using email data from a large European research institution. We have anonymized information about all incoming and outgoing email between members of the research institution. There is an edge (u, v) in the network if person u sent person v at least one email. The e-mails only represent communication between institution members (the core), and the dataset does not contain incoming messages from or outgoing messages to the rest of the world.

The dataset also contains "ground-truth" community memberships of the nodes. Each individual belongs to exactly one of 42 departments at the research institute.

This network represents the "core" of the [email-EuAll](http://snap.stanford.edu/data/email-EuAll.html) network, which also contains links between members of the institution and people outside of the institution (although the node IDs are not the same).

| Dataset statistics               |               |
| :------------------------------- | ------------- |
| Nodes                            | 1005          |
| Edges                            | 25571         |
| Nodes in largest WCC             | 986 (0.981)   |
| Edges in largest WCC             | 25552 (0.999) |
| Nodes in largest SCC             | 803 (0.799)   |
| Edges in largest SCC             | 24729 (0.967) |
| Average clustering coefficient   | 0.3994        |
| Number of triangles              | 105461        |
| Fraction of closed triangles     | 0.1085        |
| Diameter (longest shortest path) | 7             |
| 90-percentile effective diameter | 2.9           |

### Source (citation)

- Hao Yin, Austin R. Benson, Jure Leskovec, and David F. Gleich. "Local Higher-order Graph Clustering." In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2017.
- J. Leskovec, J. Kleinberg and C. Faloutsos. [Graph Evolution: Densification and Shrinking Diameters](http://www.cs.cmu.edu/~jure/pubs/powergrowth-tkdd.pdf). ACM Transactions on Knowledge Discovery from Data (ACM TKDD), 1(1), 2007.

### Files

| File                                                         | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| [email-Eu-core.txt.gz](http://snap.stanford.edu/data/email-Eu-core.txt.gz) | Email communication links between members of the institution |
| [email-Eu-core-department-labels.txt.gz](http://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz) | Department membership labels                                 |

### Data format for community membership

NODEID DEPARTMENT

- `NODEID`: id of the node (a member of the institute)
- `DEPARTMENT`: id of the member's department (number in 0, 1, ..., 41)