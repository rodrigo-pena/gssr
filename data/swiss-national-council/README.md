# README

# Swiss National Council



### Dataset information

| Dataset statistics |       |
| :----------------- | ----- |
| Nodes              | 200   |
| Edges              | 19900 |

### Instructions to reproduce the files

`council_50_info.csv`:

- Go to https://www.parlament.ch/en/ratsbetrieb/abstimmungen/abstimmungs-datenbank-nr.
- Search the database for the "50E LEGISLATURE", with all other parameters set to "ALL".
- 

`council_members_since_1848.csv`:

- Go to https://www.parlament.ch/en/ratsmitglieder.
- Click on [EXPORT HITLIST AS EXCEL FILE](https://www.parlament.ch/_layouts/15/itsystems.pd.internet/documents/CouncilMembersExcelExportHandler.ashx?language=EN&filename=Ratsmitglieder_1848_EN.xlsx).  
- Open the downloaded `.xlsx` file.
- Export the opened file as  `council_members_since_1848.csv`. 

### Files

| File                             | Description                                                  |
| :------------------------------- | :----------------------------------------------------------- |
| `council_50_info.csv`            | Data frame with information about members of the 50th Legistature |
| `adjacency_50.csv`               | Adjacency matrix of the voting-similarity graph for the members of the 50e Legistature |
| `council_members_since_1848.csv` |                                                              |

### Data format



### Acknowledgements

- The idea for gathering this data came from the students of the [NTDS'18][ntds2018] project [Voting patterns in the Swiss National Council][swiss_council].



[swiss_council]: https://github.com/nikolaiorgland/conseil_national
[ntds2018]: https://github.com/mdeff/ntds_2018

