# Swiss National Council

Using voting patterns to build a similarity graph between members of the Swiss National Council.


## Instructions to reproduce the files

1. Go to the [DATABASE OF PARLIAMENTARY VOTES][dbpv] and issue a search for the "50E LEGISLATURE", with all other parameters set to "ALL".
2. Scroll to the bottom of the page, select the export format `CSV (all search results)`, and click the "Export" button.
3. Save the exported file in this directory as `abdb-en-all-affairs-50-0.csv`.
4. Now go to the to page [COUNCIL MEMBERS SINCE 1848][cms1848] and click on "Export hitlist as Excel file".
5. Open the the downloaded `.xlsx` file and export it as `Ratsmitglieder_1848_EN.csv` to this directory.
6. Run `python3 parse_council_data.py` to rebuild the files `councillors.csv`, `affairs.csv`, `voting_matrix.csv`.


## Files

| File                             | Description                                                     |
| :------------------------------- | :-------------------------------------------------------------- |
| `councillors.csv`                | Information about council members of the 50th Legistature       |
| `affairs.csv`                    | Information on the voting affairs of the 50th Legistature       |
| `voting_matrix.csv`              | A matrix indicating how each council member voted in each affair|


## Data format

`councillors.csv`:

- `CouncillorId`: ID of the council member
- `CouncillorName`: Name of the council member
- `Active`: Council member is active (True) or not (False) at the moment of gathering the data
- `GenderAsString`: Council member is male ('m') or female ('f')
- `CantonAbbreviation`: Acronym of the Canton from which the council member comes
- `ParlGroupAbbreviation`: Acronym of the parliamentary group in which the council member belongs
- `PartyAbbreviation`: Acronym of the party in which the council member belongs
- `MaritalStatusText`: Marital status of the council member
- `DateJoining`: When the council member joined the nacional council
- `DateLeaving`: When the council member left the nacional council
- `DateOfBirth`: When the council member was born
- `DateOfDeath`: When the council member was died

`affairs.csv`:

- `AffairShortId`: ID of the voting affair
- `AffairTitle`: Title of the votin affair
- `VoteDate`: When the vote took place
- `VoteMeaningYes`: Summary of what a "Yes" vote means
- `VoteMeaningNo`: Summary of what a "No" vote means

`voting_matrix.csv`:

A matrix with `#Councillors` rows and `#Affairs` columns indicating how each council member voted on which affair. A value of `+1.0` means a "Yes" vote, and a value of `-1.0` means a "No" vote. Any other vote status (abstained, not participated, excused, etc.) was given a value of `0.0`.


## Acknowledgements

- The idea for using and processing this data came from the Diego Debruyn, Yann Morize, Nikolai Orgland, and Silvan Stettler, [NTDS'18][ntds2018] students working in the project [Voting patterns in the Swiss National Council][swiss_council].



[dbpv]: https://www.parlament.ch/en/ratsbetrieb/abstimmungen/abstimmungs-datenbank-nr
[cms1848]: https://www.parlament.ch/en/ratsmitglieder
[swiss_council]: https://github.com/nikolaiorgland/conseil_national
[ntds2018]: https://github.com/mdeff/ntds_2018

