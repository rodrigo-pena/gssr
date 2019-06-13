# High school contact and friendship networks

Source: [SocioPatterns][sociopatterns].

## Instructions to reproduce the files

1. Go to [dataset website][data-website] and download all the files at the bottom of the page.
2. Extract the downloaded files to this directory.


## Files

| File                                    | Description                                                  |
| :-------------------------------------- | :----------------------------------------------------------- |
| `Contact-diaries-network_data_2013.csv` | Weighted edgelist representing a directed network of contacts between students. Each line has the form `i j w`, indicating that the student of ID `i` reported contacts, with the student of ID `j`, of aggregate durations of (i) at most 5 min if `w = 1`, (ii) between 5 and 15 min if `w = 2`, (iii) between 15 min and 1 h if `w = 3`, (iv) more than 1 h if `w = 4`. |
| `Facebook-known-pairs_data_2013.csv`    | Weighted edgelist of pairs of students for which the presence or absence of a Facebook friendship is known. Each line has the form `i j w`, where `w=1` means that the students of IDs `i` and `j` are linked on Facebook, while `w=0` means that they are not. |
| `Friendship-network_data_2013.csv`      | Edgelist representing a directed network of reported friendships. Each line has the form `i j`, meaning that the student of ID `i` reported a friendship with the student of ID `j`. |
| `High-School_data_2013.csv`             | Tab-separated list representing the active contacts during 20-second intervals of the data collection. Each line has the form `t i j Ci Cj`, where `i` and `j` are anonymous IDs of the persons in contact, `Ci` and `Cj` are their classes, and the interval during which this contact was active is `[ t – 20s, t ]`. If multiple contacts are active in a given interval, you will see multiple lines starting with the same value of `t`. Time is measured in seconds. |
| `metadata_2013.txt`                     | ID, class and gender of each person                          |

## References

When using this data, please cite the following paper:

[R. Mastrandrea, J. Fournet, A. Barrat,
Contact patterns in a high school: a comparison between data collected using wearable sensors, contact diaries and friendship surveys.
PLoS ONE 10(9): e0136497 (2015)](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0136497)



[data-website]: http://www.sociopatterns.org/datasets/high-school-contact-and-friendship-networks/