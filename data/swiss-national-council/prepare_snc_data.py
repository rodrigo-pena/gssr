# -*- coding: utf-8 -*-

"""Prepare Swiss National Council data

"""


import os

import numpy as np
import pandas as pd

from argparse import ArgumentParser


if __name__ == "__main__":
    
    # Parse arguments
    parser = ArgumentParser(description='Get council info and voting adjacency matrix.')
    
    parser.add_argument('-p', 
                        action='store', 
                        nargs='?', 
                        default='', 
                        type=str, 
                        help="path to the folder containing the voting and council info files" )
    
    parser.add_argument('-vf', 
                        action='store', 
                        nargs='?', 
                        default='abdb-en-all-affairs-50-0.csv',
                        type=str, 
                        help='voting file name')
    
    parser.add_argument('-cf', 
                        action='store', 
                        nargs='?', 
                        default='Ratsmitglieder_1848_EN.csv', 
                        type=str, 
                        help='number of random trials for each grid point')
    
    args = parser.parse_args()
    
    
    #Read `.csv` files. Setting the proper encoding for these files is VERY important
    
    council_info = pd.read_csv(os.path.join(args.p, args.cf), encoding='mac_roman') 
    council_info['CouncillorName'] = council_info['LastName'] + ' ' + council_info['FirstName'] 
    council_info = council_info.drop(columns=['LastName', 'FirstName'])
    council_info = council_info.drop_duplicates(subset='CouncillorName', keep='first')
    
    voting_data = pd.read_csv(os.path.join(args.p, args.vf), sep=',', 
                          encoding='utf-8', engine='c', low_memory=False, 
                          error_bad_lines=False, warn_bad_lines=False)

    # Remove "Motion d'ordre/Ordnungsanträge" because is has uninteresting votes
    voting_data = voting_data[~((voting_data.AffairShortId == 1) | (voting_data.AffairShortId == 2))]
    
    # Set a more "friendly" date format
    voting_data[['VoteDate']] = voting_data[['VoteDate']].apply(lambda d: d.str.slice(4,15))
    
    # Merge council info with the IDs from the voting file
    councillors = voting_data[['CouncillorId','CouncillorName']].drop_duplicates(keep='first')
    councillors = councillors.set_index('CouncillorId')
    councillors = councillors.sort_index()
    councillors = councillors.merge(council_info, how='inner', on='CouncillorName', right_index=True)
    councillors = councillors.drop(columns=['CouncilName', 'ParlGroupName', 'PartyName',
                                            'BirthPlace_City', 'BirthPlace_Canton', 'Mandates',
                                            'Citizenship', 'CantonName'])
    
    # Standardize the naming of certain parties
    councillors.loc[councillors['PartyAbbreviation'] == 'PLR', 'PartyAbbreviation'] = 'FDP'
    councillors.loc[councillors['PartyAbbreviation'] == 'PRD', 'PartyAbbreviation'] = 'FDP'
    councillors.loc[councillors['PartyAbbreviation'] == 'pvl', 'PartyAbbreviation'] = 'PVL'
    councillors.loc[councillors['PartyAbbreviation'] == 'PLS', 'PartyAbbreviation'] = 'FDP'
    councillors.loc[councillors['PartyAbbreviation'] == 'csp-ow', 'PartyAbbreviation'] = 'CSPO'
    councillors.loc[councillors['PartyAbbreviation'] == 'MCR', 'PartyAbbreviation'] = 'MCG'
    councillors.loc[councillors['PartyAbbreviation'] == 'BastA', 'PartyAbbreviation'] = 'PES'
    councillors.loc[councillors['PartyAbbreviation'] == 'PdT', 'PartyAbbreviation'] = 'PST'
    councillors.loc[councillors['PartyAbbreviation'] == 'CVPO', 'PartyAbbreviation'] = 'PDC'
    
    # Assign colors to councillors via their parties
    # Source: en.wikipedia.org/wiki/List_of_political_parties_in_Switzerland
    party_color_map = {'UDC': '#13923E',
                       'PSS': '#DB182A',
                       'FDP': '#0E3D8F',
                       'PDC': '#E96807',
                       'PBD': '#FED809',
                       'PES': '#73A812',
                       'PVL': '#97C834',
                       'PEV': '#FDD80B',
                       'Lega': '#527FE8',
                       'MCG': '#FDE609',
                       'PST': '#E02416',
                       'CSPO': '#AF1E28',
                       'CSP': '#168397',
                       'UDF': '#B80072',
                       'AL': '#820013'
                       } 
    
    colors = np.asarray(['#000000'] * len(councillors))  # Default: black
    
    for key in party_color_map:
        mask = (councillors['PartyAbbreviation'] == key).values
        colors[mask] = party_color_map[key]
    
    councillors.loc[:, 'PartyColor'] = colors
    
    # Create a data frame to record each voting affair
    affairs = voting_data[['AffairShortId',
                           'AffairTitle', 
                           'VoteDate', 
                           'VoteMeaningYes', 
                           'VoteMeaningNo']].drop_duplicates(keep='first')
    affairs = affairs.set_index('AffairShortId')
    affairs = affairs.sort_index()
    
    # Create an empty data frame with the affairs for indices
    asids = pd.DataFrame(index=affairs.index)
    
    # Assign numeric values to the votes
    def assign_val_to_vote(df):
        """ (Yes -> 1. | No -> -1. | Others -> 0.) """
        df['VoteValue'] = 0.
        df.loc[df.CouncillorYes == 1, 'VoteValue'] = 1.
        df.loc[df.CouncillorNo == 1, 'VoteValue'] = -1.
        return df
    
    voting_data = assign_val_to_vote(voting_data)
    voting_data = voting_data[['CouncillorId','AffairShortId', 'VoteValue']] 
    voting_data = voting_data.set_index('AffairShortId')
     
    def get_councillor_votes(df, cid, asids):
        """ Extract voting vector by CouncillorId """
        votes = df.loc[df.CouncillorId == cid, :]
        votes = votes.drop(columns=['CouncillorId'])
        votes = votes[~votes.index.duplicated(keep='first')]
        votes = asids.join(votes)
        votes = votes.rename(columns={'VoteValue':cid})
        return votes
    
    # Assemble a voting matrix with rows indexed by councillors and columns by affairs
    voting_matrix = asids.copy()
    
    for cid in councillors.index:
        votes = get_councillor_votes(voting_data, cid, asids)
        voting_matrix = pd.concat([voting_matrix, votes], axis=1)

    voting_matrix = voting_matrix.T
    voting_matrix.index.name = councillors.index.name
    voting_matrix.columns.name = affairs.index.name
    voting_matrix = voting_matrix.sort_index()
    
    voting_matrix = voting_matrix.fillna(0.0)
    
    # Save files
    councillors.to_csv('councillors.csv')
    affairs.to_csv('affairs.csv')
    voting_matrix.to_csv('voting_matrix.csv', index=False)