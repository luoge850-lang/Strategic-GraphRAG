"""Final development repair; retain first build and its failed regression."""
import argparse,shutil
from deployment import request_contract_20261015 as campaign
FIRST=campaign.OUT
campaign.BASE=FIRST;campaign.OUT=FIRST/'round2';campaign.PACKAGE=campaign.OUT/'package';campaign.PORT=8008

def build():
    campaign.build()
    shutil.copyfile(FIRST/'development-labels.json',campaign.OUT/'development-labels.json')
    campaign.save('repair-record.json',dict(round=2,first_build='build_0c00d87603cc8f9a',failure='TSMC FY2023 retained earnings in the FY2024 disclosure was incorrectly refused because cross-filing UI erased the question version before retrieval; redundant candidate citations were all presented as answer facts.',repair='Explicit question version constrains retrieval. Selected answer observations separated from candidate_observations; original evidence and first results preserved.',partition='DEVELOPMENT_REGRESSION_ONLY',no_new_validation_claim=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','import','serve','evaluate']);a=p.parse_args()
    {'build':build,'import':campaign.import_graph,'serve':lambda:campaign.serve(8008),'evaluate':campaign.evaluate}[a.phase]()
