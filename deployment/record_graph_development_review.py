"""AI source reading of the first 18 nonnumeric 2024 candidates, development."""
import json
from deployment.acceptance_20261015 import OUT,save
def main():
    cp=json.loads((OUT/'five-package/checkpoints/2024-10-K.pdf.json').read_text(encoding='utf-8'))
    candidates=[x for x in cp['result']['accepted_triples'] if x['relation']!='REPORTS_METRIC'][:18]
    reasons={5:'Wrong subject: source describes NVIDIA business regulation, not United States risk exposure.',
      6:'Opposite direction: additional costs supports increase, not decrease.',
      7:'May occur if: conditional risk misclassified CONFIRMED_CAUSAL.',
      8:'May occur if: conditional risk misclassified CONFIRMED_CAUSAL.',
      9:'Lower capital expenditure by businesses does not directly support macro conditions increasing NVIDIA CAPEX.',
      10:'Currency fluctuation mention does not directly support increasing CAPEX.',
      11:'Regulation causes potential capex; supply-chain phrase is not the attributed cause.',
      12:'May disrupt and cause: conditional statement misclassified CONFIRMED_CAUSAL.',
      13:'May / could: conditional statement misclassified CONFIRMED_CAUSAL.',
      14:'Could lead: conditional statement misclassified CONFIRMED_CAUSAL.',
      15:'Potential gaming-sales effect; conditional and product scope must remain explicit.',
      16:'May: conditional statement misclassified CONFIRMED_CAUSAL.',
      17:'Opposite direction: additional costs supports increase, not decrease.'}
    reviewed=[]
    for i,c in enumerate(candidates):
        reviewed.append(dict(index=i,candidate=c,review_grade='AI_PDF_DEVELOPMENT_DIAGNOSIS_NOT_HUMAN_GOLD',
            relation_direction_direct_support=i not in {5,6,9,10,11,17},
            complete_semantic_joint=i not in reasons,reason=reasons.get(i,'Company product/market statement directly supports disclosed relation.')))
    save('graph-development-review.json',dict(frame='first 18 nonnumeric candidates of 2024 filing; convenience diagnosis, not population estimate',rows=reviewed,
         relation_support=dict(numerator=12,denominator=18,unjudged=0),joint=dict(numerator=5,denominator=18,unjudged=0)))
if __name__=='__main__':main()
