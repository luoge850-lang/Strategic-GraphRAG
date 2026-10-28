"""Narrow source-pattern extraction, never company-name co-occurrence edges."""
import re
def extract(text,company_id):
    flat=re.sub(r'\s+',' ',text)
    out=[]
    if company_id=='NVIDIA_CORPORATION':
        m=re.search(r'We utilize foundries, such as Taiwan Semiconductor.{0,220}?TSMC.{0,180}?to produce our semiconductor wafers\.',flat)
        if m:out.append(dict(source=company_id,target='TSMC',target_category='Company',relation='DEPENDS_ON_FOUNDRY',evidence_sentence=m.group(0),causal_form='DISCLOSED_DEPENDENCY'))
    if company_id=='TSMC':
        m=re.search(r'If one or more natural disasters result in a prolonged disruption.{0,700}?financial performance\.',flat)
        if m:out.append(dict(source=company_id,target='NATURAL_DISASTER',target_category='RiskFactor',relation='EXPOSED_TO',evidence_sentence=m.group(0),causal_form='CONDITIONAL_RISK'))
    for row in out:row.update(source_category='Company',causal_strength='DISCLOSED_ONLY',extraction_method='EXPLICIT_COMPANY_DISCLOSURE_PATTERN')
    return out
