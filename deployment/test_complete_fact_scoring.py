import copy
import unittest
from complete_fact_scoring import score_fact, equal_number, citation_page_score

class FalsePassGuards(unittest.TestCase):
    def setUp(self):
        self.label = dict(company='NVIDIA_CORPORATION', metric='REVENUE', fact_period='FY2025',
            source_filing='2025-10-K.pdf', currency='USD', source_scale='millions',
            source_value=100, answer_value=100, answer_unit='USD millions', tolerance='0',
            direct_locations=[dict(file='2025-10-K.pdf', page=52, statement='INCOME_STATEMENT')],
            row_anchors=['Revenue'], label_tier='SYNTHETIC_SCORER_TEST_ONLY')
        self.response = dict(calculation=dict(status='PASS', value=100, unit='USD millions',
            observations=[dict(company='NVIDIA_CORPORATION', metric='REVENUE', fact_period='FY2025',
                source_filing='2025-10-K.pdf', currency='USD', scale='millions', page=52,
                statement_type='INCOME_STATEMENT', evidence_id='e1', raw_row='Revenue 100')]),
            citations=[dict(source_filing='2025-10-K.pdf', page=52, evidence_id='e1')])
    def test_valid_fixture(self):
        self.assertTrue(score_fact(self.response, self.label)['joint'])
    def test_wrong_dimensions(self):
        for field, value in [('metric','OPERATING_COST'),('fact_period','FY2024'),
                             ('source_filing','2024-10-K.pdf'),('statement_type','FINANCIAL_STATEMENTS')]:
            with self.subTest(field=field):
                r=copy.deepcopy(self.response);r['calculation']['observations'][0][field]=value
                self.assertFalse(score_fact(r,self.label)['joint'])
    def test_wrong_citation(self):
        for field,value in [('page',53),('evidence_id','unrelated')]:
            r=copy.deepcopy(self.response);r['citations'][0][field]=value
            self.assertFalse(score_fact(r,self.label)['joint'])
    def test_background_is_not_direct(self):
        out=citation_page_score(self.response,{'2025-10-K.pdf#52':1,'2025-10-K.pdf#53':3})
        self.assertFalse(out['direct_page_hit']);self.assertTrue(out['background_page_hit'])
    def test_fragment_missing_fact(self):
        r=copy.deepcopy(self.response);r['calculation']['observations'][0]['raw_row']='Revenue overview'
        self.assertFalse(score_fact(r,self.label)['joint'])
    def test_finite_and_precision(self):
        for x in ['NaN','Infinity',None]:self.assertFalse(equal_number(x,100,'0.5'))
        self.assertFalse(equal_number(100.6,100,'0.5'))
        self.assertTrue(equal_number(-2,-2,'0'));self.assertTrue(equal_number(0,0,'0'))
    def test_semantic_alias_not_cross_metric(self):
        label=copy.deepcopy(self.label);label['metric']='SALES_GENERAL_ADMIN_EXPENSE'
        r=copy.deepcopy(self.response);r['calculation']['observations'][0]['metric']='SG_AND_A_EXPENSE'
        self.assertTrue(score_fact(r,label)['metric'])
        r['calculation']['observations'][0]['metric']='OPERATING_COST'
        self.assertFalse(score_fact(r,label)['metric'])

if __name__=='__main__':unittest.main()
