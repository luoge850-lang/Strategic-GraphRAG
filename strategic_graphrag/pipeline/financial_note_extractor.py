"""Explicit note/prose and statement-row layouts omitted by the table backend.

No page numbers or reference values. Require issuer, monetary unit, source
header periods and complete value/column cardinality before emitting a fact.
"""
import json,re
from .financial_context import describe,dated_columns
from .financial_table_extractor import _periods_for_evidence

TOKEN=r'\(?-?\d[\d,]*(?:\.\d+)?\)?|[—–]'

def amounts(text):
    return [('0' if s in {'—','–'} else ('-' if s.startswith('(') else '')+s.strip('()').replace(',','')) for s in re.findall(TOKEN,text)]

def extract(text,company,filename,filing_year):
    out=[];lines=[s.strip() for s in text.splitlines() if s.strip()]
    def emit(metric,label,row,years,values,table,statement,proof=None):
        if not years or len(years)!=len(values) or len(set(years))!=len(years):return
        context=describe(text,table,row,statement)
        if proof:context.update(financial_basis='ACTUAL_REPORTED',basis_evidence=proof)
        if context['financial_basis']!='ACTUAL_REPORTED':return
        dates=dated_columns(text,years)
        pv=[dict(period=str(y),value=v,**dates.get(y,{})) for y,v in zip(years,values)]
        out.append(dict(source=company,company_id=company,source_category='Company',target=metric,target_category='FinancialMetric',relation='REPORTS_METRIC',causal_strength='DISCLOSED_ONLY',causal_form='FINANCIAL_RELATION',source_filing=filename,report_period=f'FY{filing_year}',evidence_sentence=row,row_evidence=row,row_label=label,table_name=table,table_id=table,metric_values_json=json.dumps(pv),metric_unit='USD millions',unit='USD millions',currency='USD',scale='millions',comparability_status='SAME_TABLE',column_semantics='PERIOD_COLUMNS',**context))
    is_note=bool(re.search(r'Notes to the Consolidated Financial Statements',text,re.I))
    # Respect the explicitly paired 'respectively' construction in prose.
    prose=re.search(r'(?:Our\s+)?contribution expense for fiscal years?\s+((?:20\d{2}[,\s]*(?:and\s*)?)+)was\s+((?:\$[\d,.]+\s+(?:million|billion)[,\s]*(?:and\s*)?)+),?\s+respectively',re.sub(r'\s+',' ',text),re.I)
    if is_note and prose and re.search(r'employee retirement plans|defined contribution plans',text,re.I):
        years=[int(y) for y in re.findall(r'20\d{2}',prose.group(1))]
        values=[str(float(v.replace(',',''))*(1000 if u.lower()=='billion' else 1)) for v,u in re.findall(r'\$([\d,.]+)\s+(million|billion)',prose.group(2),re.I)]
        emit('RETIREMENT_CONTRIBUTION_EXPENSE','contribution expense',prose.group(),years,values,'Employee Retirement Plans','FINANCIAL_NOTE_TABLE','Notes to the Consolidated Financial Statements; employee retirement contribution expense')
    in_millions=bool(re.search(r'\(In millions(?:,|\))',text,re.I))
    if not in_millions:return out
    cash=bool(re.search(r'Consolidated Statements of Cash Flows',text,re.I));income=bool(re.search(r'(?im)^Consolidated Statements of Income\s*$',text))
    tax_state=None
    for line in lines:
        if re.fullmatch(r'Current(?: income taxes)?:?\s*',line,re.I):tax_state='current'
        if re.fullmatch(r'Deferred(?: income taxes)?:?\s*',line,re.I):tax_state='deferred'
        metric=label=None
        if cash:
            for label_,metric_ in [('Stock-based compensation expense','STOCK_BASED_COMPENSATION_CASH_FLOW_ADJUSTMENT'),('Dividends paid','DIVIDENDS_PAID')]:
                if re.match(re.escape(label_)+r'\s+[$(\d—–-]',line,re.I):label,metric=label_,metric_
        if income:
            for label_,metric_ in [('Other, net','OTHER_INCOME_NET'),('Other income (expense), net','TOTAL_OTHER_INCOME_NET'),('Acquisition termination cost','ACQUISITION_TERMINATION_COST')]:
                if re.match(re.escape(label_)+r'\s+[$(\d—–-]',line,re.I):label,metric=label_,metric_
        if is_note and re.search(r'income tax expense|income taxes',text,re.I):
            if tax_state=='current' and re.match(r'Federal\s+[$(\d—–-]',line):label,metric='Federal','CURRENT_FEDERAL_INCOME_TAX'
            elif re.match(r'Total deferred\s+[$(\d—–-]',line,re.I):label,metric='Total deferred','TOTAL_DEFERRED_INCOME_TAX'
        if metric:
            tail=line[len(label):];years=_periods_for_evidence(text,line,filing_year);values=amounts(tail)
            # More numbers mean an unresolved change/percentage column; omit.
            if len(values)!=len(years):continue
            emit(metric,label,line,years,values,'Consolidated Statements of Cash Flows' if cash else 'Consolidated Statements of Income' if income else 'Income Taxes', 'CASH_FLOW_STATEMENT' if cash else 'INCOME_STATEMENT' if income else 'FINANCIAL_NOTE_TABLE')
    # Segment-by-column layout: each row has one year and four segment values.
    # Read the actual header and fiscal-year marker; never treat segments as years.
    if is_note and re.search(r'Compute\s*(?:&|and)\s*Networking',text,re.I) and re.search(r'Graphics\s+All Other\s+Consolidated',text,re.I):
        year=None;header='Compute & Networking | Graphics | All Other | Consolidated'
        for line in lines:
            m=re.fullmatch(r'Year Ended\s+.+?\b(20\d{2})',line,re.I)
            if m:year=int(m.group(1));continue
            m=re.match(r'Operating income \(loss\)\s+(.+)',line,re.I)
            if year and m:
                vals=amounts(m.group(1))
                if len(vals)==4:emit('GRAPHICS_OPERATING_INCOME','Operating income (loss)',line,[year],[vals[1]],'Segment Information / Graphics','FINANCIAL_NOTE_TABLE',header)
    return out
