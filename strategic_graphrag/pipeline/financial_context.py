"""Source-derived financial context. Missing evidence stays UNKNOWN."""
import re
from datetime import datetime,timedelta


def describe(text,table,row,statement):
    lines=[x.strip() for x in text.splitlines() if x.strip()]
    local=' '.join([table]+[x for x in lines if re.search(r'consolidated|pro[ -]?forma|non[- ]gaap|^note\s+\d|percentage of revenue',x,re.I)])
    if re.search(r'pro[ -]?forma',table,re.I):basis='PRO_FORMA';proof=table
    elif re.search(r'\badjusted\b|non[- ]gaap',table,re.I):basis='ADJUSTED';proof=table
    else:
        proof=next((x for x in lines if re.search(r'consolidated (?:statements|balance sheets)|^Consolidated$',x,re.I)),'')
        basis='ACTUAL_REPORTED' if proof else 'UNKNOWN'
    if statement=='FINANCIAL_TABLE':
        if re.search(r'percentage of revenue|percent of revenue',text,re.I):statement='COMMON_SIZE_INCOME_TABLE'
        elif re.search(r'(?im)^note\s+\d+',text):statement='FINANCIAL_NOTE_TABLE'
        elif re.search(r'(?im)^(Operating Expenses|Other Income \(Expense\), Net|Liquidity and Capital Resources|Results of Operations)',text):statement='MD_AND_A_FINANCIAL_TABLE'
    section=('FINANCIAL_STATEMENTS' if statement in {'BALANCE_SHEET','INCOME_STATEMENT','CASH_FLOW_STATEMENT','COMPREHENSIVE_INCOME_STATEMENT'} else
             'FINANCIAL_NOTES' if 'NOTE' in statement else 'MD_AND_A' if statement in {'MD_AND_A_FINANCIAL_TABLE','COMMON_SIZE_INCOME_TABLE'} else 'UNKNOWN')
    # Notes/MD&A are disclosed actuals only where a dated monetary table is
    # present and no pro-forma/adjusted heading qualifies that table.
    if basis=='UNKNOWN' and section in {'FINANCIAL_NOTES','MD_AND_A'} and re.search(r'Year[s]? Ended|Jan(?:uary)?\s+\d|Unit: NT\$',text,re.I):
        basis='ACTUAL_REPORTED';proof=next((x for x in lines if re.search(r'Year[s]? Ended|^note\s+\d|^Operating Expenses|^Liquidity and Capital',x,re.I)),'')
    return dict(statement_type=statement,source_section=section,financial_basis=basis,basis_evidence=proof,
                table_context='\n'.join(lines[:12])+('\n'+row),source_row=row)


def dated_columns(text,years,*,balance=False):
    dates={}
    for m in re.finditer(r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?[ \t]+(\d{1,2}),?[ \t]+(20\d{2})\b',text,re.I):
        try:d=datetime.strptime(m.group(1)[:3].title()+' '+m.group(2)+' '+m.group(3),'%b %d %Y').date()
        except ValueError:continue
        if d.year in years:dates.setdefault(d.year,(d,m.group(0)))
    result={}
    for year in years:
        if year not in dates:continue
        end,header=dates[year]
        start=end if balance else dates[year-1][0]+timedelta(days=1) if year-1 in dates else None
        result[year]=dict(column_label=header,period_end=end.isoformat(),period_start=start.isoformat() if start else 'UNKNOWN',period_bounds_status='SOURCE_END_AND_PRIOR_END' if start else 'SOURCE_END_ONLY')
    return result
