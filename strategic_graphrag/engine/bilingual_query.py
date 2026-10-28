"""Bounded Chinese financial vocabulary; preserve unknown text for refusal.

No numbers or question-specific answers belong here. Unsupported clauses are
not silently dropped: the planner marks remaining Chinese as unresolved.
"""
import re

TERMS={
 '营业总收入':'revenue','營收':'revenue','营收':'revenue','收入':'revenue',
 '净利润率':'net margin','净利率':'net margin','收益率':'yield',
 '分别':'','以及':'and','请用':'in','以':'in','用':'in','表示':'','告诉我':'what','给出':'what',
 '投资活动现金净流出':'net cash used in investing activities',
 '经营活动现金净流入':'net cash provided by operating activities',
 '资本支出':'capital expenditure',
 '非合并':'unconsolidated','母公司单体':'standalone','客户专属收入':'customer specific revenue',
 '英伟达':'NVIDIA','英偉達':'NVIDIA','台积电':'TSMC','台積電':'TSMC',
 '苹果':'Apple','蘋果':'Apple','超微':'AMD',
 '税前利润':'pretax income','稅前利潤':'pretax income','营业收入':'revenue','營業收入':'revenue',
 '收入成本':'cost of revenue','毛利率':'gross margin','毛利润':'gross profit','毛利':'gross profit',
 '营业利润':'operating income','營業利益':'operating income','净利润':'net income','淨利潤':'net income',
 '所得税费用':'income tax expense','营业费用':'operating expenses',
 '流动资产合计':'total current assets','流动资产':'total current assets','总资产':'total assets',
 '流动负债合计':'total current liabilities','流动负债':'total current liabilities','总负债':'total liabilities',
 '留存收益':'retained earnings','保留盈余':'retained earnings','权益总额':'total equity','总权益':'total equity',
 '现金及现金等价物':'cash and cash equivalents','现金余额':'cash balance',
 '经营活动产生的现金流量净额':'net cash provided by operating activities',
 '投资活动使用的现金流量净额':'net cash used in investing activities',
 '筹资活动使用的现金流量净额':'net cash used in financing activities',
 '应收账款':'accounts receivable','应付账款':'accounts payable','存货':'inventories',
 '研发费用':'research and development','总额':'consolidated','合并':'consolidated',
 '第四季度':'Q4','第三季度':'Q3','第二季度':'Q2','第一季度':'Q1',
 '年报':'annual report','披露':'disclosed','美元':'USD','新台币':'TWD',
 '十亿美元':'USD billions','百万美元':'USD millions','千元新台币':'TWD thousands',
 '增长率':'growth rate','增长百分比':'percent change','绝对差额':'absolute difference','差额':'absolute difference',
 '除以':'divided by','计算':'calculate','比较':'compare','与':'and','和':'and',
 '客户收入':'customer revenue','客户':'customer','分部':'segment','季度':'quarter',
 '供应':'supply','依赖':'dependency','风险':'risk','地震':'earthquake',
 '截至':'balance at','年末':'year end','是多少':'what','多少':'what','请问':'what',
}

def normalize_financial_question(question):
    if not re.search(r'[\u3400-\u9fff]',question):return question
    text=question
    # Keep spending ambiguous: net investing cash flow is not gross investment
    # spending or capital expenditure. The planner asks which measure is meant.
    text=re.sub(r'投资(?:花了|用了)?多少现金', ' cash spent on investing ', text)
    # Do years before stripping grammar; disclosure-year and fact-year positions
    # remain available to the ordinary version parser.
    text=re.sub(r'(20\d{2})\s*(?:财年|財年|年)',r' fiscal \1 ',text)
    for token,value in sorted(TERMS.items(),key=lambda x:-len(x[0])):
        text=text.replace(token,' '+value+' ')
    text=re.sub(r'[的了是？?，,。；;：:]', ' ',text)
    return re.sub(r'\s+',' ',text).strip()
