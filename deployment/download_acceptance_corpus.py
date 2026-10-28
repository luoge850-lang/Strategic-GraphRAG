"""Download official TSMC annual reports without replacing any existing PDF."""
import hashlib,json,time,urllib.request
from pathlib import Path
from deployment.acceptance_20261015 import ROOT, OUT, save
def main():
    import fitz
    target=OUT/'sources'; target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for year,url in [(2023,'https://investor.tsmc.com/sites/ir/annual-report/2023/2023%20Annual%20Report_E.pdf'),(2024,'https://investor.tsmc.com/sites/ir/annual-report/2024/2024%20Annual%20Report-E.pdf')]:
        dest=target/f'TSMC-{year}-Annual-Report.pdf'; start=time.perf_counter()
        original_url=url; attempts=[]
        try:
            response=urllib.request.urlopen(url,timeout=120)
        except Exception as exc:
            attempts.append(dict(url=url,error=str(exc)))
            url=f'https://www.annualreports.com/HostedData/AnnualReportArchive/t/NYSE_TSM_{year}.pdf'
            response=urllib.request.urlopen(url,timeout=120)
        with response:
            data=response.read(); headers=dict(response.headers)
        if not data.startswith(b'%PDF'):raise ValueError('not a PDF')
        with dest.open('xb') as f:f.write(data)
        doc=fitz.open(dest)
        rows.append(dict(company_id='TSMC',fiscal_year=year,document_type='Taiwan annual report, NOT SEC Form 20-F',
            url=url,official_url=original_url,attempts=attempts,mirror_identity='requires PDF cover verification; not asserted byte-identical to inaccessible official file',path=str(dest),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),pages=len(doc),
            download_seconds=time.perf_counter()-start,headers=headers,currency='TWD; convenience USD columns must be separately identified',
            submission_date=None,submission_date_status='not verified; publication/print date is not SEC submission',
            permission='Publicly downloadable; copyrighted TSMC, internal research only; redistribution rights not established'))
        with (target/f'TSMC-{year}-pages.json').open('x',encoding='utf-8') as f:
            json.dump([dict(page=i+1,text=p.get_text()) for i,p in enumerate(doc)],f,ensure_ascii=False)
    save('new-source-downloads.json',rows)
    pages=[]
    for p in [ROOT/'data/pdfs_other/2023-10-K.pdf',ROOT/'data/pdfs_other/2024-10-K.pdf',ROOT/'data/pdfs/2025-10-K.pdf']:
        d=fitz.open(p)
        for i,page in enumerate(d):
            text=page.get_text()
            if 'TSMC' in text or 'Taiwan Semiconductor' in text:
                pages.append(dict(file=p.name,page=i+1,text=text,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    save('independent-relationship-source-pages.json',pages)
if __name__=='__main__':main()
