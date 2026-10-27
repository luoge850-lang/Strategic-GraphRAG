"""Resume a preserved loopback trial, without changing config, credentials or data."""
import argparse,json,os,socket,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state',type=Path,required=True);p.add_argument('--distribution',type=Path,required=True)
    p.add_argument('--java-home',type=Path,required=True);p.add_argument('--record',type=Path,required=True)
    a=p.parse_args();state=a.state.resolve();distribution=a.distribution.resolve()
    c=json.loads((state/'private_credentials.json').read_text())
    if not c['uri'].startswith('bolt://127.0.0.1:'):raise ValueError('not owned loopback scope')
    config=(state/'conf/neo4j.conf').read_text()
    settings=dict(line.split('=',1) for line in config.splitlines() if '=' in line and not line.startswith('#'))
    for key,child in [('server.directories.data','data'),('server.directories.run','run')]:
        if Path(settings[key]).resolve()!=state/child:raise ValueError('state path mismatch')
    port=int(c['uri'].rsplit(':',1)[1])
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',port))==0:raise ValueError('already listening; will not replace process')
    if a.record.exists():raise ValueError('record exists')
    env=dict(os.environ,JAVA_HOME=str(a.java_home.resolve()),NEO4J_CONF=str(state/'conf'))
    log=state/('resume-'+str(time.time_ns())+'.log')
    with log.open('xb') as f:
        proc=subprocess.Popen([str(distribution/'bin/neo4j.bat'),'console'],cwd=distribution,env=env,
            stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    from neo4j import GraphDatabase
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        if proc.poll() is not None:raise RuntimeError('resume exited; see private local log')
        try:
            with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password']),connection_timeout=2) as d:
                d.verify_connectivity()
                with d.session(database=c['database']) as session:
                    builds=[dict(r) for r in session.run('MATCH (n) WHERE n.build_id IS NOT NULL RETURN n.build_id AS build_id, count(n) AS nodes')]
            with a.record.open('x',encoding='utf-8') as f:json.dump(dict(status='READY',pid=proc.pid,uri=c['uri'],builds=builds,config_mutations=0),f,indent=2)
            print('READY',c['uri']);return
        except Exception:time.sleep(1)
    raise RuntimeError('resume timeout; preserved data not overwritten')

if __name__=='__main__':main()
