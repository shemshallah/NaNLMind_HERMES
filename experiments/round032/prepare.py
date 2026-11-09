import json,random,argparse,pathlib
p=argparse.ArgumentParser();p.add_argument('corpus');p.add_argument('--out',default='data');p.add_argument('--limit',type=int,default=60000)
a=p.parse_args();o=pathlib.Path(a.out);o.mkdir(parents=True,exist_ok=True)
records={}; lines=open(a.corpus,encoding='utf8')
for line in lines:
 try: r=json.loads(line)
 except ValueError:continue
 if r.get('pos')!='noun':continue
 records[r['id']]=r
 if len(records)>=a.limit:break
parents={}
for key,rec in records.items():
 arr=[]
 for ptr in rec.get('pointers',[]):
  if ptr[0] in ('@','@i'):
   pid='n'+str(ptr[1]).zfill(8)
   if pid in records:arr.append(pid)
 parents[key]=arr
names={k:i+1 for i,k in enumerate(sorted(records))}
with (o/'facts.tsv').open('w') as f:
 for child,pp in parents.items():
  for parent in pp:f.write(f'{names[child]} 1 {names[parent]}\n')
chains=[]
for child,pp in parents.items():
 for middle in pp:
  for ancestor in parents[middle]:
   if ancestor not in pp and ancestor!=child:chains.append((names[child],1,names[ancestor]))
random.Random(7).shuffle(chains)
with (o/'queries.tsv').open('w') as f:
 for row in chains[:2000]:f.write(' '.join(map(str,row))+'\n')
print(json.dumps({'synsets':len(records),'facts':sum(map(len,parents.values())),'two_hop_queries':min(2000,len(chains))}))
